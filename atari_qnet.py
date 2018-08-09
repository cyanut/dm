import numpy as np
import tensorflow as tf
import gym
import sonnet as snt
import random
import os
import scipy
from utils import CNN, DDQN, QAgent, ReplayBuffer, DummySummary 
from utils import mk_gif_image_summary
from utils import _array_to_gif
from mpi_utils import MPI_IPool
import time
import enum
import time
from utils import frame_hash

from mpi4py import MPI
mp = MPI.COMM_WORLD
rank = mp.Get_rank()
mp_size = mp.Get_size()

import logging
logging.basicConfig(level=logging.DEBUG)


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("seed", 42, "Randomization seed")
tf.flags.DEFINE_float("learning_rate", .01, "Learning rate")
tf.flags.DEFINE_float("reward_discount", .99, "Reward discount")
tf.flags.DEFINE_integer("update_freq", 50, "Steps to update target Q")
tf.flags.DEFINE_float("update_frac", .01, "update fraction")
tf.flags.DEFINE_integer("num_episode", 20000, "Number of episode to train")
tf.flags.DEFINE_integer("num_steps", 200000, "Number of steps to train")
tf.flags.DEFINE_integer("max_episode_length", 36000, "max number of steps per episode")
tf.flags.DEFINE_float("epsilon_period", 1e5, "epsilon decay rate")
tf.flags.DEFINE_float("epsilon_max", 1., "epsilon max")
tf.flags.DEFINE_float("epsilon_min", .1, "epsilon min")
tf.flags.DEFINE_integer("replay_buffer_size", 10000, "Size of experience replay buffer")
tf.flags.DEFINE_integer("batch_size", 256, "Replay batch size")
tf.flags.DEFINE_integer("report_freq", 2000, "Number of episode for reporting")
tf.flags.DEFINE_integer("test_freq", 10000, "Freq for testing")
tf.flags.DEFINE_integer("test_episode", 50, "Number of testing episode")
tf.flags.DEFINE_string("save_path", "", "Path to save model")
tf.flags.DEFINE_string("summary_path", "", "Path to save summary")
tf.flags.DEFINE_integer("save_freq", 50000, "Freq to save model")
tf.flags.DEFINE_integer("record_freq", 10000, "Number of global steps to record a test episode")
tf.flags.DEFINE_integer("record_fps", 30, "Number of global steps to record a test episode")

np.random.seed(FLAGS.seed + rank)
tf.set_random_seed(FLAGS.seed + rank)

class COMM():
    RUN = 1
    ACTION = 2
    EXPERIENCE = 3
    STATS = 4
    '''
    RUN = enum.auto()
    ACTION = enum.auto()
    EXPERIENCE = enum.auto()
    GLOBAL_STEP = enum.auto()
    RECORDING = enum.auto()
    '''

class PreprocessedAtariEnv():
    def __init__(self, name, buffer_size=4):
        self.env = gym.make(name)
        s = self._preprocess_frame(self.env.reset())
        self.frame_buffer = np.zeros(list(s.shape) + [buffer_size])

    def __getattr__(self, *args, **kwargs):
        return self.env.__getattribute__(*args, **kwargs)

    def _preprocess_frame(self, frame):
        frame = frame / 255.
        frame = np.dot(frame, [.299, .587, .114])
        frame = scipy.ndimage.zoom(frame, [84./x for x in frame.shape]).clip(0)
        return frame

    def _reset_frame_buffer(self):
        self.frame_buffer = np.zeros_like(self.frame_buffer)

    def _append_frame(self, frame):
        frame = self._preprocess_frame(frame)
        self.frame_buffer[...,0] = frame
        self.frame_buffer = np.roll(self.frame_buffer, -1, axis=-1)

    def step(self, *args, **kwargs):
        s, r, is_end, misc = self.env.step(*args, **kwargs)
        self._append_frame(s)
        return (self.frame_buffer.copy(), r, is_end, misc)

    def reset(self, *args, **kwargs):
        s = self.env.reset(*args, **kwargs)
        self._reset_frame_buffer()
        self._append_frame(s)
        return self.frame_buffer.copy()


class CNN_DDQN(snt.AbstractModule):
    def __init__(self, conv_params, output_size, layer_size, max_pool_params={}, name="CNN-DDQN"):
        super(CNN_DDQN, self).__init__(name=name)
        self.conv_params = conv_params
        self.max_pool_params = max_pool_params
        self.output_size = output_size
        self.layer_size = layer_size

    def _build(self, state):
        cnn = CNN(conv_params = self.conv_params, 
                  max_pool_params = self.max_pool_params)
        ddqn = DDQN(output_size = self.output_size, 
                    layer_size = self.layer_size)
        outputs = ddqn(cnn(state))
        return outputs

def test_atari_env(fname):
    import pickle
    env = PreprocessedAtariEnv("Breakout-v0", buffer_size=3)
    res = []
    for i in range(5):
        s = env.reset()
        is_end = False
        while not is_end:
            s, r, is_end, misc = env.step(env.action_space.sample())
            res.append(s)
        print("reset env, step", len(res))
    gif = _array_to_gif(np.array(res), 20)
    with open(fname, "wb") as fo:
        fo.write(gif)
    

def exp_hash(res):
    return (frame_hash(res[0]), res[2, 0, :5, 0], frame_hash(res[1]), res[2,1,0,0], res[2,2,0,0])

def pack_array(s, a, s_prime, r, is_end):
    res = np.array([s, s_prime, np.zeros_like(s)])
    res[2, 0, :len(a), 0] = a
    res[2, 0, len(a), 0] = -1
    res[2, 1, 0, 0] = r
    res[2, 2, 0, 0] = is_end
    return res

def unpack_array(res):
    s = res[0]
    for i in range(res.shape[2]):
        if res[2, 0, i, 0] == -1:
            break
    a = res[2, 0, :i, 0]
    s_prime = res[1]
    r = res[2, 1, 0, 0]
    is_end = np.bool(res[2, 2, 0, 0])
    return (s, a, s_prime, r, is_end)

def env_worker(rank, env_name, global_step, root=0, buffer_size=4, temp_dir=None):
    logger = logging.getLogger("worker {}".format(rank))
    is_run = mp.recv(source=root, tag=COMM.RUN)
    assert is_run == 1
    logger.info("start to run.")
    run_req = mp.irecv(source=root, tag=COMM.RUN)
    root_comm = MPI_IPool(mp, sources=[root])
    env = PreprocessedAtariEnv(env_name, buffer_size)
    if temp_dir:
        recording = {"states":[], "actions":[]}
        all_expr = []
    expr_buf = []
    local_steps = []
    while is_run:
        logger.debug("************ reset env. **************")
        s = env.reset()
        local_steps.append(0)
        if temp_dir:
            recording["states"].append(s)
        episode_reward = 0

        for step in range(FLAGS.max_episode_length):
            local_steps[-1] += 1
            explore_prob = FLAGS.epsilon_min + \
                    (FLAGS.epsilon_max - FLAGS.epsilon_min) * \
                    max(0, (1 - global_step / FLAGS.epsilon_period))
            a = np.zeros(env.action_space.n)
            if np.random.random() < explore_prob:
                a[env.action_space.sample()]=1.
                if temp_dir:
                    recording["actions"].append(None)
            else:
                data = {"expr":expr_buf, "state":s}
                logger.debug("sending action requests with {} experiences.".format(len(expr_buf)))
                expr_req = mp.isend(data, dest=root, tag=COMM.ACTION)

                logger.debug("waiting for actions.")
                action_req = mp.irecv(source=root, tag=COMM.ACTION)
                target = 1
                while target == 1:
                    target, data = MPI.Request.waitany([action_req, run_req])
                    if target == 1:
                        is_run = data
                        if not is_run:
                            logger.debug("received run signal {}, terminating.".format(is_run))
                            break
                        else:
                            logger.debug("received run signal {}."\
                                    .format(is_run))
                            run_req = mp.irecv(is_run, source=root,
                                    tag=COMM.RUN)
                expr_buf = []
                if not is_run:
                    break
                a, global_step = data
                if temp_dir:
                    recording["actions"].append(a)
                
                logger.debug("action received {}, global_step={}, explore_prob={}.".format(a, global_step, explore_prob))
            s_prime, r, is_end, _ = env.step(np.argmax(a))
            global_step += 1
            episode_reward += r
            expr_buf.append([s, a, s_prime, r, is_end])
            if temp_dir:
                all_expr.append([s, a, s_prime, r, is_end])
                recording["states"].append(s_prime)
            s = s_prime
            if is_end:
                break
        if not is_run:
            break
        summary = {"reward":episode_reward, 
                   "episode_steps":step, 
                   "global_steps":global_step, 
                   "explore_prob":explore_prob,
                   }
        summary_copy = summary.copy()
        mp.send(summary_copy, dest=root, tag=COMM.STATS)
        logger.debug("sending summaries {}".format(summary))
                    
    
    if temp_dir:
        recording["all_expr"] = all_expr
        gif = _array_to_gif(np.array(recording["states"]), 20)
        with open(temp_dir+str(rank)+".orig.png", "wb") as fo:
            fo.write(gif)
        import pickle
        with open(temp_dir+str(rank)+".orig.pkl", "wb") as fo:
            pickle.dump(recording, fo)
    
    mp.send(False, dest=root, tag=COMM.RUN)
    logger.info("{}: terminated after {} - {} steps.".format(rank, local_steps, sum(local_steps)))



def test_env_worker(temp_dir):
    env = PreprocessedAtariEnv("Breakout-v0", buffer_size=3)
    env_name = "Breakout-v0"
    if rank > 0:
        env_worker(rank, env_name, 0, buffer_size=3, temp_dir=temp_dir)
        quit()
    
    s = env.reset()
    sample_frame = s
    n = env.action_space.n
    worker_pool = MPI_IPool(comm = mp, buf_size=2**24,
                                sources = list(range(1, mp_size)))
    worker_pool.ibcast_data(data=True, tag=COMM.RUN)
    action_accum = np.zeros(mp_size, dtype=int) + 1
    action_div = np.arange(mp_size, dtype=int) * 5 + n 
    recordings = {}
    global_step = 0
    steps = np.zeros(mp_size)
    all_expr = {i:[] for i in range(1, mp_size)}
    while global_step < FLAGS.num_steps:
        res = worker_pool.icollect_data(tag=COMM.ACTION)
        print("received", len(res[0]), "experiences from", res[1])
        for r, i in zip(*res):
            if r["expr"]:
                v = []
                for e in r["expr"]:
                    all_expr[i].append(e)
                    steps[i] += 1
                    v.append(e[0])
                    if e[4]: #if is_end
                        v.append(e[2])
                
                print("received from {}:{}".format(i, [frame_hash(x) for x in v]))
                if not i in recordings:
                    recordings[i] = {"states":v, "actions":[None]*len(v)}
                elif not "states" in recordings[i]:
                    recordings[i]["states"] = v
                    recordings[i]["actions"] += [None] * len(v)
                else:
                    recordings[i]["states"] += v
                    recordings[i]["actions"] += [None] * len(v)

        action_reqs = res[1]
        print("collected action_reqs:", action_reqs)
        if len(action_reqs) > 0:
            print("received action_reqs from", action_reqs)
            action_accum[action_reqs] += 1
            actions = ((action_accum % action_div).clip(0, n-1))[action_reqs]
            action_one_hot = np.eye(n)[actions]
            print("action_one_hot:", action_one_hot)
            time.sleep(.01)
            worker_pool.iscatter_data(list(zip(action_one_hot, [global_step]*action_one_hot.shape[0])), tag=COMM.ACTION, idx=action_reqs)
        for k, _r in enumerate(zip(*res)):
            r, i = _r
            if not i in recordings:
                recordings[i] = {"actions":[]}
            recordings[i]["actions"].append(action_one_hot[k].copy())
        global_step += len(res[0])
        print("global_step:", global_step)
        summary = worker_pool.icollect_data(tag=COMM.STATS)
        print("Received summary:", summary)
        if summary[0]:
            time.sleep(3)

    worker_pool.ibcast_data(data=False, tag=COMM.RUN)
    import pickle
    reqs = [mp.irecv(source=i, tag=COMM.RUN) for i in range(1, mp_size)]
    run_status = MPI.Request.waitall(reqs)
    assert np.all(np.array(run_status) == False)

    print("received steps:", steps)
    for i in range(1, mp_size):
        recordings[i]["all_expr"] = all_expr[i]
        recordings[i]["states"] = [x for x in recordings[i]["states"] if x is not None]
        with open(temp_dir+str(i)+".orig.pkl", "rb") as fi:
            orig_data = pickle.load(fi)
            ns = len(recordings[i]["states"]) 
            ns_orig = len(orig_data["states"])
            print("ns", ns, "rank", i, "ns_orig", ns_orig)
            assert ns_orig >= ns
            assert ns_orig - ns < 10
            assert np.allclose(np.array(orig_data["states"][:ns]), 
                               np.array(recordings[i]["states"]))

            e_orig = orig_data["all_expr"]
            e_recv = recordings[i]["all_expr"]
            assert 0 <= len(e_orig) - len(e_recv) < 10
            ne_min = min(len(e_orig), len(e_recv))
            j=0
            while j < ne_min and np.all([np.allclose(x, y) for x, y in zip(e_orig[j], e_recv[j])]):
                j+=1
            assert j == ne_min

            acts = [x for x in recordings[i]["actions"] if x is not None]
            acts_orig = [x for x in orig_data["actions"] if x is not None]
            print("acts", len(acts), "rank", i, "acts_orig", len(acts_orig))
            assert len(acts_orig) == len(acts)
            assert np.allclose(np.array(acts), np.array(acts_orig))
    #assert abs(global_step - steps) <= mp_size
    for i, v in recordings.items():
        gif = _array_to_gif(np.array(v["states"]), 20)
        with open(temp_dir+str(i)+".png", "wb") as fo:
            fo.write(gif)
        with open(temp_dir+str(i)+".pkl", "wb") as fo:
            pickle.dump(v, fo)
    print("main thread done, all tests passed")


def test_env_worker_min():
    buf_size = 3
    if rank > 0:
        env = PreprocessedAtariEnv("Breakout-v0", buffer_size=buf_size)
        s = env.reset()
        s = np.random.random(s.shape)
        mp.Isend(s, dest=0, tag=5)
        print(s.shape)
        res = [s]
        for i in range(rank+1):
            #s, r, is_end, misc = env.step(i%env.action_space.n)
            s = np.random.random(s.shape)
            mp.Isend(s, dest=0, tag=5)
            res.append(s)
        print("{} frames: {}".format(rank, [frame_hash(s) for s in res]))
    if rank == 0:
        env = PreprocessedAtariEnv("Breakout-v0", buffer_size=buf_size)
        s = env.reset()
        worker_pool = MPI_IPool(comm = mp, buf_size = 2**24,
                                sources = list(range(1, mp_size)))
        while True:
            print("one round")
            frames, action_reqs = worker_pool.Icollect_data(np.empty_like(s), tag=5)
            print(action_reqs)
            for i, f in zip(action_reqs, frames):
                print("root: frame {} from rank {}".format(frame_hash(f), i))
            time.sleep(1) 
            
    

   
def root():
    #Root
    logger = logging.getLogger("root")
    buf_size = 4
    worker_pool = MPI_IPool(comm = mp, buf_size = 2 ** 25,
            sources=[x for x in range(mp_size) if x != rank],)
    env = PreprocessedAtariEnv("Breakout-v0", buffer_size=buf_size)
    s = env.reset()
    with tf.name_scope("Environment") as scope:
        state = tf.placeholder(shape=[None] + list(s.shape), dtype=tf.float32, name="state")
        new_state = tf.placeholder(shape=[None] + list(s.shape), dtype=tf.float32, name="new_state")
        action = tf.placeholder(shape=[None, env.action_space.n], dtype=tf.float32, name="action")
        reward = tf.placeholder(shape=[None], dtype=tf.float32, name="reward")
        is_running = tf.placeholder(shape=[None], dtype=tf.float32, name="is_running")
    #global_step_op = tf.train.create_global_step()
    global_step_op = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    agent = QAgent(Model=CNN_DDQN, 
                   update_frac=FLAGS.update_frac,
                   learning_rate = FLAGS.learning_rate,
                   reward_discount = FLAGS.reward_discount,
                   model_params = dict(\
                       output_size = env.action_space.n,
                       layer_size = [512, 128],
                       conv_params = dict(\
                               kernel_shape = [8, 4, 3],
                               output_channels = [32, 64, 64],
                               stride = [4, 2, 1],
                       ),
                   ),
            )
    cost_op, trainer_op = agent(state=state, action=action, new_state=new_state, reward=reward, is_running=is_running, global_step=global_step_op)
    with tf.name_scope("Summary") as scope:
        cost_summary_op = tf.summary.scalar("cost", tf.reduce_mean(cost_op))
    agent_model_update_op = agent.update_target_network()
    summary_op = tf.summary.merge_all()
    inference_op = agent.inference(state)
    average_step = 0
    average_reward = 0
    global_step = 0
    replay_buff = ReplayBuffer(FLAGS.replay_buffer_size)
    saver = tf.train.Saver()
    if FLAGS.summary_path:
        summ_writer = tf.summary.FileWriter(FLAGS.summary_path)
    else:
        summ_writer = DummySummary()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        summ_writer.add_graph(sess.graph)
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(FLAGS.save_path))
        if ckpt and ckpt.model_checkpoint_path: 
            saver.restore(sess, ckpt.model_checkpoint_path)
        last_video_at_step = 0
        last_update_step = 0
        last_train_report_step = 0
        last_test_step = 0
        last_save_step = 0
        t = 0
        stats = {}
        worker_pool.ibcast_data(data=True, tag=COMM.RUN)
        while global_step < FLAGS.num_steps:
            inference_states = []
            res = worker_pool.icollect_data(tag=COMM.ACTION)
            #logger.debug("received {} experiences from {}".format(len(res[0]), res[1]))
            states = []
            expr = []
            for r, worker_rank in zip(*res):
                expr += r["expr"]
                states.append(r["state"])

            if states:
                qa = sess.run(inference_op, {state: np.array(states)})
                worker_pool.iscatter_data(\
                        list(zip(qa, [global_step]*qa.shape[0])),
                        tag=COMM.ACTION, 
                        idx=res[1])
            global_step += len(expr)
            for x in expr:
                replay_buff.add(x)
            if len(replay_buff.buffer) < 1:
                logger.info("populating replay buffer ...")
                continue
            t = time.time()
            s_batch, a_batch, s_prime_batch, r_batch, is_running_batch = \
                    replay_buff.sample(FLAGS.batch_size)
            _, loss, summary, global_step = sess.run(\
                    [trainer_op, cost_op, summary_op, global_step_op], 
                    {state: s_batch,
                     action: a_batch,
                     new_state: s_prime_batch,
                     reward: r_batch,
                     is_running: is_running_batch})
            summ_writer.add_summary(summary, global_step=global_step) 
            if global_step - last_update_step > FLAGS.update_freq:
                sess.run(agent_model_update_op)
                last_update_step = global_step
            logger.debug("GPU time: {.3f}".format(time.time() -t)

                
            worker_stats = worker_pool.icollect_data(tag=COMM.STATS)
            for d in worker_stats[0]:
                for k,v in d.items():
                    if not k in stats:
                        stats[k] = []
                    stats[k].append(v)

            if global_step - last_train_report_step > FLAGS.report_freq:
                last_train_report_step = global_step
                fmt_string = ["{} = {}".format(k, np.mean(v)) \
                        for k, v in stats.items()]
                logger.info("Training summary at global step {}, {}"\
                        .format(global_step, ", ".join(fmt_string)))
                stats = {}
            '''
            if num_ep % FLAGS.report_freq == 0:
                average_step /= FLAGS.report_freq
                average_reward /= FLAGS.report_freq
                logger.info("Training Episode {} of {}, average survived {} of {} steps, average reward = {:.3f}, explore_prob={:.3f}".format(num_ep, FLAGS.num_episode, average_step, FLAGS.max_episode_length, average_reward, explore_prob))
                average_step = 0
                average_reward = 0
            '''
            if global_step - last_test_step > FLAGS.test_freq:
                last_test_step = global_step
                summary = tf.Summary()
                test_reward = 0
                test_step = 0
                images = []
                for test_num_ep in range(FLAGS.test_episode):
                    recording = False
                    if global_step - last_video_at_step > FLAGS.record_freq:
                        recording = True
                        last_video_at_step = global_step
                    s = env.reset()
                    for step in range(FLAGS.max_episode_length):
                        if recording:
                            images.append(env.render(mode="rgb_array"))
                        a = sess.run(inference_op, {state: s[None,...]})[0]
                        s_prime, r, is_end, _ = env.step(np.argmax(a))
                        s = s_prime
                        test_reward += r
                        if is_end:
                            break
                    test_step += step

                test_reward /= FLAGS.test_episode
                test_step /= FLAGS.test_episode
                summary.value.add(tag="testing episode length", simple_value = test_step)
                summary.value.add(tag="testing reward per episode", simple_value = test_reward)
                if images:
                    images = np.array(images)
                    gif_summary = mk_gif_image_summary(images, 
                                   frame_rate = tf.FLAGS.record_fps)
                    summary.value.add(image = gif_summary, 
                        tag = "sample video at step {}".format(global_step))
                    logger.info(" *** Recorded {} frames at step {} *** ".format(\
                            images.shape[0], global_step))
                summ_writer.add_summary(summary, global_step=global_step)
                logger.info("Testing {} episodes: average survived {} of {} steps, average reward = {:.3f}".format(FLAGS.test_episode, test_step, FLAGS.max_episode_length, test_reward))

            if FLAGS.save_path != "" and \
                    global_step - last_save_step > FLAGS.save_freq:
                last_save_step = global_step
                saver.save(sess, FLAGS.save_path, global_step=global_step)
                logger.info(" *** model saved to {} at step {} ***".format(FLAGS.save_path, global_step))
                
        #FIXME
        time.sleep(1)
    worker_pool.ibcast_data(data=False, tag=COMM.RUN)

def main():
    env_name = "Breakout-v0"
    if rank > 0:
        env_worker(rank, env_name, 0)
    else:
        root()

            
if __name__ == "__main__":
    main()
    #import os
    #test_atari_env("/scratch/cyan/env_worker_test/env.png")
    #test_env_worker(os.environ["SLURM_TMPDIR"]+"/")
    #test_env_worker("/scratch/cyan/env_worker_test/")
    #test_env_worker_min()




        
        
        


