import numpy as np
import tensorflow as tf
import gym
import sonnet as snt
import random
import os
from utils import CNN, DDQN, QAgent, ReplayBuffer, DummySummary
from utils import mk_gif_image_summary
from utils import _array_to_gif


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("seed", 42, "Randomization seed")
tf.flags.DEFINE_float("learning_rate", .01, "Learning rate")
tf.flags.DEFINE_float("reward_discount", .99, "Reward discount")
tf.flags.DEFINE_integer("update_freq", 20, "Steps to update target Q")
tf.flags.DEFINE_float("update_frac", .01, "update fraction")
tf.flags.DEFINE_integer("num_episode", 20000, "Number of episode to train")
tf.flags.DEFINE_integer("max_episode_length", 1000, "max number of steps per episode")
tf.flags.DEFINE_float("epsilon_period", 1e6, "epsilon decay rate")
tf.flags.DEFINE_float("epsilon_max", 1., "epsilon max")
tf.flags.DEFINE_float("epsilon_min", .1, "epsilon min")
tf.flags.DEFINE_integer("replay_buffer_size", 500000, "Size of experience replay buffer")
tf.flags.DEFINE_integer("batch_size", 64, "Replay batch size")
tf.flags.DEFINE_integer("report_freq", 50, "Number of episode for reporting")
tf.flags.DEFINE_integer("test_freq", 100, "Freq for testing")
tf.flags.DEFINE_integer("test_episode", 50, "Number of testing episode")
tf.flags.DEFINE_string("save_path", "", "Path to save model")
tf.flags.DEFINE_string("summary_path", "", "Path to save summary")
tf.flags.DEFINE_integer("save_freq", 1000, "Freq to save model")
tf.flags.DEFINE_integer("record_freq", 10000, "Number of global steps to record a test episode")

np.random.seed(FLAGS.seed)
tf.set_random_seed(FLAGS.seed)


def PreprocessedAtariEnv():
    def __init__(self, name, buffer_size=4):
        self.env = gym.make(name)
        self.frame_buffer = None
        self.buffer_size = 4

    def __getattribute__(self, *args, **kwargs):
        if args[0] == "step":
            return self.step(*args[1:], **kwargs)
        return self.env.__getattribute__(*args, **kwargs)

    def _preprocess_frame(self, frame):
        frame = np.dot(frame, [.299, .587, .114])
        frame = scipy.ndimage.zoom(frame, [88./x for x in frame.shape])
        return frame

    def step(self, *args, **kwargs):
        s, r, is_end, misc = self.env.step(*args, **kwargs)
        s = self._preprocess_frame(s)
        if self.frame_buffer is None:
            self.frame_buffer = np.zeros(list(s.shape) + [self.buffer_size])
        self.frame_buffer[...,0] = s
        self.frame_buffer = np.roll(self.frame_buffer, -1, axis=-1)
        return (self.frame_buffer, r, is_end, misc)


def main():
    env = gym.make("CartPole-v0")
    #env = gym.make("MountainCar-v0")
    s = env.reset()
    with tf.name_scope("Environment") as scope:
        state = tf.placeholder(shape=[None] + list(s.shape), dtype=tf.float32, name="state")
        new_state = tf.placeholder(shape=[None] + list(s.shape), dtype=tf.float32, name="new_state")
        action = tf.placeholder(shape=[None, env.action_space.n], dtype=tf.float32, name="action")
        reward = tf.placeholder(shape=[None], dtype=tf.float32, name="reward")
        is_running = tf.placeholder(shape=[None], dtype=tf.float32, name="is_running")
    #global_step_op = tf.train.create_global_step()
    global_step_op = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    agent = QAgent(Model=DDQN, 
                   model_params=dict(output_size=env.action_space.n, 
                                     layer_size=[64,64]),
                   reward_discount = FLAGS.reward_discount,
                   update_frac = FLAGS.update_frac,
                   learning_rate = FLAGS.learning_rate,
                   )
    cost_op, trainer_op = agent(state=state, action=action, new_state=new_state, reward=reward, is_running=is_running, global_step=global_step_op)
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
        for num_ep in range(FLAGS.num_episode):
            s = env.reset()
            #s, r, is_end, _ = env.step(env.action_space.sample())
            total_reward = 0

            for step in range(FLAGS.max_episode_length):
                explore_prob = FLAGS.epsilon_min + \
                        (FLAGS.epsilon_max - FLAGS.epsilon_min) * \
                        max(0, (1 - global_step / FLAGS.epsilon_period))
                a = np.zeros(env.action_space.n, dtype=np.float32)
                if np.random.random() < explore_prob:
                    a[env.action_space.sample()]=1.
                else:
                    qa = sess.run(inference_op, {state: s[None,...]})[0]
                    a[np.argmax(qa)] = 1.
                s_prime, r, is_end, _ = env.step(np.argmax(a))
                total_reward += r
                if is_end:
                    s_prime = np.zeros_like(s_prime)
                replay_buff.add([s, a, s_prime, r, 1. - is_end])
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
                if global_step % FLAGS.update_freq == 0:
                    sess.run(agent_model_update_op)
                s = s_prime
                if is_end:
                    break
                    
            average_step += step
            average_reward += total_reward
            summary = tf.Summary()
            if num_ep % FLAGS.report_freq == 0:
                average_step /= FLAGS.report_freq
                average_reward /= FLAGS.report_freq
                print("Training Episode {} of {}, average survived {} of {} steps, average reward = {:.3f}, explore_prob={:.3f}".format(num_ep, FLAGS.num_episode, average_step, FLAGS.max_episode_length, average_reward, explore_prob))
                average_step = 0
                average_reward = 0
            if num_ep % FLAGS.test_freq == 0:
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
                    print(' *** images min: {}, max: {}, dtype: {}, shape: {} *** '.format(np.min(images), np.max(images), images.dtype, images.shape))
                    gif_summary = mk_gif_image_summary(images,
                                                        frame_rate=15)
                    summary.value.add(image = gif_summary,
                        tag="sample video at step {}".format(global_step))
                    print(" *** Recorded videos at step {}".format(global_step))
                summ_writer.add_summary(summary, global_step=global_step)
                print("Testing {} episodes: average survived {} of {} steps, average reward = {:.3f}".format(FLAGS.test_episode, test_step, FLAGS.max_episode_length, test_reward))

            if FLAGS.save_path != "" and num_ep % FLAGS.save_freq == 0:
                saver.save(sess, FLAGS.save_path, global_step=global_step)
                print(" *** model saved to {} at step {} ***".format(FLAGS.save_path, global_step))

            #record total_reward
if __name__ == "__main__":
    #test_replay_buffer()
    main()




        
        
        


