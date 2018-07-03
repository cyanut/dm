import numpy as np
import tensorflow as tf
import gym
import sonnet as snt
import random
import os


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

np.random.seed(FLAGS.seed)
tf.set_random_seed(FLAGS.seed)

class DQN(snt.AbstractModule):
    def __init__(self, output_size, layer_size=[64,64], name="DQN"):
        super(DQN, self).__init__(name=name)
        self.output_size = output_size
        self.layer_size=layer_size

    def _build(self, state):
        outputs = state
        for layer_size in self.layer_size:
            outputs = snt.Linear(output_size=layer_size)(outputs)
            outputs = tf.nn.relu(outputs)
        outputs = snt.Linear(output_size=self.output_size)(outputs)

        return outputs

class DDQN(snt.AbstractModule):
    def __init__(self, output_size, name="DDQN", layer_size=[64,64]):
        super(DDQN, self).__init__(name=name)
        self.output_size = output_size
        self.layer_size = layer_size

    def _build(self, state):
        a_net = DQN(output_size=self.output_size, layer_size=self.layer_size, name="advantage_network")
        v_net = DQN(output_size=1, layer_size=self.layer_size, name="value_network")
        a_output = a_net(state)
        a_output = a_output - tf.reduce_mean(a_output, axis=1, keepdims=True)
        outputs = a_output + v_net(state)
        return outputs


class QAgent(snt.AbstractModule):
    def __init__(self, Model, output_size, update_frac, name="Agent"):
        super(QAgent, self).__init__(name=name)
        self.Model = Model
        self.output_size = output_size
        self.update_frac = update_frac

    def _build(self, state, new_state, action, reward, is_running, global_step):
        self.q_net = self.Model(output_size = self.output_size, name="q_net")
        self.target_q_net = self.Model(output_size = self.output_size, name="q_target")
        q_target = is_running * tf.reduce_max(\
                self.target_q_net(state=new_state), axis=1)
        #q_target = is_running * tf.reduce_max(\
        #        self.q_net(state=new_state), axis=1)

        q_all = self.q_net(state=state)
        q = tf.reduce_sum(q_all * action, axis=1)
        cost = (q - reward - FLAGS.reward_discount * q_target) ** 2
        trainer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)\
                    .minimize(cost, 
                              var_list = self.q_net.get_all_variables(),
                              global_step=global_step)
        return cost, trainer 

    def update_target_network(self):
        ops = []
        for v_q, v_target_q in zip(self.q_net.get_all_variables(), 
                self.target_q_net.get_all_variables()):
            ops.append(v_target_q.assign(self.update_frac * v_q + (1 - self.update_frac) * v_target_q))
        return ops

    def inference(self, state):
        return self.q_net(state)

class ReplayBuffer:
    def __init__(self, size):
        self.buffer = []
        self.size = size
        
    def add(self, experience):
        self.buffer.append(experience)
        if len(self.buffer) > self.size:
            self.buffer.pop(0)

    def sample(self, sample_size):
        if sample_size > len(self.buffer):
            samples = self.buffer.copy()
        else:
            samples = random.sample(self.buffer, sample_size)
        random.shuffle(samples)
        return [np.array(x) for x in zip(*samples)]

def test_replay_buffer():
    rb = ReplayBuffer(10)
    for i in range(1):
        rb.add([np.arange(i*10, i*10+6).reshape((2,3)), i, i*2])
    print(rb.sample(3))
    for i in range(5, 20):
        rb.add([np.arange(i*10, i*10+6).reshape((2,3)), i, i*2])
    print(rb.sample(3))
    print(rb.sample(20))


class DummySummary():
    def __init__(self, *args):
        pass
    def add_summary(self, *args):
        pass
    def add(self, *args):
        pass
    def add_graph(self, *args):
        pass

def main():
    env = gym.make("CartPole-v0")
    #env = gym.make("MountainCar-v0")
    s = env.reset()
    state = tf.placeholder(shape=[None] + list(s.shape), dtype=tf.float32, name="state")
    new_state = tf.placeholder(shape=[None] + list(s.shape), dtype=tf.float32, name="new_state")
    action = tf.placeholder(shape=[None, env.action_space.n], dtype=tf.float32, name="action")
    reward = tf.placeholder(shape=[None], dtype=tf.float32, name="reward")
    is_running = tf.placeholder(shape=[None], dtype=tf.float32, name="is_running")
    #global_step_op = tf.train.create_global_step()
    global_step_op = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    agent = QAgent(Model=DDQN, output_size=env.action_space.n, 
            update_frac=FLAGS.update_frac)
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
                for test_num_ep in range(FLAGS.test_episode):
                    s = env.reset()
                    for step in range(FLAGS.max_episode_length):
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
                summ_writer.add_summary(summary, global_step=global_step)
                print("Testing {} episodes: average survived {} of {} steps, average reward = {:.3f}".format(FLAGS.test_episode, test_step, FLAGS.max_episode_length, test_reward))

            if FLAGS.save_path != "" and num_ep % FLAGS.save_freq == 0:
                saver.save(sess, FLAGS.save_path, global_step=global_step)
                print(" *** model saved to {} at step {} ***".format(FLAGS.save_path, global_step))

            #record total_reward
if __name__ == "__main__":
    #test_replay_buffer()
    main()




        
        
        


