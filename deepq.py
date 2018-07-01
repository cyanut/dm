import numpy as np
import tensorflow as tf
import gym
import sonnet as snt
import random


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("seed", 42, "Randomization seed")
tf.flags.DEFINE_float("learning_rate", .0001, "Learning rate")
tf.flags.DEFINE_float("reward_discount", .99, "Reward discount")
tf.flags.DEFINE_integer("update_freq", 1000, "Steps to update target Q")
tf.flags.DEFINE_integer("num_episode", 10000, "Number of episode to train")
tf.flags.DEFINE_integer("max_episode_length", 1000, "max number of steps per episode")
tf.flags.DEFINE_float("epsilon_decay", 400, "epsilon decay rate")
tf.flags.DEFINE_float("epsilon_max", 1., "epsilon max")
tf.flags.DEFINE_float("epsilon_min", .0001, "epsilon min")
tf.flags.DEFINE_integer("replay_buffer_size", 50000, "Size of experience replay buffer")
tf.flags.DEFINE_integer("batch_size", 64, "Replay batch size")
tf.flags.DEFINE_integer("report_freq", 50, "Number of episode for reporting")
tf.flags.DEFINE_integer("test_freq", 100, "Freq for testing")
tf.flags.DEFINE_integer("test_episode", 10, "Number of testing episode")

np.random.seed(FLAGS.seed)
tf.set_random_seed(FLAGS.seed)

class DQN(snt.AbstractModule):
    def __init__(self, output_size, name="DQN"):
        super(DQN, self).__init__(name=name)
        self.output_size = output_size

    def _build(self, state):
        outputs = snt.Linear(output_size=64)(state)
        outputs = tf.nn.relu(outputs)
        outputs = snt.Linear(output_size=64)(outputs)
        outputs = tf.nn.relu(outputs)
        outputs = snt.Linear(output_size=self.output_size)(outputs)

        return outputs


class QAgent(snt.AbstractModule):
    def __init__(self, Model, output_size, name="Agent"):
        super(QAgent, self).__init__(name=name)
        self.Model = Model
        self.output_size = output_size

    def _build(self, state, action, reward, is_running):
        self.q_net = self.Model(output_size = self.output_size)
        self.target_q_net = self.Model(output_size = self.output_size)
        q_target = is_running * tf.reduce_max(\
                self.target_q_net(state=state), axis=1)
        q_target = is_running * tf.reduce_max(\
                self.q_net(state=state), axis=1)

        q_all = self.q_net(state=state)
        q = tf.reduce_sum(q_all * tf.one_hot(action, self.output_size), axis=1)
        cost = (q - reward - FLAGS.reward_discount * q_target) ** 2
        print(self.q_net.get_all_variables())
        trainer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)\
                    .minimize(cost, var_list = self.q_net.get_all_variables())
        return cost, trainer 

    def update_target_network(self):
        ops = []
        for v_q, v_target_q in zip(self.q_net.get_all_variables(), 
                self.target_q_net.get_all_variables()):
            ops.append(v_target_q.assign(v_q))
        return ops

    def inference(self, state):
        return tf.argmax(self.q_net(state), axis=1)

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

def main():
    env = gym.make("CartPole-v0")
    s = env.reset()
    state = tf.placeholder(shape=[None] + list(s.shape), dtype=tf.float32)
    action = tf.placeholder(shape=[None], dtype=tf.int32)
    reward = tf.placeholder(shape=[None], dtype=tf.float32)
    is_running = tf.placeholder(shape=[None], dtype=tf.float32)
    agent = QAgent(Model=DQN, output_size=env.action_space.n)
    cost_op, trainer_op = agent(state=state, action=action,
                                reward=reward, is_running=is_running)
    agent_model_update_op = agent.update_target_network()
    inference_op = agent.inference(state)
    global_step = 0
    average_step = 0
    average_reward = 0
    replay_buff = ReplayBuffer(FLAGS.replay_buffer_size)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #sess.run(tf.local_variables_initializer())
        for num_ep in range(FLAGS.num_episode):
            s = env.reset()
            #s, r, is_end, _ = env.step(env.action_space.sample())
            total_reward = 0

            for step in range(FLAGS.max_episode_length):
                global_step += 1
                explore_prob = FLAGS.epsilon_min + \
                        (FLAGS.epsilon_max - FLAGS.epsilon_min) * \
                        np.exp(-num_ep / FLAGS.epsilon_decay)
                if np.random.random() < explore_prob:
                    a = env.action_space.sample()
                else:
                    a = sess.run(inference_op, {state: s[None,...]})[0]
                s_prime, r, is_end, _ = env.step(a)
                total_reward += r
                replay_buff.add([s, a, r, 1. - is_end])
                s_batch, a_batch, r_batch, is_running_batch = \
                        replay_buff.sample(FLAGS.batch_size)
                _, loss = sess.run([trainer_op, cost_op], 
                        {state: s_batch,
                         action: a_batch,
                         reward: r_batch,
                         is_running: is_running_batch})
                if global_step % FLAGS.update_freq == 0:
                    sess.run(agent_model_update_op)
                s = s_prime
                if is_end:
                    break
            average_step += step
            average_reward += total_reward
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
                        s_prime, r, is_end, _ = env.step(a)
                        s = s_prime
                        test_reward += r
                        if is_end:
                            break
                    test_step += step
                test_reward /= FLAGS.test_episode
                test_step /= FLAGS.test_episode
                print("Testing {} episodes: average survived {} of {} steps, average reward = {:.3f}".format(FLAGS.test_episode, test_step, FLAGS.max_episode_length, test_reward))

            #record total_reward
if __name__ == "__main__":
    #test_replay_buffer()
    main()




        
        
        

