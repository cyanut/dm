import numpy as np
import tensorflow as tf
import sonnet as snt
import random
import os
from numpngw import write_apng
import io

class CNN(snt.AbstractModule):
    def __init__(self, conv_params, max_pool_params = {}, name="CNN"):
        super(CNN, self).__init__(name=name)
        self.conv_params = conv_params
        self.n_layer = len(conv_params["output_channels"])
        if not "padding" in conv_params:
            self.conv_params["padding"] = ["VALID"] * self.n_layer
        if max_pool_params:
            self.max_pool_params = dict(\
                    ksize = self.conv_params["kernel_shape"],
                    strides = self.conv_params["strides"],
                    padding = self.conv_params["padding"],
                    data_format = 'NHWC')
            self.max_pool_params.update(max_pool_params)
        else:
            self.max_pool_params = max_pool_params

    def _build(self, inputs):
        outputs = inputs
        for i in range(self.n_layer):
            outputs = snt.Conv2D(\
                    output_channels=self.conv_params["output_channels"][i],
                    kernel_shape=self.conv_params["kernel_shape"][i],
                    stride = self.conv_params["stride"][i],
                    padding= self.conv_params["padding"][i],
                    )(outputs)
            outputs = tf.nn.relu(outputs)
            if self.max_pool_params:
                outputs = tf.nn.max_pool(outputs, **self.max_pool_params)
        outputs = snt.BatchFlatten()(outputs)
        return outputs
        

class DNN(snt.AbstractModule):
    def __init__(self, output_size, layer_size=[64,64], name="DNN"):
        super(DNN, self).__init__(name=name)
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
        a_net = DNN(output_size=self.output_size, layer_size=self.layer_size, name="advantage_network")
        v_net = DNN(output_size=1, layer_size=self.layer_size, name="value_network")
        a_output = a_net(state)
        a_output = a_output - tf.reduce_mean(a_output, axis=1, keepdims=True)
        outputs = a_output + v_net(state)
        return outputs


class QAgent(snt.AbstractModule):
    def __init__(self, Model, model_params, reward_discount, learning_rate, update_frac, name="QAgent"):
        super(QAgent, self).__init__(name=name)
        self.Model = Model
        self.model_params = model_params
        self.update_frac = update_frac
        self.reward_discount = reward_discount
        self.learning_rate = learning_rate

    def _build(self, state, new_state, action, reward, is_running, global_step):
        self.q_net = self.Model(name="q_net", **self.model_params)
        self.target_q_net = self.Model(name="q_target", **self.model_params)
        q_target = is_running * tf.reduce_max(\
                self.target_q_net(new_state), axis=1)
        #q_target = is_running * tf.reduce_max(\
        #        self.q_net(state=new_state), axis=1)

        q_all = self.q_net(state)
        q = tf.reduce_sum(q_all * action, axis=1)
        cost = (q - reward - self.reward_discount * q_target) ** 2
        trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)\
                    .minimize(cost, 
                              var_list = self.q_net.get_all_variables(),
                              global_step=global_step)
        return cost, trainer 

    @snt.reuse_variables
    def update_target_network(self):
        ops = []
        for v_q, v_target_q in zip(self.q_net.get_all_variables(), 
                self.target_q_net.get_all_variables()):
            ops.append(v_target_q.assign(self.update_frac * v_q + (1 - self.update_frac) * v_target_q))
        return ops

    @snt.reuse_variables
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

'''
def _array_to_gif(arr, frame_rate):
    if len(arr.shape) == 3:
        pix_fmt = 'gray8'
    elif len(arr.shape) == 4 and arr.shape[-1] == 3:
        pix_fmt = 'rgb24'
    if arr.dtype == np.float:
        arr = np.round(arr * 255).astype('uint8')
    cmdstr = ('ffmpeg', '-y', '-r', str(frame_rate), 
              '-nostats', '-loglevel', '0',
              '-f', 'rawvideo', '-pix_fmt', pix_fmt, 
              '-s', '{}x{}'.format(arr.shape[-2], arr.shape[-3]), '-i', '-',
              '-filter_complex', 
              '[0:v] split [a][b]; [a] fifo [a0]; [b] fifo [b0]; [a0] palettegen,fifo [p]; [b0][p] paletteuse,fifo',
              '-f', 'gif', '-')
    with subprocess.Popen(cmdstr, stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, shell=False, bufsize=2**20) as p:
        res,_ = p.communicate(arr.tobytes())
    return res
'''
def _array_to_gif(arr, frame_rate):
    if len(arr.shape) == 3:
        arr = np.array([arr]*3).transpose([1,2,3,0])
    if arr.dtype == np.float:
        arr = np.round(arr * 255).astype('uint8')
    res = io.BytesIO(b"")
    write_apng(res, arr, delay=1000./frame_rate)
    return res.getbuffer()


def mk_gif_image_summary(video, frame_rate=10):
    img = tf.Summary.Image()
    img.height = video.shape[2]
    img.width = video.shape[1]
    if len(video.shape) == 4:
        img.colorspace = video.shape[3]
    elif len(video.shape) == 3:
        img.colorspace = 1
    img.encoded_image_string = _array_to_gif(video, frame_rate)
    return img

def test_replay_buffer():
    rb = ReplayBuffer(10)
    for i in range(1):
        rb.add([np.arange(i*10, i*10+6).reshape((2,3)), i, i*2])
    print(rb.sample(3))
    for i in range(5, 20):
        rb.add([np.arange(i*10, i*10+6).reshape((2,3)), i, i*2])
    print(rb.sample(3))
    print(rb.sample(20))

def test_mk_gif(fname):
    vid = np.random.random(size=(10, 64, 32, 3))
    img = mk_gif_image_summary(vid)
    with open(fname, "wb") as fo:
        fo.write(img.encoded_image_string)
    

class DummySummary():
    def __init__(self, *args):
        pass
    def add_summary(self, *args):
        pass
    def add(self, *args):
        pass
    def add_graph(self, *args):
        pass

def frame_hash(arr):
    return hex(hash(np.round(arr*16).astype('uint8').tostring()))


if __name__ == "__main__":
    import gym
    import time
    import sys
    from atari_qnet import PreprocessedAtariEnv
    env = PreprocessedAtariEnv("Breakout-v0")
    env.reset()
    img = []
    is_end = False
    counter = 0
    while not is_end:
        counter += 1
        img.append(env.render(mode="rgb_array"))
        s, r, is_end, _ = env.step(env.action_space.sample())
        if counter % 100 == 0:
            print(counter)
    img = []
    is_end = False
    env.reset()

    while not is_end:
        counter += 1
        img.append(env.render(mode="rgb_array"))
        s, r, is_end, _ = env.step(env.action_space.sample())
        if counter % 100 == 0:
            print(counter)
    print(len(img), img[0].shape)
    with open(sys.argv[1], "wb") as fo:
        res = _array_to_gif(np.array(img), frame_rate=30)
        print(len(res))
        fo.write(res)
        


        
