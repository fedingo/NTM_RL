# credit: this code is derived from https://github.com/snowkylin/ntm
# the major changes made are to make this compatible with the abstract class tf.contrib.rnn.RNNCell
# an LSTM controller is used instead of a RNN controller
# 3 memory inititialization schemes are offered instead of 1
# the outputs of the controller heads are clipped to an absolute value
# we find that our modification result in more reliable training (we never observe gradients going to NaN) and faster convergence

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import collections

MANNControllerState = collections.namedtuple('MANNControllerState', ('controller_state', 'read_vector', 'attention', 'M_val'))


class MANNCell(tf.contrib.rnn.RNNCell):
    def __init__(self, controller_arch, embeddings_dims, memory_vector_dim, read_head_num, feedback=False):

        self.controller_arch = controller_arch + [memory_vector_dim]
        self.memory_size = 20
        self.memory_vector_dim = memory_vector_dim
        self.input_dim = embeddings_dims

        self.read_head_num = read_head_num
        self.feedback = feedback

        def single_cell(num_units):
            return tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias=1.0)

        self.controller = tf.contrib.rnn.MultiRNNCell([single_cell(layer) for layer in self.controller_arch])
        self.memory_val_weights = tf.get_variable("M_Val_Weights", [self.memory_vector_dim, self.input_dim])

        self.attention_layer = tf.layers.Dense(self.memory_size, name="attention_distribution")
        self.key_layer = tf.layers.Dense(self.memory_vector_dim, name="key_generator")
        self.controller_layer = tf.layers.Dense(self.memory_vector_dim, name="controller_dense")
        self.indexing_layer = tf.layers.Dense(1, name="index_generator")

        self.step = 0

    def __call__(self, x, prev_state):
        M_val = prev_state.M_val

        mode = "recurrent"

        controller_input = x
        if self.feedback:
            controller_input = tf.concat([x, prev_state.read_vector], axis=1)

        with tf.variable_scope('controller'):
            if mode == "recurrent":
                controller_output, controller_state = self.controller(controller_input, prev_state.controller_state)
            elif mode == "dense":
                controller_state = prev_state.controller_state
                controller_output = self.controller_layer(controller_input)

        with tf.name_scope("reading_op"):
            attention_distribution = tf.nn.softmax(tf.matmul(M_val, controller_output, transpose_b=True))
            read_vector = tf.matmul(attention_distribution, M_val, transpose_a=True)

        with tf.name_scope("writing_op"):
            # new Memory Value
            new_val = tf.matmul(x,self.memory_val_weights, transpose_b=True)

            M_val = self.write_mem(M_val, new_val)

        MANN_next_state = MANNControllerState(
                            attention=tf.squeeze(attention_distribution),
                            controller_state=controller_state,
                            read_vector=read_vector,
                            M_val=M_val)

        MANN_output = tf.concat([controller_output, read_vector], axis=1)
        # MANN_output = controller_output

        self.step += 1
        return MANN_output, MANN_next_state

    def zero_state(self, batch_size, dtype):

        init_mode = "eye"
        init_mem = None
        controller_init_state = self.controller.zero_state(batch_size, dtype)
        # Memory Initialization
        if init_mode == "constant":
            init_mem = np.zeros([self.memory_size, self.memory_vector_dim])
        elif init_mode == "eye":
            init_mem = np.eye(self.memory_size, M=self.memory_vector_dim)
        elif init_mode == "random":
            init_mem = np.random.rand(self.memory_size, self.memory_vector_dim)
            l2norm = np.sqrt((init_mem * init_mem).sum(axis=1))
            init_mem = init_mem / l2norm.reshape(self.memory_size, 1)

        init_attention = np.zeros(self.memory_size)

        return MANNControllerState(
            attention=tf.constant(init_attention,dtype=tf.float32),
            controller_state=controller_init_state,
            read_vector=tf.zeros([1, self.memory_vector_dim], dtype=tf.float32),
            M_val=tf.constant(init_mem, dtype=tf.float32) )

    @property
    def state_size(self):
        return MANNControllerState(
            attention=self.memory_size,
            controller_state=self.controller.state_size,
            read_vector=self.memory_vector_dim,
            M_val=[self.memory_size, self.memory_vector_dim])

    @property
    def output_size(self):
        return self.controller_arch[-1] + self.memory_vector_dim

    def get_attention(self, prev_M, new_data, mode="indexing"):
        # straight: NN generates the attention
        # by_key: NN generates a key to have the attention
        attention_distribution = None

        if mode == "straight":
            attention_distribution = self.attention_layer(new_data)
            attention_distribution = tf.nn.softmax(attention_distribution)
        elif mode == "by_key":
            key = self.key_layer(new_data)
            attention_distribution = tf.matmul(key, prev_M, transpose_b=True)
            attention_distribution = tf.nn.softmax(attention_distribution)
        elif mode == "implicit":
            attention_distribution = tf.matmul(new_data, prev_M, transpose_b=True)
            attention_distribution = tf.nn.softmax(attention_distribution)
        elif mode == "indexing":
            indexes = tf.range(self.memory_size)/(self.memory_size-1)
            indexes = tf.cast(indexes, dtype=tf.float32)
            indexes = tf.expand_dims(indexes, axis=0)
            mu = self.indexing_layer(new_data)
            mu = tf.nn.sigmoid(mu)

            dist = tfd.Normal(loc=mu, scale=1/self.memory_size)
            attention_distribution = dist.prob(indexes)
            attention_distribution = tf.nn.softmax(attention_distribution)

        return attention_distribution

    def write_mem(self, prev_M, new_data, mode="overwrite", normalize=True):
        # fifo: First In First Out
        # replace: Replacement with Attention M*(1-A) + A*V
        # overwrite: M + A*V
        new_mem = None

        if mode == "fifo":
            new_mem = tf.concat([new_data, prev_M], axis=0)[:-1, :]
        elif mode == "replace":
            attention_distribution = self.get_attention(prev_M, new_data)

            forget_distribution = np.ones(attention_distribution.shape) - attention_distribution
            multiplier = tf.tile(tf.transpose(forget_distribution), multiples=[1, self.memory_vector_dim])
            new_mem = tf.multiply(prev_M, multiplier) + tf.matmul(tf.transpose(attention_distribution), new_data)
        elif mode == "overwrite":
            attention_distribution = self.get_attention(prev_M, new_data)
            new_mem = prev_M + tf.matmul(tf.transpose(attention_distribution), new_data)

        if normalize:
            new_mem = tf.math.l2_normalize(new_mem, axis = 1)

        return new_mem
