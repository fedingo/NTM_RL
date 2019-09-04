import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import collections

MANNControllerState = collections.namedtuple('MANNControllerState', ('read_vector', 'read_attentions', 'M'))


class MANNCell(tf.contrib.rnn.RNNCell):
    def __init__(self, controller_arch, embeddings_dims, memory_vector_dim, read_head_num, write_head_num, feedback=False):

        self.controller_arch = controller_arch + [memory_vector_dim]
        self.memory_size = 12
        self.memory_vector_dim = memory_vector_dim
        self.input_dim = embeddings_dims
        self.routing_size = 2*self.input_dim

        self.read_head_num = read_head_num
        self.write_head_num = write_head_num
        self.feedback = feedback

        self.write_mode = "overwrite"  # overwrite, replace, fifo
        self.zero_state_init_mode = "constant"  # eye, constant, random
        self.write_attention_mode = "straight"  # indexing, straight, by_key, implicit
        self.read_attention_mode = "straight"  # indexing, straight, by_key
        self.normalize_memory = self.write_mode == "overwrite"

        self.memory_value,self.routing, self.attention_layer = {}, {}, {}
        for i in range(write_head_num):
            self.init_write_heads(i)
        for i in range(read_head_num):
            self.init_read_heads(i)

    def init_write_heads(self, head_index):

        head_id = "W"+str(head_index)
        self.memory_value[head_id] = tf.layers.Dense(self.memory_vector_dim, name="write_data"+str(head_index))

        if self.write_attention_mode == "straight":
            self.attention_layer[head_id] = tf.layers.Dense(self.memory_size)
        elif self.write_attention_mode == "by_key":
            self.attention_layer[head_id] = tf.layers.Dense(self.memory_vector_dim)
        elif self.write_attention_mode == "indexing":
            self.attention_layer[head_id] = tf.layers.Dense(1)

    def init_read_heads(self, head_index):

        head_id = "R"+str(head_index)
        self.routing[head_id] = tf.layers.Dense(self.routing_size)

        if self.read_attention_mode == "straight":
            self.attention_layer[head_id] = tf.layers.Dense(self.memory_size)
        elif self.read_attention_mode == "by_key":
            self.attention_layer[head_id] = tf.layers.Dense(self.memory_vector_dim)
        elif self.read_attention_mode == "indexing":
            self.attention_layer[head_id] = tf.layers.Dense(1)

    def __call__(self, x, prev_state):
        M = prev_state.M

        inp = x
        inp = tf.math.l2_normalize(inp, axis=-1)
        if self.feedback:
            inp = tf.concat([x, prev_state.read_vector], axis=1)

        with tf.name_scope("reading_op"):
            # Read vectors from memory
            read_vector, read_attentions = self.read_mem(M, inp)

        with tf.name_scope("writing_op"):
            # new Memory Value
            M = self.write_mem(M, inp)

        next_state = MANNControllerState(
                            read_attentions=read_attentions,
                            read_vector=read_vector,
                            M=M)
        output = tf.concat([x, read_vector], axis=-1)

        return output, next_state

    def zero_state(self, batch_size, dtype):
        init_mode = self.zero_state_init_mode
        init_mem = None
        # Memory Initialization
        if init_mode == "constant":
            init_mem = np.zeros([self.memory_size, self.memory_vector_dim])
        elif init_mode == "eye":
            init_mem = np.eye(self.memory_size, M=self.memory_vector_dim)
        elif init_mode == "random":
            init_mem = np.random.rand(self.memory_size, self.memory_vector_dim)
            l2norm = np.sqrt((init_mem * init_mem).sum(axis=1))
            init_mem = init_mem / l2norm.reshape(self.memory_size, 1)

        init_attention = np.zeros([self.read_head_num, self.memory_size])

        return MANNControllerState(
            read_attentions=tf.constant(init_attention, dtype=tf.float32),
            read_vector=tf.zeros([1, self.memory_vector_dim*self.read_head_num], dtype=tf.float32),
            M=tf.constant(init_mem, dtype=tf.float32) )

    @property
    def state_size(self):
        return MANNControllerState(
            read_attentions=self.memory_size*self.read_head_num,
            read_vector=[self.read_head_num, self.memory_vector_dim],
            M=[self.memory_size, self.memory_vector_dim])

    @property
    def output_size(self):
        return self.input_dim + self.memory_vector_dim * self.read_head_num

    def get_attention(self, prev_M, new_data, head_id, mode):
        # straight: NN generates the attention
        # by_key: NN generates a key to have the attention

        if mode == "straight":
            attention_distribution = self.attention_layer[head_id](new_data)
            attention_distribution = tf.nn.softmax(attention_distribution)
        elif mode == "by_key":
            key = self.attention_layer[head_id](new_data)
            #Missing Activation?
            key = tf.nn.relu(key)
            if self.normalize_memory:
                key = tf.math.l2_normalize(key, axis=-1)

            attention_distribution = tf.matmul(key, prev_M, transpose_b=True)
            attention_distribution = tf.nn.softmax(attention_distribution)
        elif mode == "implicit":
            attention_distribution = tf.matmul(new_data, prev_M, transpose_b=True)
            attention_distribution = tf.nn.softmax(attention_distribution)
        elif mode == "indexing":
            indexes = tf.range(self.memory_size)/(self.memory_size-1)
            indexes = tf.cast(indexes, dtype=tf.float32)
            indexes = tf.expand_dims(indexes, axis=0)
            mu = self.attention_layer[head_id](new_data)
            mu = tf.nn.sigmoid(mu)

            dist = tfd.Normal(loc=mu, scale=.1/self.memory_size)
            attention_distribution = dist.prob(indexes)
            attention_distribution = tf.nn.softmax(attention_distribution)
            #attention_distribution = tf.math.l2_normalize(attention_distribution)
        else:
            attention_distribution = None
            raise Exception("Attention mode not recognized")

        return attention_distribution

    def write_mem(self, mem, sample):
        # fifo: First In First Out
        # replace: Replacement with Attention M*(1-A) + A*V
        # overwrite: M + A*V
        new_mem = mem
        mode = self.write_mode

        for i in range(self.write_head_num):
            head_id = "W"+str(i)
            new_data = self.memory_value[head_id](sample)
            # new_data = tf.nn.relu(new_data)

            if mode == "fifo":
                new_mem = tf.concat([new_data, new_mem], axis=0)[:-1, :]

            elif mode == "replace":
                # Compute attention on the previous state of the memory, and then is sums on the new instance
                # of the memory
                attention_distribution = self.get_attention(mem, new_data, head_id, self.write_attention_mode)
                forget_distribution = np.ones(attention_distribution.shape) - attention_distribution
                multiplier = tf.tile(tf.transpose(forget_distribution), multiples=[1, self.memory_vector_dim])

                new_mem = tf.multiply(new_mem, multiplier) + tf.matmul(attention_distribution, new_data, transpose_a=True)

            elif mode == "overwrite":
                # Compute attention on the previous state of the memory, and then is sums on the new instance
                # of the memory
                attention_distribution = self.get_attention(mem, new_data, head_id, self.write_attention_mode)
                new_mem = new_mem + tf.matmul(attention_distribution, new_data, transpose_a=True)

            if self.normalize_memory:
                new_mem = tf.math.l2_normalize(new_mem, axis=-1)

        return new_mem

    def read_mem(self, mem, sample):

        read_vectors = []
        attentions = []
        for i in range(self.read_head_num):
            head_id = "R"+str(i)

            # router = self.routing[head_id](sample)
            attention_distribution = self.get_attention(mem, sample, head_id, self.read_attention_mode)
            read_vector = tf.matmul(attention_distribution, mem)
            attentions.append(attention_distribution)
            read_vectors.append(read_vector)

        return tf.concat(read_vectors, axis=-1), tf.concat(attentions, axis=0)
