from NN_models.custom_mann import MANNCell, MANNControllerState
from NN_models.abstractModel import abstractModel

import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class MQN_model(abstractModel):
    def __init__(self, input_size, num_actions, envname="tmp", load=False):

        self.controller_arch = []
        self.memory_size = 24

        self.num_read_heads = 1
        self.num_write_heads = 1
        #self.output_dim = 96
        self.num_actions = num_actions
        self.input_size = input_size

        self.embeddings_dims = 64

        self.tau = 0.5
        self.feedback = True

        learning_rate = 1e-3

        self.batch_size = 1
        self.save_path = "models/" + envname + "/model.ckpt"

        self.inputs = tf.placeholder(tf.float32, shape=(self.batch_size, None, input_size), name="Input_Node")
        self.target = tf.placeholder(tf.float32, shape=(self.batch_size, None, num_actions), name="Target_Node")

        with tf.variable_scope('q_network'):
            self.cell = MANNCell(self.controller_arch, self.embeddings_dims, self.memory_size,
                                 read_head_num=self.num_read_heads, write_head_num=self.num_write_heads, feedback=self.feedback)
            self.model = self._build_model(self.cell)

        with tf.variable_scope('target_q_network'):
            self.target_cell = MANNCell(self.controller_arch, self.embeddings_dims, self.memory_size,
                                        read_head_num=self.num_read_heads, write_head_num=self.num_write_heads, feedback=self.feedback)
            self.frozen_model = self._build_model(self.target_cell)

        with tf.variable_scope('q_values_training_operations'):
            self.q_loss = tf.reduce_sum(tf.squared_difference(self.target, self.model['outputs']))
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

            max_grad_norm = 20
            trainable_variables = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.q_loss, trainable_variables), max_grad_norm)
            self.q_train_op = optimizer.apply_gradients(zip(grads, trainable_variables))

        with tf.name_scope("update_target_network"):
            critic_params = [t for t in tf.trainable_variables() if t.name.startswith('q_network')]
            critic_params = sorted(critic_params, key=lambda v: v.name)

            target_critic_params = [t for t in tf.trainable_variables() if t.name.startswith('target_q_network')]
            target_critic_params = sorted(target_critic_params, key=lambda v: v.name)

            self.target_update = []
            for p_qn_v, p_tqn_v in zip(critic_params, target_critic_params):
                op = p_tqn_v.assign(p_qn_v * self.tau + p_tqn_v * (1 - self.tau))
                self.target_update.append(op)

        self.tf_session = tf.Session()
        self.saver = tf.train.Saver()

        if load:
            self.saver.restore(self.tf_session, self.save_path)
        else:
            self.tf_session.run(tf.global_variables_initializer())

    def save_model(self):
        self.saver.save(self.tf_session, self.save_path)

    def _build_model(self, cell):

        node_dict = {}

        with tf.name_scope("before_ntm"):
            arch = [196, self.embeddings_dims]
            rnn_output_sequence = self.inputs

            for l in arch:
                rnn_output_sequence = tf.contrib.layers.fully_connected(
                    inputs=rnn_output_sequence,
                    num_outputs=l,
                    activation_fn=tf.nn.relu
                )

        rnn_output_sequence, memory_sequence = tf.nn.dynamic_rnn(
            cell=cell,
            inputs=rnn_output_sequence,
            initial_state=None,
            dtype=tf.float32,
            time_major=False)

        q_output_sequence = rnn_output_sequence

        with tf.name_scope("after_ntm"):
            with tf.name_scope("state_value"):
                arch = [128, 64]
                output_sequence = q_output_sequence

                for l in arch:
                    output_sequence = tf.contrib.layers.fully_connected(
                        inputs=output_sequence,
                        num_outputs=l,
                        activation_fn=tf.nn.relu
                    )

                node_dict['current_latent'] = output_sequence

                output_sequence = tf.contrib.layers.fully_connected(
                    inputs=output_sequence,
                    num_outputs=1,
                    activation_fn=None
                )

                tile_shape = list(np.copy([self.num_actions]))
                tile_shape = [1, 1] + tile_shape
                output_sequence = tf.tile(output_sequence, multiples=tile_shape)
                node_dict['state_value'] = output_sequence
                node_dict['outputs'] = output_sequence

            with tf.name_scope("advantages"):
                arch = [128, 64]
                output_sequence = q_output_sequence

                for l in arch:
                    output_sequence = tf.contrib.layers.fully_connected(
                        inputs=output_sequence,
                        num_outputs=l,
                        activation_fn=tf.nn.relu
                    )

                output_sequence = tf.contrib.layers.fully_connected(
                    inputs=output_sequence,
                    num_outputs=self.num_actions,
                    activation_fn=tf.nn.sigmoid
                )

                mean = tf.reduce_mean(output_sequence, axis=-1, keepdims=True)
                output_sequence = output_sequence - mean

                node_dict['outputs'] += output_sequence

        node_dict['rnn_outputs'] = rnn_output_sequence
        node_dict['memory'] = memory_sequence

        return node_dict

    def predict(self, states_vector):

        outputs, memory = self.tf_session.run(
            [self.model['outputs'], self.model['memory']],
            feed_dict={
                self.inputs: states_vector
            })

        return outputs[-1], memory

    def predict_frozen(self, states_vector):

        outputs = self.tf_session.run(
            self.frozen_model['outputs'],
            feed_dict={
              self.inputs: states_vector
            })

        return outputs[-1]

    def train_on_batch(self, data, target, mem=None):

        loss, _ = self.tf_session.run(
            [self.q_loss, self.q_train_op],
            feed_dict={
              self.inputs: data,
              self.target: target
            })


        return loss

    def update_frozen(self):
        self.tf_session.run(self.target_update)

# obj = MQN_model(25, 3)
#
#
# q_vals = obj.predict(np.random.rand(1,3,25))
#
# print("First")
# print(q_vals[-1])
#
# q_vals = obj.predict(np.random.rand(1,4,25))
#
# print("Second")
# print(q_vals[-1])
#
# #print(obj.predict_frozen((np.random.rand(1,4,25))))
#
# print(obj.train_on_batch(np.random.rand(1,4,25),
# 						 np.random.rand(1,4,3)
# 	))
#
# obj.update_frozen()
