import sys
sys.path.insert(0, '..')

from NN_models.ntm import NTMCell, NTMControllerState
from NN_models.abstractModel import abstractModel

import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class LSTM_model(abstractModel):
	def __init__(self, input_size, num_actions, load = False, envname = "tmp"):

		self.num_layers = 2
		self.num_units = 32
		self.num_memory_locations = 10
		self.memory_size = 10
		self.num_read_heads = 1
		self.num_write_heads = 1
		self.output_dim = 32
		self.num_actions = num_actions

		self.constant_value = 1e-6

		self.print_memory = False

		self.batch_size = 1

		self.inputs = tf.placeholder(tf.float32, shape=(self.batch_size, None, input_size), name="Input_Node")
		self.target = tf.placeholder(tf.float32, shape=(self.batch_size, None, num_actions), name="Target_Node")
		self.memory_init = tf.placeholder(tf.float32, shape=(self.batch_size, 2, self.output_dim),
										  name="Memory_Node")

		self.save_path = "models/" + envname + "/model.ckpt"

		self.cell = tf.nn.rnn_cell.LSTMCell(self.output_dim)
		self.target_cell = tf.nn.rnn_cell.LSTMCell(self.output_dim)

		c = self.memory_init[:,0]
		h = self.memory_init[:,1]
		self.init_tuple = tf.nn.rnn_cell.LSTMStateTuple(c,h)


		with tf.variable_scope('q_network'):
			self.model = self._build_model(self.cell)

		with tf.variable_scope('target_q_network'):
			self.frozen_model = self._build_model(self.target_cell)

		learning_rate = 5e-4

		with tf.variable_scope('training_operations'):
			self.loss = tf.reduce_sum(tf.squared_difference(self.target, self.model['outputs']))
			optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

			max_grad_norm = 50
			trainable_variables = tf.trainable_variables()
			grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, trainable_variables), max_grad_norm)
			self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))


		with tf.name_scope("update_target_network"):
			critic_params = [t for t in tf.trainable_variables() if t.name.startswith('q_network')]
			critic_params = sorted(critic_params, key=lambda v: v.name)

			target_critic_params = [t for t in tf.trainable_variables() if t.name.startswith('target_q_network')]
			target_critic_params = sorted(target_critic_params, key=lambda v: v.name)

			tau = 0.5
			self.target_update = []
			for p_qn_v, p_tqn_v in zip(critic_params, target_critic_params):
				op = p_tqn_v.assign(p_qn_v * tau + p_tqn_v * (1 - tau))
				self.target_update.append(op)



		self.tf_session = tf.Session()
		self.tf_session.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver()

		if load:
			self.saver.restore(self.tf_session, self.save_path)
		else:
			self.tf_session.run(tf.global_variables_initializer())

	def save_model(self):
		self.saver.save(self.tf_session, self.save_path)

	def _null_memory(self):
		return np.ones([self.batch_size, 2, self.output_dim]) * self.constant_value

	def _build_model(self, cell):

		node_dict = {}

		rnn_output_sequence, memory_sequence = tf.nn.dynamic_rnn(
			cell = cell,
			inputs = self.inputs,
			initial_state = self.init_tuple,
			dtype=tf.float32,
			time_major = False)

		output_sequence = rnn_output_sequence

		with tf.name_scope("after_lstm"):

			with tf.name_scope("state_value"):
				arch = [32]
				output_sequence = rnn_output_sequence

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
				tile_shape = [1,1] + tile_shape
				output_sequence = tf.tile(output_sequence, multiples=tile_shape)
				node_dict['state_value'] = output_sequence
				node_dict['outputs'] = output_sequence

			#output_sequence = tf.concat([output_sequence,self.inputs], axis=-1)

			with  tf.name_scope("advantages"):
				arch = [64, 64]

				for l in arch:
					output_sequence = tf.contrib.layers.fully_connected(
						inputs=output_sequence,
						num_outputs=l,
						activation_fn=tf.nn.relu
					)

				output_sequence = tf.contrib.layers.fully_connected(
					inputs=output_sequence,
					num_outputs=self.num_actions,
					activation_fn=None
				)

				mean = tf.reduce_mean(output_sequence, axis=2, keepdims=True)
				output_sequence = output_sequence - mean

				node_dict['rnn_outputs'] = rnn_output_sequence
				node_dict['outputs'] += output_sequence
				node_dict['memory' ] = memory_sequence

		return node_dict

	def predict(self, state, mem = None):

		if mem is None:
			mem = self._null_memory()

		outputs, memory = self.tf_session.run([self.model['outputs'], self.model['memory']],
			feed_dict={
				self.inputs: state,
				self.memory_init: mem
			})

		encoded = np.array([memory.c[0], memory.h[0]])
		encoded = np.reshape(encoded, [1, 2, self.output_dim])
		return outputs[0], encoded

	def predict_frozen(self, state, mem = None):

		if mem is None:
			mem = self._null_memory()

		outputs, memory = self.tf_session.run([self.frozen_model['outputs'], self.frozen_model['memory']],
			feed_dict={
				self.inputs: state,
				self.memory_init: mem
			})

		return outputs[0]

	def train_on_batch(self, data, target, mem = None):

		if mem is None:
			mem = self._null_memory()

		loss, _ = self.tf_session.run([self.loss, self.train_op],
			feed_dict={
				self.inputs: data,
				self.target: target,
				self.memory_init: mem
			})

		return loss

	def update_frozen(self):
		self.tf_session.run(self.target_update)

#
# obj = NTM_model(25, 3)
#
#
# q_vals, memory = obj.predict(np.random.rand(1,3,25))
#
# print("First")
# print(q_vals[-1])
#
# print("Memory")
# print(memory)
#
# q_vals, memory = obj.predict(np.random.rand(1,4,25), memory)
#
# print("Second")
# print(q_vals[-1])
#
# print("Memory")
# print(memory)
#
# #print(obj.predict_frozen((np.random.rand(1,4,25))))
#
# print(obj.train_on_batch(np.random.rand(1,4,25),
# 						 np.random.rand(1,4,3)
# 	))
#
# obj.update_frozen()