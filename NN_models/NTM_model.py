import sys
sys.path.insert(0, '..')

from ntm import NTMCell, NTMControllerState
from NN_models.abstractModel import abstractModel

import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class NTM_model(abstractModel):
	def __init__(self, input_size, num_actions, envname = "tmp", load = False):

		self.num_layers = 2
		self.num_units = 32
		self.num_memory_locations = 32
		self.memory_size = 32
		self.num_read_heads = 2
		self.num_write_heads = 1
		self.output_dim = 32
		self.num_actions = num_actions
		self.input_size = input_size

		self.tau = 0.5

		self.constant_value = 1e-6
		learning_rate = 2e-3

		self.print_memory = False

		self.batch_size = 1
		self.save_path = "models/"+envname+"/model.ckpt"

		self.inputs = tf.placeholder(tf.float32, shape=(self.batch_size, None, input_size), name="Input_Node")
		self.target = tf.placeholder(tf.float32, shape=(self.batch_size, None, num_actions), name="Target_Node")
		self.predict_state = tf.placeholder(tf.float32, shape=(self.batch_size, None, input_size), name="Target_Node")
		self.memory_init = tf.placeholder(tf.float32, shape=(self.batch_size, self.memory_size, self.num_memory_locations),
										  name="Memory_Node")

		self.cell = NTMCell(self.num_layers, self.num_units, self.num_memory_locations, self.memory_size,
							self.num_read_heads, self.num_write_heads, output_dim=self.output_dim, reuse = False)
		self.target_cell = NTMCell(self.num_layers, self.num_units, self.num_memory_locations, self.memory_size,
							self.num_read_heads, self.num_write_heads, output_dim=self.output_dim, reuse=False)

		default_init_tuple = self.cell.zero_state(self.batch_size, dtype=tf.float32)
		self.init_tuple = NTMControllerState(controller_state= default_init_tuple.controller_state,
											 read_vector_list= default_init_tuple.read_vector_list,
											 w_list= default_init_tuple.w_list,
											 M= self.memory_init)

		with tf.variable_scope('q_network'):
			self.model = self._build_model(self.cell)

		with tf.variable_scope('target_q_network'):
			self.frozen_model = self._build_model(self.target_cell)

		with tf.variable_scope('q_values_training_operations'):
			self.q_loss = tf.reduce_sum(tf.squared_difference(self.target, self.model['outputs']))
			optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

			max_grad_norm = 50
			trainable_variables = tf.trainable_variables()
			grads, _ = tf.clip_by_global_norm(tf.gradients(self.q_loss, trainable_variables), max_grad_norm)
			self.q_train_op = optimizer.apply_gradients(zip(grads, trainable_variables))

		with tf.variable_scope('predictor_training_operations'):
			self.pred_loss = tf.reduce_sum(tf.squared_difference(self.predict_state, self.model['predictor']))
			optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

			max_grad_norm = 50
			trainable_variables = tf.trainable_variables()
			grads, _ = tf.clip_by_global_norm(tf.gradients(self.pred_loss, trainable_variables), max_grad_norm)
			self.pred_train_op = optimizer.apply_gradients(zip(grads, trainable_variables))


		with tf.name_scope("update_target_network"):
			critic_params = [t for t in tf.trainable_variables() if t.name.startswith('q_network')]
			critic_params = sorted(critic_params, key=lambda v: v.name)

			target_critic_params = [t for t in tf.trainable_variables() if t.name.startswith('target_q_network')]
			target_critic_params = sorted(target_critic_params, key=lambda v: v.name)

			self.target_update = []
			for p_qn_v, p_tqn_v in zip(critic_params, target_critic_params):
				op = p_tqn_v.assign(p_qn_v*self.tau + p_tqn_v*(1-self.tau))
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
			arch = [256]
			rnn_output_sequence = self.inputs

			for l in arch:
				rnn_output_sequence = tf.contrib.layers.fully_connected(
					inputs=rnn_output_sequence,
					num_outputs=l,
					activation_fn=tf.nn.relu
				)

		rnn_output_sequence, memory_sequence = tf.nn.dynamic_rnn(
			cell = cell,
			inputs = rnn_output_sequence,
			initial_state = self.init_tuple,
			dtype=tf.float32,
			time_major = False)

		q_output_sequence = rnn_output_sequence[:, :, :self.output_dim]
		s_output_sequence = rnn_output_sequence[:, :, self.output_dim:]

		# output_sequence = tf.concat([output_sequence,self.inputs], axis=-1)
		with tf.name_scope("after_ntm"):
			with tf.name_scope("q_values"):
				arch = [128]
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
					activation_fn=None
				)
				node_dict['outputs'] = output_sequence

			with tf.name_scope("predictor"):

				arch = [48]
				output_sequence = s_output_sequence

				for l in arch:
					output_sequence = tf.contrib.layers.fully_connected(
						inputs=output_sequence,
						num_outputs=l,
						activation_fn=tf.nn.relu
					)

				output_sequence = tf.contrib.layers.fully_connected(
					inputs=output_sequence,
					num_outputs=self.input_size,
					activation_fn=tf.nn.sigmoid
				)
				node_dict['predictor'] = output_sequence

		node_dict['rnn_outputs'] = rnn_output_sequence
		node_dict['memory' ] = memory_sequence

		return node_dict

	def predict(self, state, mem = None):

		if mem is None:
			mem = np.ones([self.batch_size, self.memory_size, self.num_memory_locations]) * self.constant_value

		outputs, memory, rnn_o = self.tf_session.run([self.model['outputs'], self.model['memory'], self.model['rnn_outputs']],
			feed_dict={
				self.inputs: state,
				self.memory_init: mem
			})

		return outputs[0], memory.M

	def predict_frozen(self, state, mem = None):

		if mem is None:
			mem = np.ones([self.batch_size, self.memory_size, self.num_memory_locations]) * self.constant_value

		outputs, memory = self.tf_session.run([self.frozen_model['outputs'], self.frozen_model['memory']],
			feed_dict={
				self.inputs: state,
				self.memory_init: mem
			})

		return outputs[0]

	def train_on_batch(self, data, target, data_next = None, mem = None):

		if mem is None:
			mem = np.ones([self.batch_size, self.memory_size, self.num_memory_locations]) * self.constant_value

		loss, _ = self.tf_session.run([self.q_loss, self.q_train_op],
			feed_dict={
				self.inputs: data,
				self.target: target,
				self.memory_init: mem
			})

		p_loss = None

		if data_next is not None:
			p_loss, _ = self.tf_session.run([self.pred_loss, self.pred_train_op],
											feed_dict={
											  self.inputs: data,
											  self.predict_state: data_next,
											  self.memory_init: mem
											})

		return loss, p_loss

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