import tensorflow as tf
import numpy as np

from abstractModel import abstractModel


class basic_RNN(abstractModel):

	def __init__(self, input_size, num_actions):
		total_series_length = 100
		truncated_backprop_length = 15
		batch_size = 1
		hidden_state_size = num_actions
		num_layers = 2

		self.zero_state = tuple([np.zeros((batch_size, hidden_state_size)) for _ in range(num_layers)])
		self.internal_current_state = self.zero_state

		''' GRAPH Building '''
		self.batchX_placeholder = tf.placeholder(tf.float32, [batch_size, None, input_size])
		self.batchY_placeholder = tf.placeholder(tf.float32, [batch_size, None, num_actions])

		self.init_state = tf.placeholder(tf.float32, [num_layers, batch_size, hidden_state_size])

		init = tf.unstack(self.init_state, axis=0)

		print(self.init_state.get_shape().as_list())
		print(self.batchX_placeholder.get_shape().as_list())

		# Unpack columns
		current_length = 2
		inputs_series = self.batchX_placeholder # tf.split(1, truncated_backprop_length, )
		target_series = self.batchY_placeholder # tf.unstack(tf.reshape(self.batchY_placeholder, [batch_size, current_length, num_actions]), axis=1)

		# Forward passes
		cell_class = tf.nn.rnn_cell.BasicRNNCell
		cell = tf.nn.rnn_cell.MultiRNNCell([cell_class(hidden_state_size, dtype= tf.float32) for _ in range(num_layers)])
		self.output_series, self.current_state = tf.nn.dynamic_rnn(cell, inputs_series, initial_state = init)

		print(self.output_series.get_shape().as_list())
		print(target_series.get_shape().as_list())

		self.total_loss = tf.reduce_mean(tf.squared_difference(target_series, self.output_series))
		self.train_step = tf.train.AdagradOptimizer(0.1).minimize(self.total_loss)
		self.tf_session = tf.Session()
		self.tf_session.run(tf.global_variables_initializer())

	def clone(self):
		raise NotImplementedError("Please implement the clone method")

	def train_on_batch(self, data, target):

		sampled_index = np.random.randint(len(data)-truncated_backprop_length)

		start_idx = sampled_index
		end_idx = start_idx + truncated_backprop_length

		batchX = data[:,start_idx:end_idx]
		batchY = target[:,start_idx:end_idx]

		_total_loss, _train_step, _current_state, _ = self.tf_session.run(
			[self.total_loss, self.train_step, self.current_state, self.output_series],
			feed_dict={
				self.batchX_placeholder: batchX,
				self.batchY_placeholder: batchY,
				self.init_state: self.zero_state
			})

	def predict(self, data):
		
		_predictions, self.internal_current_state = self.tf_session.run(
			[self.output_series, self.current_state],
			feed_dict={
				self.batchX_placeholder: data,
				self.init_state: self.internal_current_state
			})

		return _predictions


	def reset_internal_state(self):
		self.internal_current_state = self.zero_state



obj = basic_RNN(12, 6)

print(obj.predict(np.random.rand(1,2,10)))