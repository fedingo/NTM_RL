from collections import deque
import tensorflow as tf

class abstractAgent:

	def __init__(self, state_size, action_size, architecture, \
				learning_rate = 1e-3, gamma = .99,\
				memSize = 5000):

		self.state_size = state_size
		self.action_size = action_size

		self.mem_size = memSize
		self.memory = deque( maxlen = memSize)	
		self.learning_rate = learning_rate
		self.gamma = gamma

		self.architecture = architecture
		

		self.tf_session = tf.Session()
		self.build()
		self.tf_session.run(tf.global_variables_initializer())

	def store(self, dict_summary):
		self.memory.append(dict_summary)

	def clear_memory(self):
		self.memory.clear()


	# function that creates the network
	def build(self):
		raise NotImplementedError("Please implement the build of the network")

	# function to define when to train the network
	def hasToTrain(self, step, done, episode):
		raise NotImplementedError("Please implement the hasToTrain function")

	# function that trains the network
	def train(self):
		raise NotImplementedError("Please implement the training method")

	# function that returns some action
	def act(self, state, testing = False):
		raise NotImplementedError("Please implement the acting method")
