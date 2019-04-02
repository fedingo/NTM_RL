class abstractModel():

	def __init__(self, input_size, num_actions):
		raise NotImplementedError("Please implement the __init__ method")

	def update_frozen(self):
		raise NotImplementedError("Please implement the clone method")

	def train_on_batch(self, data, target):
		raise NotImplementedError("Please implement the train method")

	def predict(self, data):
		raise NotImplementedError("Please implement the clone method")

	def predict_frozen(self, data):
		raise NotImplementedError("Please implement the clone method")
