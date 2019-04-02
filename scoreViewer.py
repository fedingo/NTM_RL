import matplotlib.pyplot as plt

class scoreViewer:

	def __init__(self, lowest_score = 0, target_score = 0):
		self.fig = plt.figure()
		self.ax1 = self.fig.add_subplot(1,1,1)

		self.score_mean = lowest_score
		self.mean_array = []
		self.score_mean_100 = lowest_score
		self.mean_array_100 = []

		self.target_score = target_score

		plt.ion()
		plt.show()

	## Function to add the score to the graph 
	##	score: score achieved in the the last episode
	##	Return: boolean to handle if the window is still open
	def addScore(self, score):

		if self.score_mean == 0:
			self.score_mean = score
			self.score_mean_100 = score

		self.score_mean *= 0.9
		self.score_mean += 0.1*score
		self.mean_array += [self.score_mean]

		self.score_mean_100 *= 0.99
		self.score_mean_100 += 0.01*score
		self.mean_array_100 += [self.score_mean_100]

		x_array = range(len(self.mean_array))
		plt.plot(x_array, self.mean_array, 'C1')
		plt.plot(x_array, self.mean_array_100, 'C2')
		plt.draw()
		plt.pause(0.001)
		return  plt.fignum_exists(1)

	## Loggin function that prints the current number of episodes and the average scores  (10 and 100)
	def printMeans(self):

		episode = self.getEpisodeNumber()
		print('Episodes performed: %d - Score Mean 10: %f - Score Mean 100: %f          ' %\
				 (episode, self.score_mean, self.score_mean_100))

	## Function that returns if the target mean score has been reached 
	## 	Return: boolean value that evaluates if the target score has been reached
	def isFinished(self):

		return (self.score_mean_100 >= self.target_score\
				 and self.target_score != 0)
		
	def saveToFile(self, string):
		x_array = range(len(self.mean_array))
		plt.plot(x_array, self.mean_array, 'C1')
		plt.plot(x_array, self.mean_array_100, 'C2')
		plt.draw()
		plt.savefig(string)

	def getEpisodeNumber(self):
		return len(self.mean_array)
	