import gym
import gym.spaces
import gym_maze_exploration
import gym_unblockme
import numpy as np

import time
# Used to hide gym core warning
gym.logger.set_level(40)

import sys, inspect
import importlib
import scoreViewer as sv

# Used to hide the Tensorflow warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def test_agent(tested_agent, env):

	score = 0
	total_score = 0
	episode = 0
	test_episodes = 5
	state = env.reset()
	tested_agent.reset_mem()
	if len(state.shape) > 1:
		state = state.flatten()

	print("\n Starting Test \n")

	while episode < test_episodes:
		action = tested_agent.act(state, testing = True)
		if action_reshape:
			action_ = np.unravel_index(action, original_action_size)
			next_state, reward, done, info = env.step(action_)
		else:
			next_state, reward, done, info = env.step(action)
		total_score += reward; score += reward

		state = next_state
		if len(state.shape) > 1:
			state = state.flatten()

		env.render()

		if done:
			episode += 1
			state = env.reset()
			if len(state.shape) > 1:
				state = state.flatten()
			tested_agent.reset_mem()
			print("episode done: score %d" % score)
			score = 0

	print("\nModel Score is %.2f" % (total_score/test_episodes))

	return total_score/test_episodes


print("\n======= RL Agent Testing with the OpenAI gym =======\n")

if len(sys.argv) < 3:
	print ("Usage: <operation> <agent_class> <environment>")
	print ("Optional: <target_episodes>")
	sys.exit(0)

operation = sys.argv[1]
agent_class = sys.argv[2]
environment = sys.argv[3]
target_score = 0
target_episodes = 1000

if len(sys.argv) == 5:
	target_episodes = int(sys.argv[4])

# Importing the given class
module = importlib.import_module(agent_class)
class_tuples = inspect.getmembers(module, inspect.isclass)
agents_list = []
for el in class_tuples:
	if 'Agent' in el[0]:
		agents_list += [el]

agent_tuple = agents_list[0]
agent_class = agent_tuple[1]
print('Loading class: ' + agent_tuple[0])

# Preparing image target
image_path  = "images/" + agent_tuple[0] + "_" + environment + ".png"

# Created user defined environment
print('Load gym environment: ' + environment + "\n")
env = gym.make(environment)

# Creating the Gym environment
state = env.reset()
if len(state.shape) > 1:
	state = state.flatten()

state_size = state.shape[0]
action_reshape = False
if type(env.action_space) == type(gym.spaces.MultiDiscrete([])):
	original_action_size = env.action_space.nvec
	action_size = np.prod(original_action_size); action_reshape = True
	action_size = np.asscalar(action_size)
else:
	action_size = env.action_space.n

print("State size is %d, action size is %d" % (state_size, action_size))

step = 0; score = 0; episodes = 0
start_time = time.time()

agent = agent_class(state_size, action_size, load = (operation == "test"), envname = environment)
#tested_performance = test_agent(agent, env)
viewer = sv.scoreViewer(target_score = target_score) # tested_performance)

if operation == "train":
	while True:

		action = agent.act(state)
		if action_reshape:
			action_ = np.unravel_index(action, original_action_size)
			next_state, reward, done, info = env.step(action_)
		else:
			next_state, reward, done, info = env.step(action)
		score += reward

		if len(next_state.shape) > 1:
			next_state = next_state.flatten()

		obj = { \
			"state"  : state,
			"action" : action,
			"reward" : reward,
			"done"   : done,
			"next_state" : next_state
		}

		agent.store(obj)
		state = next_state

		if agent.hasToTrain(step, done, episodes):
			agent.train()
		step += 1

		if done:
			episodes += 1
			agent.reset_mem()
			if not viewer.addScore(score):
				break

			time_taken = (time.time() - start_time)
			print('Episode: %d - Score: %f - Time Elapsed: %d s' %\
					 (viewer.getEpisodeNumber(), score, time_taken), end="\r")

			score = 0; state = env.reset()
			if len(state.shape) > 1:
				state = state.flatten()

			if episodes % 25 == 0:
				viewer.printMeans()

			if episodes % 200 == 0:
				agent.save()

			if viewer.isFinished():
				break

			if episodes >= target_episodes:
				break
	viewer.saveToFile(image_path)
	test_agent(agent, env)
elif "test":
	test_agent(agent, env)

print("Closing...")
