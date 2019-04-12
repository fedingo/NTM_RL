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
from NTMAgent_2O import NTMAgent

def test_agent(tested_agent, env):
    score = 0
    total_score = 0
    episode = 0
    test_episodes = 5
    state = env.reset()
    env.render()
    tested_agent.reset_mem()
    if len(state.shape) > 1:
        state = state.flatten()
    state = np.true_divide(state, scale_input_constant)

    print("\n Starting Test \n")

    while episode < test_episodes:
        action = tested_agent.act(state, testing=True)
        if action_reshape:
            action_ = np.unravel_index(action, original_action_size)
            next_state, reward, done, info = env.step(action_)
        else:
            next_state, reward, done, info = env.step(action)
        total_score += reward;
        score += reward

        state = next_state
        if len(state.shape) > 1:
            state = state.flatten()
        state = np.true_divide(state, scale_input_constant)
        env.render()

        if done:
            episode += 1
            state = env.reset()
            env.render()
            if len(state.shape) > 1:
                state = state.flatten()
            tested_agent.reset_mem()
            print("episode done: score %d" % score)
            score = 0

    print("\nModel Score is %.2f" % (total_score / test_episodes))

    return total_score / test_episodes


agent_class = NTMAgent
environment = "UnblockMeCompactFixedMap-v0" # "UnblockMeListedFixedMap-v0" # "UnblockMeFixedMap-v0" #
operation = "train"
target_episodes = 500

scale_input_constant = 1

# Preparing image target
image_path = "images/NTMAgent" + "_" + environment + ".png"

# Created user defined environment
print('Load gym environment: ' + environment + "\n")
env = gym.make(environment)

# Creating the Gym environment
state = env.reset()
if len(state.shape) > 1:
    state = state.flatten()
state = np.true_divide(state, scale_input_constant)

state_size = state.shape[0]
action_reshape = False
if type(env.action_space) == type(gym.spaces.MultiDiscrete([])):
    action_size = env.action_space.nvec
else:
    action_size = env.action_space.n

print("State size is %d, action size is %s" % (state_size, str(action_size)))

step = 0;
score = 0;
episodes = 0
start_time = time.time()

agent = agent_class(state_size, action_size, load=(operation == "test"), envname=environment)
# tested_performance = test_agent(agent, env)
viewer = sv.scoreViewer(target_score=0, lowest_score = -100)  # tested_performance)

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

        next_state = np.true_divide(next_state, scale_input_constant)

        obj = { \
            "state": state,
            "action": action,
            "reward": reward,
            "done": done,
            "next_state": next_state
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
            print('Episode: %d - Score: %f - Time Elapsed: %d s' % \
                  (viewer.getEpisodeNumber(), score, time_taken), end="\r")

            score = 0
            state = env.reset()
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
