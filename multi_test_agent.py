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
import tensorflow as tf
import winsound

# Used to hide the Tensorflow warning
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def test_agent(tested_agent, env, render=False, test_episodes=1000, test=True):
    score = 0
    total_score = 0
    episode = 0
    state = env.reset(training=not test)
    if render:
        env.render()
    tested_agent.reset_mem()
    if len(state.shape) > 1:
        state = state.flatten()

    solved = 0.0

    print("\n Starting Test \n")

    while episode < test_episodes:
        action = tested_agent.act(state, testing=True)
        next_state, reward, done, info = env.step(action)
        total_score += reward
        score += reward

        state = next_state
        if len(state.shape) > 1:
            state = state.flatten()
        state = np.true_divide(state, input_scaling)
        if render:
            env.render()

        if done:
            episode += 1
            state = env.reset(training=not test)
            if render:
                env.render()
            if len(state.shape) > 1:
                state = state.flatten()
            tested_agent.reset_mem()

            if score > 0:
                solved += 1
            score = 0

    print("\nModel Score is %.2f" % (total_score / test_episodes))

    return total_score / test_episodes, solved/test_episodes


print("\n======= RL Agent Testing with the OpenAI gym =======\n")

if len(sys.argv) < 3:
    print("Usage: <operation> <agent_class> <environment>")
    print("Optional: <target_episodes>")
    sys.exit(0)

operation = sys.argv[1]
agent_class_name = sys.argv[2]
environment = sys.argv[3]
target_score = 1
target_episodes = 1000
input_scaling = 1

stacking_states = 10

if len(sys.argv) == 5:
    target_episodes = int(sys.argv[4])

# Importing the given class
module = importlib.import_module(agent_class_name)
class_tuples = inspect.getmembers(module, inspect.isclass)
agents_list = []
for el in class_tuples:
    if 'Agent' in el[0]:
        agents_list += [el]

agent_tuple_input = agents_list[0]
agent_class_input = agent_tuple_input[1]
print('Loading class: ' + agent_tuple_input[0])
print('Load gym environment: ' + environment + "\n")


def train_and_test(agent_tuple, agent_class, index):
    # Preparing image target
    image_path = "images/" + agent_tuple[0] + "_" + environment + "_" + str(index) + ".png"

    # Created user defined environment
    env = gym.make(environment)

    # Creating the Gym environment
    state = env.reset()
    if len(state.shape) > 1:
        state = state.flatten()
    state = np.true_divide(state, input_scaling)

    state_size = state.shape[0]
    if type(env.action_space) == type(gym.spaces.MultiDiscrete([])):
        original_action_size = env.action_space.nvec
        action_size = np.prod(original_action_size)
        action_size = np.asscalar(action_size)
    else:
        action_size = env.action_space.n

    print("State size is %d, action size is %d" % (state_size, action_size))
    print("Experiment N° %d" % index)

    step = 0
    score = 0
    episodes = 0
    start_time = time.time()

    tf.keras.backend.clear_session()
    agent = agent_class(state_size, action_size, load=(operation == "test"), envname=environment)
    # tested_performance = test_agent(agent, env)
    viewer = sv.scoreViewer(target_score=target_score)  # tested_performance)

    while True:

        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        score += reward

        if len(next_state.shape) > 1:
            next_state = next_state.flatten()
        next_state = np.true_divide(next_state, input_scaling)

        obj = {"state": state,
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

    train_score, train_solved = test_agent(agent, env, render=False, test_episodes=100, test=False)
    test_score,  test_solved  = test_agent(agent, env, render=False, test_episodes=100, test=True)

    return train_score, train_solved, test_score, test_solved, episodes


experiments = 10
total_train_score, mean_episodes, total_train_solved = 0, 0, 0
total_test_score, total_test_solved = 0, 0

start_time = time.time()

for i in range(experiments):

    train_score, train_solved, test_score, test_solved, episodes_taken = train_and_test(agent_tuple_input, agent_class_input, i)
    total_train_score  += train_score
    total_train_solved += train_solved
    total_test_score   += test_score
    total_test_solved  += test_solved
    mean_episodes += episodes_taken

print("=======================Results===========================")
print("----------------------Training---------------------------")
print("Score: %.3f - Solved: %.4f - Mean Episodes required %d" %
      (total_train_score/experiments, total_train_solved/experiments, mean_episodes/experiments))

print("------------------------Test-----------------------------")
print("Score: %.3f - Solved: %.4f - Mean Episodes required %d" %
      (total_test_score/experiments, total_test_solved/experiments, mean_episodes/experiments))

print("Time Taken: %d s" % (time.time()-start_time))

print("=========================================================")

for i in range(4):
    winsound.Beep(i*500 + 500, 100)
