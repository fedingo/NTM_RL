import random
import numpy as np
from abstractAgent import *
import matplotlib as plt

from NN_models.MQN_model import *


class MQNAgent(abstractAgent):

    def __init__(self, state_size, action_size, load=False, envname="tmp"):

        self.exploration_rate = 1
        self.exploration_decay = 0.995
        self.exploration_min = 0.02

        self.train_rate = 8
        self.update_frozen = 500
        self.batch_size = 1
        self.double = True

        self.stacked_states = None
        self.load = load
        self.show_memory = True

        self.sequence_length = 48
        self.envname = envname

        learning_rate = 0

        architecture = {'conv': [4, 16],
                        'fc': [64, 64]
                        }

        super().__init__(state_size, action_size, architecture, learning_rate=learning_rate, memSize = 1000)

    def __reshaped_state_size__(self, dim=1):
        result = 0

        if type(self.state_size) == type(1):
            result = [1, dim, self.state_size]
        elif type(self.state_size) == type(list()):
            result = [1, dim] + self.state_size
        else:
            raise Exception("State_size type not supported")

        return result

    def build(self):
        self.model = MQN_model(self.state_size, self.action_size, envname=self.envname, load=self.load)

        self.model.update_frozen()

    # function to define when to train the network
    def hasToTrain(self, step, done, episode):

        if step % self.update_frozen == 0:
            self.sync_models()

        if done:
            self.decay_explore()

        return step % self.train_rate == 0

    def train(self):

        if len(self.memory) < self.mem_size:
            return

        index_list = range(len(self.memory) - 500)
        minibatch = random.sample(index_list, self.batch_size)

        x_train = []
        x_next = []
        y_train = []

        for index in minibatch:

            while index < len(self.memory) and not self.memory[index]['done']:
                index += 1

            if index == len(self.memory):
                return

            obj_list = [self.memory[i] for i in range(index + 1, len(self.memory))]
            obj = []
            for o in obj_list:
                obj.append(o)
                if o['done'] or len(obj) > self.sequence_length:
                    break

            # print("\n%d -> %d (%d)" % (index, len(obj), len(obj_list)))
            # print(len(self.memory) - index)

            state = [o['state'] for o in obj]
            action = [o['action'] for o in obj]  # obj['action']
            reward = [o['reward'] for o in obj]  # obj['reward']
            next_s = [o['next_state'] for o in obj]  # obj['next_state']
            done = [o['done'] for o in obj]  # obj['done']

            current_sequence_length = len(obj)

            stateSize = self.__reshaped_state_size__(dim=current_sequence_length)

            data_train = np.reshape(state, stateSize)
            target, _ = self.model.predict(data_train)

            data_next = np.reshape(next_s, stateSize)
            if self.double:
                next_target, _ = self.model.predict(data_next)
                next_target_frozen = self.model.predict_frozen(data_next)

                max_next_target = []
                for i in range(len(next_target)):
                    max_next_target.append(next_target_frozen[i, np.argmax(next_target[i])])
            else:
                max_next_target = np.max(self.model.predict_frozen(data_next), axis=0)

            for i in range(len(target)):
                target[i, action[i]] = reward[i] + self.gamma * max_next_target[i]

                # target[:,action] = reward + self.gamma*max_next_target
                if done[i]:
                    target[i, action[i]] = reward[i]

            x_train.append(data_train)
            y_train.append(target)

        stateSizeBatch = self.__reshaped_state_size__(current_sequence_length)
        x_train = np.array(x_train)
        x_train = np.reshape(x_train, stateSizeBatch)
        y_train = np.array(y_train)
        y_train = np.reshape(y_train, [self.batch_size, current_sequence_length, self.action_size])

        loss = self.model.train_on_batch(x_train, y_train)

    def act(self, state, testing=False, display_mem=False):

        stateSize = self.__reshaped_state_size__()
        state = np.reshape(state, stateSize)

        if self.stacked_states is None:
            self.stacked_states = state
        else:
            self.stacked_states = np.concatenate([self.stacked_states, state], axis = 1)

        q_values, memory = self.model.predict(self.stacked_states)

        if np.random.uniform(0, 1) < self.exploration_rate and not testing:
            action = np.random.randint(self.action_size)
        else:
            action = np.argmax(q_values[-1])

        if testing and self.show_memory and display_mem:
            attentions = memory.read_attentions

            fig, (ax1, ax2) = plt.subplots(2, 1, num=2)
            ax1.matshow(np.transpose(memory.M), vmin=-1, vmax=1)
            ax2.matshow(attentions, vmin=0, vmax=1)

        return action

    def reset_mem(self):
        self.stacked_states = None

    def decay_explore(self):
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay
        else:
            if self.exploration_rate != 0:
                print("\n No more Exploration!")
                self.exploration_rate = 0

    def sync_models(self):
        self.model.update_frozen()

    def save(self):
        self.model.save_model()