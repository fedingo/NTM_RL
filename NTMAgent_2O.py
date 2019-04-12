import random
import numpy as np
from abstractAgent import *
import matplotlib as plt

from NN_models.NTM_2_outputs import *
from NN_models.LSTM_model import *


class NTMAgent(abstractAgent):

    def __init__(self, state_size, action_size, load=False, envname="tmp"):
        print("Creating a DQN Agent")

        self.exploration_rate = 1
        self.exploration_decay = 0.99
        self.exploration_min = 0.03

        self.train_rate = 16
        self.update_frozen = 1000
        self.batch_size = 1
        self.double = True

        self.lstm = False
        self.load = load

        self.sequence_length = 100
        self.envname = envname

        self.current_memory = None

        learning_rate = 0
        mem_size = 800

        architecture = {'conv': [4, 16],
                        'fc': [64, 64]
                        }

        super().__init__(state_size, action_size, architecture, learning_rate=learning_rate, memSize = mem_size)

    def __reshaped_state_size__(self, dim=1, batch_size=1):
        result = 0

        if type(self.state_size) == type(1):
            result = [self.batch_size, dim, self.state_size]
        elif type(self.state_size) == type(list()):
            result = [self.batch_size, dim] + self.state_size
        else:
            raise Exception("State_size type not supported")

        return result

    def build(self):

        if self.lstm:
            self.model = LSTM_model(self.state_size, self.action_size, batch_size = self.batch_size)
        else:
            self.model = NTM_model(self.state_size, self.action_size, envname=self.envname, load=self.load,
                                   batch_size = self.batch_size)

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

        index_list = range(len(self.memory) - self.sequence_length)
        minibatch = random.sample(index_list, self.batch_size)

        x_train = []
        x_next = []
        y_train = []

        for index in minibatch:

            while index > 0 and not self.memory[index]['done']:
                index -= 1


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
                    x = np.argmax(next_target[i])
                    x = np.unravel_index(x, next_target[i].shape)
                    max_next_target.append(next_target_frozen[(i,) + x])
            else:
                max_next_target = np.max(self.model.predict_frozen(data_next), axis=0)

            for i in range(len(target)):
                target[(i,) + tuple(action[i])] = reward[i] + self.gamma * max_next_target[i]

                # target[:,action] = reward + self.gamma*max_next_target
                if done[i]:
                    target[(i,) + tuple(action[i])] = reward[i]

            x_train.append(data_train)
            x_next.append(data_next)
            y_train.append(target)

        stateSizeBatch = self.__reshaped_state_size__(current_sequence_length)
        x_train = np.array(x_train)
        x_train = np.reshape(x_train, stateSizeBatch)
        y_train = np.array(y_train)
        y_train = np.reshape(y_train, (self.batch_size,current_sequence_length)+tuple(self.action_size))

        if self.lstm:
            self.model.train_on_batch(x_train, y_train)
        else:
            loss, predict_loss = self.model.train_on_batch(x_train, y_train)

    def act(self, state, testing=False):

        stateSize = self.__reshaped_state_size__()
        state = np.reshape(state, stateSize)
        q_values, self.current_memory = self.model.predict(state, mem=self.current_memory)

        if np.random.uniform(0, 1) < self.exploration_rate and not testing:
            action = (np.random.rand(len(self.action_size))*self.action_size).astype(int)
        else:
            action = np.argmax(q_values[-1])
            action = np.unravel_index(action, q_values[-1].shape)

        if testing:
            # plt.figure(num=99)
            plt.clf()
            # axes = plt.gca()
            # axes.set_ylim([-10, 10])
            # plt.bar(range(4), q_values[0], align='center', alpha=0.5)
            plt.matshow(self.current_memory[0], fignum=0, cmap=plt.cm.gray)

        return action

    def reset_mem(self):
        self.current_memory = None

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