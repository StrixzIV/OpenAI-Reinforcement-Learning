import gym
import numpy as np
from matplotlib import pyplot as plt
from tabulate import tabulate

class Agent():

    def __init__(self):
        self.qTable = np.zeros((5, 2))
        self.learningRate = 0.05
        self.discountFactor = 0.95
        self.epsilon = 0.5
        self.decayFactor = 0.999
        self.rewardOfEpisode = []
    
    def autoPlay(self, env, number_of_episode = 20):

        for i in range(number_of_episode):
            print('Episode {} of {}'.format(i + 1, number_of_episode))
            state = env.reset()
            endGame = False

            self.epsilon *= self.decayFactor
            total_reward = 0

            while not endGame:

                if self._qTableIsEmpty(state) or self._probability(self.epsilon):
                    action = self._actionRandomer(env)

                else:
                    action = self._getHighestRewardExpectation(state)
                
                newState, reward, endGame, _ = env.step(action)
                self.qTable[state, action] += self.learningRate * (reward + self.discountFactor * self._getExpectedRewrd(newState) - self.qTable[state, action])
                total_reward += reward
                state = newState

            self.rewardOfEpisode.append(total_reward)
            print(tabulate(self.qTable, showindex = 'always', headers = ['State', 'Action 0(Forward 1 step)', 'Action 1(Back to 0)']))

    def _qTableIsEmpty(self, state):
        return np.sum(self.qTable[state, :]) == 0

    def _probability(self, probability):
        return np.random.random() < probability

    def _actionRandomer(self, env):
        return env.action_space.sample()

    def _getHighestRewardExpectation(self, state):
        return np.argmax(self.qTable[state, :])

    def _getExpectedRewrd(self, state):
        return np.max(self.qTable[state, :])


env = gym.make('NChain-v0')
agent = Agent()

agent.autoPlay(env)

plt.plot(agent.rewardOfEpisode)

plt.title('Performance over time')
plt.xlabel('Episode')
plt.ylabel('Total reward')

plt.show()
