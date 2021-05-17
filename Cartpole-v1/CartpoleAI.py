import gym
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tabulate import tabulate

class Agent():

    def __init__(self):

        n_bins = 10

        self.cartPosition_bins = pd.cut([-1.3, 1.3], bins = n_bins, retbins = True)[1][1:-1]
        self.cartVelocity_bins = pd.cut([-3, 3], bins = n_bins, retbins = True)[1][1:-1]
        self.poleAngle_bins = pd.cut([-0.3, 0.3], bins = n_bins, retbins = True)[1][1:-1]
        self.angleRate_bins = pd.cut([-3, 3], bins = n_bins, retbins = True)[1][1:-1]

        self.qTables = np.zeros((n_bins, n_bins, n_bins, n_bins) + (2, ))
        self.learningRate = 0.05
        self.discountFactor = 0.95
        self.decayFactor = 0.999
        self.epsilon = 0.5
        self.episodeReward = []

    def autoPlay(self, env, numberOfEpisode = 3000, isRender = False):

        for i in range(numberOfEpisode):
            print('Episode {} of {}'.format(i + 1, numberOfEpisode))
            observation = env.reset()
            cartPosition, cartVelocity, poleAngle, angleRateOfChange = observation

            state = (self.toBin(cartPosition, self.cartPosition_bins), self.toBin(cartVelocity, self.cartVelocity_bins), self.toBin(poleAngle, self.poleAngle_bins), self.toBin(angleRateOfChange, self.angleRate_bins))

            self.epsilon *= self.decayFactor
            totalReward = 0

            endGame = False

            while not endGame:

                if isRender:
                    env.render()

                if self.qTablesIsEmpty(state) or self.probability(self.epsilon):
                    action = self.getRandomAction(env)

                else:
                    action = self.getHighestExpectedRewardAction(state)

                newObservation, reward, endGame, _ = env.step(action)

                if endGame:
                    reward = -200

                else:
                    totalReward += reward

                newCartPosition, newCartVelocity, newPoleAngle, newAngleRateOfChange = newObservation
                newState = (self.toBin(newCartPosition, self.cartPosition_bins), self.toBin(newCartVelocity, self.cartVelocity_bins), self.toBin(newPoleAngle, self.poleAngle_bins), self.toBin(newAngleRateOfChange, self.angleRate_bins))

                self.qTables[state][action] += self.learningRate * (reward + self.discountFactor * self.getExpectedReward(newState) - self.qTables[state][action])
                state = newState

            self.episodeReward.append(totalReward)

    def qTablesIsEmpty(self, state):
        return np.sum(self.qTables[state])

    def probability(self, probability):
        return np.random.random() < probability

    def getRandomAction(self, env):
        return env.action_space.sample()

    def getHighestExpectedRewardAction(self, state):
        return np.argmax(self.qTables[state])

    def getExpectedReward(self, state):
        return np.max(self.qTables[state])

    def toBin(self, value, bins):
        return np.digitize(x = value, bins = bins)

env = gym.make('CartPole-v1')
agent = Agent()

agent.autoPlay(env)
agent.autoPlay(env, numberOfEpisode = 1, isRender = True)
env.close()

plt.plot(agent.episodeReward)

plt.title('Performance over time')
plt.xlabel('Episode')
plt.ylabel('Total Reward')

plt.show()
