import math
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed
from keras.layers import SimpleRNN
from keras.layers import Dropout
from keras.optimizers import Adam
import gym
import gym_pathfinding
import matplotlib.pyplot as plt
from time import sleep

# Number of episodes to train
EPISODES = 200

class DQNAgent:

    def __init__(self, state_size, action_size):
        # Initial max memory and other hyper parameters
        self.state_size = state_size
        self.action_size = action_size
        self.memory = list()
        self.max_mem = 1000 # max memory decreased to protect against old bad learnt routes
        self.gamma = 0.95   # discount rate lowered slightly to improve perf in 11x11 scenarios
        self.epsilon = 0.9 # exploration rate reduced for same reason
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.98
        self.learning_rate = 0.003 # learning rate increased 
        self.replay_cnt = 0 
        self.actions = [0,1,2,3]

        self.target_model = self._build_model()
        self.model = self._build_model()
        self.update_model = 10

    def _build_model(self):
        # NN to approximate Q function
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu')) # less dense network with rectified linear activation function
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        # Store experience learnt
        self.memory.append((state, action, reward, next_state, done))
        if(len(self.memory)>self.max_mem) :
            self.memory.pop(0)

    def act(self, state):
        # pick one action, up, down, left or right
        if np.random.uniform(0,1) <= self.epsilon:
            return random.choice(range(self.action_size) )
        act_values = self.model.predict(state)[0]
        #print(self.model.predict(state))
        return np.argmax(act_values)  # returns action

    def replay(self, batch_size):
        self.replay_cnt+=1 
        # Train model using random sampled experience from replay buffer
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward # q value of the state  if it is the final 
            if not done:
                target = (reward + self.gamma *
                          np.max(self.target_model.predict(next_state)[0])) # q value of state is computed by bellman
            target_f = self.target_model.predict(state) # value of the current state q_value 
            target_f[0][action] = target #
            self.model.fit(state, target_f, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        if (self.replay_cnt % agent.update_model ==0) :
                agent.target_train() 

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.target_model.set_weights(target_weights)

    def load(self, name):   
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def reset(self):
        self.env.reset()
        self.env.seed(1)
        # self.env.game.player = self.randomStart()
        self.state = self.env.getPlayer()
        self.state_actions = []



if __name__ == "__main__":
    env = gym.make('pathfinding-obstacle-11x11-v0')
    steps_per_episode = []
    cumulative_reward_per_episode = []
    env.reset()
    env.reset()

    state_size= 625

    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)

    done = False
    batch_size = 64

    print("START DQN")
    donecount=0 
    for e in range(EPISODES):

        reward_sum = 0

        state = env.reset()

        env.seed(1)

        state = np.zeros((1, state_size))
        player_pos = env.getPlayer()
        state[0, player_pos[0]*25 + player_pos[1]] = 1

        state = np.reshape(state, [1, state_size])

        steps = 0
        while steps < 300:

            env.render()
            sleep(0.001)

            action = agent.act(state)

            next_state, reward, done, _ = env.step(action)

            next_state = [[0]*state_size]*1
            next_state[0][env.getPlayer()[0]*25 + env.getPlayer()[1]] = 1

            next_state = np.reshape(next_state, [1, state_size])
            if(done == True):
                donecount+=1
                reward += abs(reward * 0.05)
            reward_sum += reward
            
               # below, useful if running into memory limits and debugging
               # print("memory length", len(agent.memory))
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                break
            steps += 1
            
        print("episode: {}/{}, score: {}, e: {:.2} iteration:{}"
                  .format(e, EPISODES, reward_sum, agent.epsilon, steps))
        
        if len(agent.memory) > batch_size & donecount>0:
                agent.replay(batch_size)
        #if e % 1000 == 0:
        #     agent.save("./save/dqn" + str(e) + ".h5")
        # above, option to write to disk periodically
        steps_per_episode.append(steps)
        cumulative_reward_per_episode.append(reward_sum)

    # Calculate average cumulative rewards
    average_cumulative_rewards = np.mean(cumulative_reward_per_episode)

    plt.figure(1)
    plt.plot(range(EPISODES), steps_per_episode)
    plt.xlabel("Episodes")
    plt.ylabel("Steps per episode")
    plt.ylim(0, 500)

    plt.figure(2)
    plt.plot(range(EPISODES), cumulative_reward_per_episode)
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative reward")
    # Printing the average cumulative rewards
    print("Average Cumulative Reward:", average_cumulative_rewards)

    plt.show()