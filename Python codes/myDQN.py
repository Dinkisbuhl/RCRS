from time import sleep
import gym
import gym_pathfinding
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers, optimizers
from collections import deque
import os

class DQNAgent:
    def __init__(self, state_size, action_size, episodes=200):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.episodes = episodes
        
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = models.Sequential()
        model.add(layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=optimizers.Adam(lr=self.learning_rate))
        return model
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

if __name__ == "__main__":
    env = gym.make('pathfinding-obstacle-11x11-v0')
    env.reset()
    env.reset()
    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    batch_size = 32
    episodes = 50
    
    scores = []  # list to hold the scores (total rewards) of each episode
    steps = []   # list to hold the number of steps taken in each episode

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        score = 0
        for time in range(200):  # replace 500 with the max steps you prefer
            env.render()
            sleep(0.005)
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            score += reward
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            
            if done:
                print(f"episode: {e}/{episodes}, score: {score}, e: {agent.epsilon:.2}")
                scores.append(score)
                steps.append(time)
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        
        # if e % 50 == 0:
        #     agent.save(os.path.join("/mnt/data", f"dqn_pathfinding_{e}.h5"))

    # Plotting the results
    plt.figure(1)
    plt.plot(range(episodes), steps)
    plt.xlabel("Episodes")
    plt.ylabel("Steps per episode")
    plt.title("Steps to complete each episode")
    
    plt.figure(2)
    plt.plot(range(episodes), scores)
    plt.xlabel("Episodes")
    plt.ylabel("Score per episode")
    plt.title("Score of each episode")

    plt.show()