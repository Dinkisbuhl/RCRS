import gym
import gym_pathfinding
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
from time import sleep

class DynaAgent:

    def __init__(self, exp_rate=0.7, lr=0.9, gamma=0.9, max_epochs=500, buffer_size=1000, batch_size=32, episodes=1):
        self.env = gym.make('pathfinding-obstacle-25x25-v0')
        self.state = self.env.getPlayer() # (x,y)
        self.actions = [0, 1, 2, 3] # 0 up, 1 down, 2 left, 3 right
        self.state_actions = [] # state & action track
        self.exp_rate = exp_rate # Epsilon
        self.lr = lr # learning rate
        self.gamma = gamma                  
        self.max_epochs = max_epochs # maximum trials each episode
        self.success_episodes = 0           
        self.convergence_episode = 0               
        self.episodes = episodes # number of episodes going to play
        self.steps_per_episode = []
        self.cumulative_reward_per_episode = []

        self.Q_values = {}
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

        for row in range(self.env.getLines()):
            for col in range(self.env.getColumns()):
                self.Q_values[(row, col)] = {}
                for a in self.actions:
                    self.Q_values[(row, col)][a] = 0
        
    def chooseAction(self):
        # epsilon-greedy
        mx_nxt_reward = -999
        action = ""
        
        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions)
        else:
            # greedy action
            current_position = self.state
            if len(set(self.Q_values[current_position].values())) == 1:
                action = np.random.choice(self.actions)
            else:
                for a in self.actions:
                    nxt_reward = self.Q_values[current_position][a]
                    if nxt_reward >= mx_nxt_reward:
                        action = a
                        mx_nxt_reward = nxt_reward
        return action
    
    def reset(self):
        self.env.reset()
        self.env.seed(1)
        self.state = self.env.getPlayer()
        self.state_actions = []

    def experience_replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)

        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.max(list(self.Q_values[next_state].values()))
            self.Q_values[state][action] += self.lr * (target - self.Q_values[state][action])


    def play(self):
        self.steps_per_episode = []
        self.cumulative_reward_per_episode = []
        self.reset()
        
        for ep in range(self.episodes):
            epoches = 0
            cumulative_reward = 0
            while epoches < self.max_epochs:
                self.env.render()
                sleep(0.005)
                action = self.chooseAction()
                self.state_actions.append((self.state, action))

                next_state_2d, reward, done, _ = self.env.step(action)
                next_state = self.env.getPlayer()

                cumulative_reward += reward

                self.memory.append((self.state, action, reward, next_state, done))
                self.experience_replay()

                self.state = next_state

                epoches += 1
                if epoches % 100 == 0:
                    print(f"Epoches: {epoches}")

                if done:
                    if epoches < self.max_epochs:
                        self.max_epochs = epoches
                        self.convergence_episode = ep
                    self.success_episodes += 1
                    break

            if self.exp_rate > 0.01:
                self.exp_rate *= 0.95

            if ep % 100 == 0:
                print("episode", ep)

            self.steps_per_episode.append(len(self.state_actions))
            self.cumulative_reward_per_episode.append(cumulative_reward)
            self.reset()

if __name__ == "__main__":
    N_EPISODES = 500

    agent = DynaAgent(episodes=N_EPISODES)
    agent.play()

    steps_episode = agent.steps_per_episode
    cumulative_r = agent.cumulative_reward_per_episode

    with open('training_test5.txt', 'w+') as f:
        f.write(str(agent.Q_values))

    noEpoches = agent.max_epochs
    conv = agent.convergence_episode
    succ = agent.success_episodes
    successRate = (succ / N_EPISODES) * 100

    plt.figure(1)
    plt.plot(range(N_EPISODES), steps_episode, label="Steps per episode")
    plt.xlabel("Episodes")
    plt.ylabel("Steps per episode")
    plt.legend()
    
    plt.figure(2)
    plt.plot(range(N_EPISODES), cumulative_r, label="Cumulative reward")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative reward")
    plt.legend()

    print(f"No. epoches: {noEpoches}")
    print(f"Succsess episodes: {succ}")
    print(f"Success Rate: {successRate}")
    print(f"Convergence speed After {conv} episodes")
    
    plt.show()