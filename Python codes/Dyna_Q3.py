import gym
import gym_pathfinding
import numpy as np
import random
import matplotlib.pyplot as plt
from time import sleep

class DynaAgent:

    def __init__(self, exp_rate=0.7, lr=0.9, gamma=0.9, max_epochs=500, n_steps=5, episodes=1):
        self.env = gym.make('pathfinding-obstacle-25x25-v0')
        self.state = self.env.getPlayer()  # (x,y)
        self.actions = [0, 1, 2, 3]  # 0 up, 1 down, 2 left, 3 right
        self.state_actions = []  # state & action track
        self.exp_rate = exp_rate  # Epsilon
        self.lr = lr  # learning rate
        self.gamma = gamma
        self.max_epochs = max_epochs  # maximum trials each episode
        self.success_episodes = 0
        self.convergence_episode = 0
        self.steps = n_steps  # Planning steps
        self.episodes = episodes  # number of episodes going to play
        self.steps_per_episode = []
        self.cumulative_reward_per_episode = []

        self.Q1_values = {}
        self.Q2_values = {}
        # model function
        self.model = {}
        for row in range(self.env.getLines()):
            for col in range(self.env.getColumns()):
                self.Q1_values[(row, col)] = {a: 0 for a in self.actions}
                self.Q2_values[(row, col)] = {a: 0 for a in self.actions}

    def chooseAction(self):
        # epsilon-greedy
        if np.random.uniform(0, 1) <= self.exp_rate:
            return np.random.choice(self.actions)
        else:
            return np.argmax([self.Q1_values[self.state][a] + self.Q2_values[self.state][a] for a in self.actions])

    def reset(self):
        self.env.reset()
        self.env.seed(1)
        self.state = self.env.getPlayer()
        self.state_actions = []

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

                nxtState2D, reward, self.env.game.terminal, _ = self.env.step(action)
                nxtState = self.env.getPlayer()

                cumulative_reward += reward

                # Update Q-values
                if np.random.uniform(0, 1) < 0.5:
                    max_next = np.max(list(self.Q1_values[nxtState].values()))
                    self.Q1_values[self.state][action] += self.lr * (
                            reward + self.gamma * max_next - self.Q1_values[self.state][action])
                else:
                    max_next = np.max(list(self.Q2_values[nxtState].values()))
                    self.Q2_values[self.state][action] += self.lr * (
                            reward + self.gamma * max_next - self.Q2_values[self.state][action])

                # Update model
                if self.state not in self.model:
                    self.model[self.state] = {}
                self.model[self.state][action] = (reward, nxtState)
                self.state = nxtState

                # Loop n times to randomly update Q-values
                for _ in range(self.steps):
                    rand_idx = np.random.choice(range(len(self.model.keys())))
                    _state = list(self.model)[rand_idx]
                    rand_idx = np.random.choice(range(len(self.model[_state].keys())))
                    _action = list(self.model[_state])[rand_idx]

                    _reward, _nxtState = self.model[_state][_action]

                    if np.random.uniform(0, 1) < 0.5:
                        max_next = np.max(list(self.Q1_values[_nxtState].values()))
                        self.Q1_values[_state][_action] += self.lr * (
                                _reward + self.gamma * max_next - self.Q1_values[_state][_action])
                    else:
                        max_next = np.max(list(self.Q2_values[_nxtState].values()))
                        self.Q2_values[_state][_action] += self.lr * (
                                _reward + self.gamma * max_next - self.Q2_values[_state][_action])

                epoches += 1
                if epoches % 100 == 0:
                    print(f"Epoches: {epoches}")
                if self.env.getTerminal():
                    if epoches < self.max_epochs:
                        self.max_epochs = epoches
                        self.convergence_episode = ep
                    self.success_episodes += 1
                    break

            # End of game
            if self.exp_rate > 0.01:
                self.exp_rate *= 0.95
            if ep % 100 == 0:
                print("episode", ep)
            self.steps_per_episode.append(len(self.state_actions))
            self.cumulative_reward_per_episode.append(cumulative_reward)
            self.reset()


if __name__ == "__main__":
    N_EPISODES = 500

    agent = DynaAgent(n_steps=100, episodes=N_EPISODES)
    agent.play()

    steps_episode_100 = agent.steps_per_episode
    cumulative_r_100 = agent.cumulative_reward_per_episode

    # Save the Q-table in a text file
    with open(r'training_test3.txt', 'w+') as f:
        f.write(str(agent.Q1_values))
        f.write('\n')
        f.write(str(agent.Q2_values))

    noEpoches = agent.max_epochs
    conv = agent.convergence_episode
    succ = agent.success_episodes
    successRate = (succ / N_EPISODES) * 100

    plt.figure(1)
    plt.plot(range(N_EPISODES), steps_episode_100, label="step=100")
    plt.xlabel("Episodes")
    plt.ylabel("Steps per episode")
    plt.legend()

    plt.figure(2)
    plt.plot(range(N_EPISODES), cumulative_r_100, label="step=100")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative reward")
    plt.legend()

    print(f"No. epochs: {noEpoches}")
    print(f"Success episodes: {succ}")
    print(f"Success Rate: {successRate}%")
    print(f"Convergence speed after {conv} episodes")

    plt.show()