import gym
import gym_pathfinding
import numpy as np
import random
import matplotlib.pyplot as plt
from time import sleep

class DynaAgent:

    def __init__(self, exp_rate=0.7, lr=0.9, gamma=0.9, max_epochs=500, n_steps=5, episodes=1):
        self.env = gym.make('pathfinding-obstacle-25x25-v0')
        self.state = self.env.getPlayer()
        self.actions = [0, 1, 2, 3]
        self.state_actions = []
        self.exp_rate = exp_rate
        self.lr = lr
        self.gamma = gamma
        self.max_epochs = max_epochs
        self.success_episodes = 0
        self.convergence_episode = 0
        self.steps = n_steps
        self.episodes = episodes
        self.steps_per_episode = []
        self.cumulative_reward_per_episode = []

        self.Q1_values = {}
        self.Q2_values = {}
        self.model = {}  # Environment model: {state: {action: (reward, next_state)}}
        # Initialize Q-values for all state-action pairs
        for row in range(self.env.getLines()):
            for col in range(self.env.getColumns()):
                self.Q1_values[(row, col)] = {a: 0 for a in self.actions}
                self.Q2_values[(row, col)] = {a: 0 for a in self.actions}

    def chooseAction(self):
        # Epsilon-greedy action selection
        if np.random.uniform(0, 1) <= self.exp_rate:
            return np.random.choice(self.actions)
        else:
            return np.argmax([self.Q1_values[self.state][a] + self.Q2_values[self.state][a] for a in self.actions])

    def reset(self):
        # Reset environment and agent state
        self.env.reset()
        self.env.seed(1)
        self.state = self.env.getPlayer()
        self.state_actions = []

    def play(self):
        self.steps_per_episode = []
        self.cumulative_reward_per_episode = []
        for ep in range(self.episodes):
            self.reset()
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

                # Update Q-values using Q-learning with Double Q-learning
                if np.random.uniform(0, 1) < 0.5:
                    max_next = np.max(list(self.Q1_values[nxtState].values()))
                    self.Q1_values[self.state][action] += self.lr * (
                            reward + self.gamma * max_next - self.Q1_values[self.state][action])
                else:
                    max_next = np.max(list(self.Q2_values[nxtState].values()))
                    self.Q2_values[self.state][action] += self.lr * (
                            reward + self.gamma * max_next - self.Q2_values[self.state][action])

                # Update environment model
                if self.state not in self.model:
                    self.model[self.state] = {}
                self.model[self.state][action] = (reward, nxtState)
                self.state = nxtState

                # Perform n planning steps using model
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

            # Decrease exploration rate over episodes
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

    # Save the Q-values in a text file
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