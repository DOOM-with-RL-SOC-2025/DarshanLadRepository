import numpy as np
from base import Agent, MultiArmedBandit
import matplotlib.pyplot as plt

class UCBAgent(Agent):
    def __init__(self, time_horizon, bandit: MultiArmedBandit):
        super().__init__(time_horizon, bandit)
        self.reward_memory = np.zeros(self.arms)
        self.count_memory = np.zeros(self.arms)
        self.time_step = 0

    def give_pull(self):
        if self.time_step < self.arms:
            # Pull each arm once initially
            reward = self.bandit.pull(self.time_step)
            self.reinforce(reward, self.time_step)
            return

        ucb_values = np.zeros(self.arms)
        for i in range(self.arms):
            if self.count_memory[i] == 0:
                ucb_values[i] = float('inf')  # Force exploration
            else:
                mean_reward = self.reward_memory[i] / self.count_memory[i]
                explore = np.sqrt((2 * np.log(self.time_step + 1)) / self.count_memory[i])
                ucb_values[i] = mean_reward + explore

        chosen_arm = np.argmax(ucb_values)
        reward = self.bandit.pull(chosen_arm)
        self.reinforce(reward, chosen_arm)

    def reinforce(self, reward, arm):
        self.reward_memory[arm] += reward
        self.count_memory[arm] += 1
        self.time_step += 1
        self.rewards.append(reward)

    def plot_arm_graph(self):
        indices = np.arange(self.arms)

        plt.figure(figsize=(12, 6))
        plt.bar(indices, self.count_memory, color='cornflowerblue', edgecolor='black')
        plt.title('UCB Arm Pull Counts', fontsize=16)
        plt.xlabel('Arm', fontsize=14)
        plt.ylabel('Pull Count', fontsize=14)
        plt.grid(axis='y')
        plt.xticks(indices, [f'Arm {i}' for i in indices], rotation=45)

        for i, count in enumerate(self.count_memory):
            plt.text(i, count + 0.5, str(int(count)), ha='center', va='bottom', fontsize=12)

        plt.tight_layout()
        plt.show()



# Code to test
if __name__ == "__main__":
    # Init Bandit
    TIME_HORIZON = 10_000
    bandit = MultiArmedBandit(np.array([0.23,0.55,0.76,0.44]))
    agent = UCBAgent(TIME_HORIZONE, bandit) ## Fill with correct constructor

    # Loop
    for i in range(TIME_HORIZON):
        agent.give_pull()

    # Plot curves
    agent.plot_reward_vs_time_curve()
    agent.plot_arm_graph()
    bandit.plot_cumulative_regret()
