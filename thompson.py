import numpy as np
from base import Agent, MultiArmedBandit
import matplotlib.pyplot as plt

class ThompsonSamplingAgent(Agent):
    def __init__(self, time_horizon, bandit: MultiArmedBandit):
        super().__init__(time_horizon, bandit)
        self.successes = np.ones(self.arms)  # Alpha = 1 for prior (Beta(1,1))
        self.failures = np.ones(self.arms)  # Beta = 1 for prior
        self.time_step = 0

    def give_pull(self):
        # Sample from Beta(alpha, beta) for each arm
        samples = np.random.beta(self.successes, self.failures)
        chosen_arm = np.argmax(samples)
        reward = self.bandit.pull(chosen_arm)
        self.reinforce(reward, chosen_arm)

    def reinforce(self, reward, arm):
        if reward == 1:
            self.successes[arm] += 1
        else:
            self.failures[arm] += 1
        self.time_step += 1
        self.rewards.append(reward)

    def plot_arm_graph(self):
        total_pulls = self.successes + self.failures - 2  # subtract prior
        indices = np.arange(len(total_pulls))

        plt.figure(figsize=(12, 6))
        plt.bar(indices, total_pulls, color='orange', edgecolor='black')
        plt.title('Thompson Sampling Arm Pull Counts', fontsize=16)
        plt.xlabel('Arm', fontsize=14)
        plt.ylabel('Pull Count', fontsize=14)
        plt.grid(axis='y')
        plt.xticks(indices, [f'Arm {i}' for i in indices], rotation=45)

        for i, count in enumerate(total_pulls):
            plt.text(i, count + 0.5, str(int(count)), ha='center', va='bottom', fontsize=12)

        plt.tight_layout()
        plt.show()



# Code to test
if __name__ == "__main__":
    # Init Bandit
    TIME_HORIZON = 10_000
    bandit = MultiArmedBandit(np.array([0.23,0.55,0.76,0.44]))
    agent = ThompsonSamplingAgent(TIME_HORIZON, bandit) ## Fill with correct constructor

    # Loop
    for i in range(TIME_HORIZON):
        agent.give_pull()

    # Plot curves
    agent.plot_reward_vs_time_curve()
    agent.plot_arm_graph()
    bandit.plot_cumulative_regret()
