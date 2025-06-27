import numpy as np
from base import Agent, MultiArmedBandit
import matplotlib.pyplot as plt

class KLUCBAgent(Agent):
    def __init__(self, time_horizon, bandit: MultiArmedBandit, c=3):
        super().__init__(time_horizon, bandit)
        self.reward_memory = np.zeros(len(bandit.arms))
        self.count_memory = np.zeros(len(bandit.arms))
        self.time_step = 0
        self.c = c  # exploration constant

    def kl_divergence(self, p, q):
        eps = 1e-15  # avoid log(0)
        p = min(max(p, eps), 1 - eps)
        q = min(max(q, eps), 1 - eps)
        return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

    def solve_upper_bound(self, p_hat, count, time):
        upper_bound = (np.log(time) + self.c * np.log(np.log(max(time, 2)))) / count
        # Binary search for q such that KL(p_hat, q) <= upper_bound
        low = p_hat
        high = 1.0
        for _ in range(25):  # 25 iterations for sufficient precision
            mid = (low + high) / 2
            if self.kl_divergence(p_hat, mid) > upper_bound:
                high = mid
            else:
                low = mid
        return low

    def give_pull(self):
        if self.time_step < self.arms:
            # Initialize by pulling each arm once
            reward = self.bandit.pull(self.time_step)
            self.reinforce(reward, self.time_step)
            return

        ucb_values = np.zeros(self.arms)
        for i in range(self.arms):
            if self.count_memory[i] == 0:
                ucb_values[i] = 1.0
            else:
                p_hat = self.reward_memory[i] / self.count_memory[i]
                ucb_values[i] = self.solve_upper_bound(p_hat, self.count_memory[i], self.time_step + 1)

        chosen_arm = np.argmax(ucb_values)
        reward = self.bandit.pull(chosen_arm)
        self.reinforce(reward, chosen_arm)

    def reinforce(self, reward, arm):
        self.count_memory[arm] += 1
        self.reward_memory[arm] += reward
        self.time_step += 1
        self.rewards.append(reward)

    def plot_arm_graph(self):
        counts = self.count_memory
        indices = np.arange(len(counts))

        plt.figure(figsize=(12, 6))
        plt.bar(indices, counts, color='lightgreen', edgecolor='black')
        plt.title('KL-UCB Arm Pull Counts', fontsize=16)
        plt.xlabel('Arm', fontsize=14)
        plt.ylabel('Pull Count', fontsize=14)
        plt.grid(axis='y')
        plt.xticks(indices, [f'Arm {i}' for i in indices], rotation=45)

        for i, count in enumerate(counts):
            plt.text(i, count + 0.5, str(int(count)), ha='center', va='bottom', fontsize=12, color='black')

        plt.tight_layout()
        plt.show()



# Code to test
if __name__ == "__main__":
    # Init Bandit
    TIME_HORIZON = 10_000
    bandit = MultiArmedBandit(np.array([0.23,0.55,0.76,0.44]))
    agent = KLUCBAgent(TIME_HORIZON, bandit) ## Fill with correct constructor

    # Loop
    for i in range(TIME_HORIZON):
        agent.give_pull()

    # Plot curves
    agent.plot_reward_vs_time_curve()
    agent.plot_arm_graph()
    bandit.plot_cumulative_regret()
