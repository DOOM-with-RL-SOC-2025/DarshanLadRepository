import numpy as np
import matplotlib.pyplot as plt

from epsilon_greedy import EpsilonGreedyAgent
from ucb import UCBAgent
from klucb import KLUCBAgent
from thompson import ThompsonSamplingAgent
from base import MultiArmedBandit


TIME_HORIZON = 30_000
ARM_PROBABILITIES = np.array([0.23, 0.55, 0.76, 0.44])
NUM_AGENTS = 4


np.random.seed(1)


reward_streams = []
for arm_prob in ARM_PROBABILITIES:
    rewards = np.random.binomial(1, arm_prob, TIME_HORIZON)
    reward_streams.append(rewards)
reward_streams = np.array(reward_streams)


def run_agent(agent_class, name):
    bandit = MultiArmedBandit(ARM_PROBABILITIES.copy())
    agent = agent_class(TIME_HORIZON, bandit)
    for t in range(TIME_HORIZON):
        
        def patched_pull(arm):
            reward = reward_streams[arm][t]
            bandit.cumulative_regret_array.append(bandit.cumulative_regret_array[-1] + bandit.best_arm - reward)
            return reward
        bandit.pull = patched_pull
        agent.give_pull()
    return agent, bandit


agents = {}
agents["Epsilon-Greedy"] = run_agent(EpsilonGreedyAgent, "Epsilon-Greedy")
agents["UCB"] = run_agent(UCBAgent, "UCB")
agents["KL-UCB"] = run_agent(KLUCBAgent, "KL-UCB")
agents["Thompson"] = run_agent(ThompsonSamplingAgent, "Thompson")


plt.figure(figsize=(10, 5))
for name, (agent, _) in agents.items():
    rewards = np.array(agent.rewards)
    avg_rewards = np.cumsum(rewards) / np.arange(1, TIME_HORIZON + 1)
    plt.plot(avg_rewards, label=name)
plt.title("Average Reward vs Time")
plt.xlabel("Timestep")
plt.ylabel("Average Reward")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 5))
for name, (_, bandit) in agents.items():
    plt.plot(bandit.cumulative_regret_array[1:], label=name)
plt.title("Cumulative Regret vs Time")
plt.xlabel("Timestep")
plt.ylabel("Regret")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


optimal_arm = np.argmax(ARM_PROBABILITIES)
optimal_counts = []
for name, (agent, _) in agents.items():
    count = sum([1 for i in range(len(agent.rewards)) if agent.count_memory[optimal_arm]])
    optimal_counts.append(agent.count_memory[optimal_arm])

plt.figure(figsize=(8, 5))
bars = plt.bar(agents.keys(), optimal_counts, color='skyblue', edgecolor='black')
plt.title("Number of times Optimal Arm was Pulled")
plt.ylabel("Count")
plt.grid(axis='y')


for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 100, str(int(height)), ha='center', va='bottom')

plt.tight_layout()
plt.show()
