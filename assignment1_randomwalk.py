from bandit_environments import *
from bandit_agents import *
from bandit_policies import *
from bandit_testrunner import *

# Experiment 1: Non-Stationary: Sample Average vs Recency-Weighted Average
# set-up the environment (# arms, distributions)
bandit = BanditEnvironment(k = 10, mu = 0, sigma = 1)

# Policy: Epsilon-Greedy 10%
egreedy_10perc = BanditEpsilonGreedyPolicy(epsilon = 0.1)

# Agent with different action-value method: Sample Average vs Recency Weighted Average
agent1 = SampleAverageBanditAgent(policy = egreedy_10perc, bandit_env = bandit)
agent2 = RecencyWeightedAverageBanditAgent(policy = egreedy_10perc, bandit_env = bandit)

agents = [agent1, agent2]

runner1 = BanditTestRunner(agents = agents, bandit_env = bandit)

# non-stationary environment
r, o = runner1.perform_runs(timesteps = 1000, runs = 2000, stationarity = False)

runner1.visualize_results(
    save_filename = None,
        title = "Non-Stationary: Sample Average vs Recency-Weighted Average", 
        rewards_histories = r, 
        optimal_action_histories = o
        )


# Experiment 2: Non-Stationary: Epsilon-Greedy 10% vs 1% using Sample Average
# Policy: Epsilon-Greedy 10% vs 1%
egreedy_10perc = BanditEpsilonGreedyPolicy(epsilon = 0.1)
egreedy_1perc = BanditEpsilonGreedyPolicy(epsilon = 0.01)

# Agent: Sample Average
agent3 = SampleAverageBanditAgent(policy = egreedy_10perc, bandit_env = bandit)
agent4 = SampleAverageBanditAgent(policy = egreedy_1perc, bandit_env = bandit)

agents = [agent3, agent4]

runner2 = BanditTestRunner(agents = agents, bandit_env = bandit)

# non-stationary environment
r, o = runner2.perform_runs(timesteps = 1000, runs = 2000, stationarity = False)

runner2.visualize_results(
    save_filename = None,
        title = "Non-Stationary: Epsilon-Greedy 10% vs 1%", 
        rewards_histories = r, 
        optimal_action_histories = o
        )


# Experiment 3: Non-Stationary: Sample Average(10%) vs Recency-Weighted Average(1%)
# Policy: Epsilon-Greedy 10% vs 1%
egreedy_10perc = BanditEpsilonGreedyPolicy(epsilon = 0.1)
egreedy_1perc = BanditEpsilonGreedyPolicy(epsilon = 0.01)

# Agent: Sample Average vs Recency Weighted Average
agent5 = SampleAverageBanditAgent(policy = egreedy_10perc, bandit_env = bandit)
agent6 = RecencyWeightedAverageBanditAgent(policy = egreedy_1perc, bandit_env = bandit)

agents = [agent5, agent6]

runner3 = BanditTestRunner(agents = agents, bandit_env = bandit)

# non-stationary environment
r, o = runner3.perform_runs(timesteps = 1000, runs = 2000, stationarity = False)

runner3.visualize_results(
    save_filename = None,
        title = "Non-Stationary: Sample Average(10%) vs Recency-Weighted Average(1%)", 
        rewards_histories = r, 
        optimal_action_histories = o
        )


