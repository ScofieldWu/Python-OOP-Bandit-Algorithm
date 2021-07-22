from bandit_environments import *
from bandit_agents import *
from bandit_policies import *
from bandit_testrunner import *

# Experiment 1: compare UCB c=1 vs. Epsilon-Greedy 10% 
# set-up the environment (# arms, distributions)
bandit = BanditEnvironment(k = 10, mu = 0, sigma = 1)

# create policy objects
egreedy_10perc = BanditEpsilonGreedyPolicy(epsilon = 0.1)
UCB_1 = BanditUCBPolicy(c = 1)

# create agents using the above policies
# both use sample average method to update action value estimates
agent1 = SampleAverageBanditAgent(policy = UCB_1, bandit_env = bandit)
agent2 = SampleAverageBanditAgent(policy = egreedy_10perc, bandit_env = bandit)

agents = [agent1, agent2]

runner1 = BanditTestRunner(agents = agents, bandit_env = bandit)

# stationary environment
r, o = runner1.perform_runs(timesteps = 1000, runs = 2000)

runner1.visualize_results(
    save_filename = None,
        title = "UCB c=1 vs. Epsilon-Greedy 10%", 
        rewards_histories = r, 
        optimal_action_histories = o
        )



# Experiment 2: compare UCB c=2 vs. Epsilon-Greedy 10% 
# create policy objects
egreedy_10perc = BanditEpsilonGreedyPolicy(epsilon = 0.1)
UCB_2 = BanditUCBPolicy(c = 2)

# create agents using the above policies
# both use sample average method to update action value estimates
agent3 = SampleAverageBanditAgent(policy = UCB_2, bandit_env = bandit)
agent4 = SampleAverageBanditAgent(policy = egreedy_10perc, bandit_env = bandit)

agents = [agent3, agent4]

runner2 = BanditTestRunner(agents = agents, bandit_env = bandit)

# stationary environment
r, o = runner2.perform_runs(timesteps = 1000, runs = 2000)

runner2.visualize_results(
    save_filename = None,
        title = "UCB c=2 vs. Epsilon-Greedy 10%", 
        rewards_histories = r, 
        optimal_action_histories = o
        )



# Experiment 3: compare UCB c=5 vs. Epsilon-Greedy 10% 
# create policy objects
egreedy_10perc = BanditEpsilonGreedyPolicy(epsilon = 0.1)
UCB_5 = BanditUCBPolicy(c = 5)

# create agents using the above policies
# both use sample average method to update action value estimates
agent5 = SampleAverageBanditAgent(policy = UCB_5, bandit_env = bandit)
agent6 = SampleAverageBanditAgent(policy = egreedy_10perc, bandit_env = bandit)

agents = [agent5, agent6]

runner3 = BanditTestRunner(agents = agents, bandit_env = bandit)

# stationary environment
r, o = runner3.perform_runs(timesteps = 1000, runs = 2000)

runner3.visualize_results(
    save_filename = None,
        title = "UCB c=5 vs. Epsilon-Greedy 10%", 
        rewards_histories = r, 
        optimal_action_histories = o
        )