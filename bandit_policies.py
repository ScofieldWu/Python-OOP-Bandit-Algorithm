import numpy as np
import random

"""
Recall that policies are a class of things that choose actions. They represent
the computation that the agent carries out that outputs what to do.

As a result, policies are a great candidate for the Strategy design pattern.
Every policy follows the same general structure, and so we have them inherit
their structure from a single, higher-level Policy object.

Each new class is a variant on the Policy base framework. Just like how each new
iPhone is kind of different... but not really lol

Amazing for faster experimentation. Off we go!
"""

class BanditPolicy():
    """
    This is the base/parent policy class, not meant to be used directly.

    We place this at the top of the file so that anyone reading this code in 
    future - including future you - will know that to use any policy class, they
    mainly just need to read this one. Read one thing, and learn most of how
    all the others work. So convenient!!
    """

    # We don't need an init, since this is about the shared methods of choice.

    def argmax_with_random_tiebreaker(self, action_value_estimates):
        """
        Chooses the maximum of the provided action-value estimates,
        with ties broken randomly.

        Args:
            action_value_estimates: A numpy array containing action-value
            estimates.
        Returns:
            The index of the max element.
        """
        return np.random.choice(
            np.where( action_value_estimates == action_value_estimates.max())[0]
        )
    
    def choose_action(self):
        """
        This method is just here to be overridden, so it is 'empty'.
        """
        pass

class BanditEpsilonGreedyPolicy(BanditPolicy):
    """
    The epsilon-greedy action selection policy. An agent following the 
    epsilon-greedy policy will choose a random action with probability epsilon,
    and will greedily choose the best action (argmax) with probability 
    1 - epsilon. If multiple actions are tied for the best choice, ties are
    broken randomly.
    
    Attributes:
        epsilon: A value [0,1] that determines the probability that an agent will
        randomly choose an action at each timestep.
    """
    def __init__(self, epsilon = 0.1):
        self.epsilon = epsilon
    
    def __str__(self):
        return "Epsilon-Greedy Policy: {e}".format(e = self.epsilon)
    
    def choose_action(self, agent_data):
        """
        This is where we override the method that's in the base class.

        Args:
            agent_data: a dictionary contains action_value_estimates and action_counts
            action_value_estimates: A numpy array containing the action-value
            estimates for a given bandit problem environment.
        Returns:
            action: The index of the chosen action.
        """
        action_value_estimates = agent_data['action_value_estimates']
        roll = random.uniform(0,1)
        if roll <= self.epsilon:
            action = random.choice( list( range(0,len(action_value_estimates))))
        else:
            action = self.argmax_with_random_tiebreaker(action_value_estimates)
        return action
    

class BanditUCBPolicy(BanditPolicy):
    """
    Upper-Confidence-Bound Action Selection Policy

    YOUR OWN EXPLANATION FOR HOW THIS POLICY WORKS GOES HERE.
    THIS WILL BE GRADED.

    Attributes:
        c: A number > 0 that controls the degree of exploration.
    """
    # CODE GOES HERE

    def __init__(self, c = 2):
        self.c = c
    
    def __str__(self):
        return "UCB Policy: {c}".format(c = self.c)
    
    def choose_action(self, agent_data):
        """
        This is where we override the method that's in the base class.

        Args:
            agent_data: a dictionary contains action_value_estimates and action_counts
            
            action_value_estimates: A numpy array containing the action-value
            estimates for a given bandit problem environment.
            
            action_counts: A numpy array containing the number of time each
            action has been selected for a given bandit problem environment.
        Returns:
            action: The index of the chosen action.
        """
        # pull action counts and value from agent class
        action_counts = agent_data['action_counts']
        action_value_estimates = agent_data['action_value_estimates']
        
        # total timestep
        t = np.sum(action_counts, axis = 0)

        # at the begining, randomly choose an action
        if t == 0:
            action = action = random.choice( list( range(0,len(action_value_estimates))))
        
        # otherwise, implement UCB policy - equation 2.10
        else:
            uncertainty = np.sqrt(np.log(t) / (action_counts + 0.000001))
            action = self.argmax_with_random_tiebreaker(action_value_estimates \
                 + self.c * uncertainty)
        return action


class BanditSoftmaxPolicy(BanditPolicy):
    """
    As in equation 2.9 in the text.
    """

    def __str__(self):
        return "Softmax Policy"

    def softmax(self, x):
        probabilities = np.exp(x) / np.sum(np.exp(x), axis = 0)
        return probabilities

    def choose_action(self, agent_data):
        action_value_estimates = agent_data['action_value_estimates']
        probabilities = self.softmax(action_value_estimates)
        action_choices = range(0,len(action_value_estimates))
        action = random.choices(action_choices, probabilities)[0]
        return action

