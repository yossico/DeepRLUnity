import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6, epsilon = 1, epsilon_decay = 0.9997, alpha = 0.1):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.epsilon_min = 0.04

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        if (np.random.random() < self.epsilon):
            action = int(np.random.random()*self.nA)
        else:
            action = np.argmax(self.Q[state])
        return action #np.random.choice(self.nA)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        if done:
            self.Q[state][action] += self.alpha*(reward-self.Q[state][action])
            self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)
        else:
            bestnextaction = np.argmax(self.Q[next_state])
            self.Q[state][action] += self.alpha*(reward+self.Q[next_state][bestnextaction]-self.Q[state][action]) # 1

