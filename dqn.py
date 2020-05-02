from unityagents import UnityEnvironment
import numpy as np
from Agent import Agent
from collections import deque
import torch

class dqn:

    def __init__(self, fullfilename, ind_brain=0 ):
        self.env = UnityEnvironment(file_name= fullfilename)
        self.ind_brain = ind_brain
        self.brain_name = self.env.brain_names[ind_brain]
        self.brain = self.env.brains[self.brain_name]
        self.actionsize = self.brain.vector_action_space_size
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        state = env_info.vector_observations[ind_brain]
        self.statesize = len(state)



    def train(self, agent, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, thresh = 15):
        """Deep Q-Learning.

        Params
        ======
            n_episodes (int): maximum number of training episodes
            max_t (int): maximum number of timesteps per episode
            eps_start (float): starting value of epsilon, for epsilon-greedy action selection
            eps_end (float): minimum value of epsilon
            eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        """
        scores = []  # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores
        eps = eps_start  # initialize epsilon
        for i_episode in range(1, n_episodes + 1):
            env_info = self.env.reset(train_mode=True)[self.brain_name]
            state = env_info.vector_observations[self.ind_brain]
            score = 0
            for t in range(max_t):
                action = agent.act(state, eps)
                env_info = self. env.step(int(action))[self.brain_name]
                next_state = env_info.vector_observations[0]
                reward = env_info.rewards[self.ind_brain]
                done = env_info.local_done[self.ind_brain]
                agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break
            scores_window.append(score)  # save most recent score
            scores.append(score)  # save most recent score
            eps = max(eps_end, eps_decay * eps)  # decrease epsilon
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            if np.mean(scores_window) >= thresh:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100, np.mean(scores_window)))
                torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
                break
        return scores

    def test(self, agent, n_episodes = 10):
        scores = []
        for i_episode in range(1, n_episodes + 1):
            env_info = self.env.reset(train_mode=True)[self.brain_name]
            state = env_info.vector_observations[0]
            score = 0
            for t in range(300):
                action = agent.act(state, 0)
                env_info = self.env.step(int(action))[self.brain_name]
                state = env_info.vector_observations[self.ind_brain]
                reward = env_info.rewards[self.ind_brain]
                done = env_info.local_done[self.ind_brain]
                score += reward
                if done:
                    break
            scores.append(score)  # save most recent score
        avg_score = np.mean(scores)
        print('\nEnvironment average score is: {:.2f}'.format(avg_score))
        return scores
