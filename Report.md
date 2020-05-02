Report.md

The project is based on the solution of DQN from  https://github.com/udacity/deep-reinforcement-learning/tree/master/dqn/solution



## Source files description



### model.py

The network architecture is the basic implementation of DQN taken from the DQN solution without any change:

![DQN_network.jpg](https://github.com/yossico/DeepRLUnity/blob/master/DQN_network.jpg?raw=true)

## dqn.py

This file contains the dqn class holding the training and testing functions

In addition, this class also holds several environment parameters as a refactoring required for running a Unity ML environment which is designed to work with several brains with code was origannly written for AI.gym which has only "one brain" and different parameters return methods.

## agent.py

This class is based on the agent class of the DQN solution 

It holds the learning and act functions as well as the replay buffer which enables learning from the same action sequence more than once



## The training process

The training and testing process was performed in the Jupyter notebook: `Navigation.ipynb` 

Due to the high variability in results the system was trained to reach an average of 15:

![windowed_learning_score.jpg](https://github.com/yossico/DeepRLUnity/blob/master/windowed_learning_score.jpg?raw=true)

A 10 

## Challenges and further discussion

There are several challenges when trying to estimate the current score of the agent:

- There are many methods that haven't been explored in this solution duo to timing constraints (Im one week late on submission) like DDQN, 

- Basic analysis of the rewards histogram (below) shows distinctly two histograms: 

  - main histogram centered at 18
  - smaller histogram centered on  4.5

  The reason for this as could be seen by the looking at the agent steps is that the agent confronts a wall or a corner

## Future work:

