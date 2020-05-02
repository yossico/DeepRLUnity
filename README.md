# DeepRLUnity

This is the the solution for Udacity DeepRL Unity project 

The solution is mostly refactoring of a previously learned DQN environment 

The goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas. The minimal requirement for success is to have a windowed average score of at least 13.0 points in 100 consecutive episodes.

The agent runs on Python 3.6 + PyTorch. 



## Environment

State space is  a feature vector of 37 floats which describes the 3D world 

Action space: 4 discrete Actions

Reward: +1 for collecting a yellow banana, and -1 for collecting a purple banana

Episodic based training with 300 steps per episode. 



## Installation 

To set up a python environment to run the code in this repository, please follow the instructions below:

###### Create (and activate) a new environment with Python 3.6.

Linux:
`conda create --name drlnd python=3.6`
`source activate drlnd`

Windows:
`conda create --name drlnd python=3.6` 
`conda activate drlnd`

Install pytorch using conda:
`conda install pytorch=0.4.0 -c pytorch`



###### Clone DeepRL Unity repo:

`git clone: https://github.com/yossico/DeepRLUnity.git` 
`cd DeepRLUnity`
`pip install .`



###### `Create Ipython kernel`

`python -m ipykernel install --user --name drlnd --display-name "drlnd"`



###### Download and install the Unity simulator:


Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)

Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)

Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)

Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)



to run the environment and test the unity simulator change to drlnd environment created:

Linux: source activate drlnd 

windows: activate drlnd



#### Attention: change unity environment path in the second section



## Usage

Run the Navigation.ipynb using:

`Jupiter notebook Navigation.ipynb` 



## Report

Report of the learning algorithm in Report.md