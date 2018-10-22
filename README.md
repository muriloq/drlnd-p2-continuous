# Project 2 - Continuous Control Using Deep Deterministic Policy Gradients (DDPG) 

## Introduction

The second project in Udacity Deep Reinforcement Learning nanodegree consists on solving a continuous problem named "Reacher" - make the hand of a double-jointed robotic armm follow the goal location - using reinforcement learning. 

Our solution uses a standard Deep Deterministic Policy Gradients (DDPG) implementation as described in the original [research paper](https://arxiv.org/pdf/1509.02971.pdf), to solve the *single-agent* version of the problem. 

The implementation was based on the sample provided as an exercise in the course, modified to use the [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) environment, and use the [PyTorch](https://www.pytorch.org/) framework.

## The Environment

In this project we'll train an agent to control a double-jointed robotic arm, making its hand to follow a specified goal. A reward varying from 0.0 to 0.4 is provided for each step that the agent's hand is in the goal location (please note that this is slightly different from the description provided in the course; this information was obtained by direct observation of the provided precompiled environment). Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible, up to a limit of 1000 time steps, also hard-coded in the precompiled environment, making the maximum possible reward is 40. 

The state space perceived by the agent is a vector with *33 continuous dimensions*, representing the position, rotation, velocity and angular velocities of the arm. After each observation of the state space, the agent may produce an action consisting of a vector with four numbers between -1 and 1, representing the torque to be applied to the two joints. 

This environment is a variant created for the nanodegree and provided as a compiled Unity binary. The animated image below was part of the problem description and illustrates the multi-agent version of the problem (we decided to solve the single-agent version, consisting of a single arm and target). 

![iReacher](reacher.gif)

## Getting Started

All the work was performed on a Windows 10 laptop, with a GeForce GTX 970M GPU. The training was performed using CUDA. 

After cloning the project, download and extract the pre-built "Reacher" environment using the link adequate to your operational system, in the same directory of the project:

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

It is also necessary to install [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md), [unityagents](https://pypi.org/project/unityagents/) and [NumPy](http://www.numpy.org/). Our development used an Anaconda (Python 3.6) environment to install all packages.  

## Training the agent

Run `python ./train.py` to train the agent using DDPG. The average rewards over 100 consecutive episodes will be printed to the standard output. 

At the end, the plot showing the agent progress will be saved in the image `training.png`, and the model (weights learned by the agent) will be saved in files `checkpoint_actor.pth` and `checkpoint_critic.pth` . 

## Running a trained agent

The repository already contains weights trained using DDPG (files `checkpoint_actor.pth` and `checkpoint_critic.pth`).

Run `python ./test.py checkpoint_actor.pth checkpoint_critic.pth` to see the agent in action! 

## Report 

Please refer to file `Report.md` for a detailed description of the solution, including neural network architecture and hyperparameters used. 

