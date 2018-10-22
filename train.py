""" A script for training a DDPG agent in the single-agent Reacher environment of Udacity Deep Reinforcement Learning nanodegree (Project 2).
Learning statistics will be printed in the standard output; a plot of the progress will be saved in file training.png.
After completion, neural network weights will be saved in files checkpoint_actor.pth checkpoint_critic.pth. 
This code is heavily based on the proposed exercise, with small changes for compatibility with the Unity environment, and use in the command-line.
"""

from unityagents import UnityEnvironment
from ddpg_agent import Agent
import numpy as np
import sys 
import torch
from collections import deque
import matplotlib.pyplot as plt
import time 

plt.ion()

env = UnityEnvironment(file_name='Reacher_Windows_x86_64_single/Reacher.exe', no_graphics=True)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of actions and state size
action_size = brain.vector_action_space_size
state = env_info.vector_observations[0]
state_size = len(state)

# Instantate the agent, using DQN or Double DQN according to the command line argument
agent = Agent(state_size=state_size, action_size=action_size, random_seed=64)

def ddpg(n_episodes=2000, max_t=1000):
    """DDPG
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]   # initial state

        score = 0
        timestep = time.time()
        agent.reset()
        for t in range(max_t):
            action = agent.act(state) 
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        print('\rEpisode {} Score: {:.2f}, Average Score: {:.2f}, Max: {:.2f}, Min: {:.2f}, Time: {:.2f}'.format(i_episode, score, np.mean(scores_window), np.max(scores), np.min(scores), time.time() - timestep), end="\n")   

        if np.mean(scores_window)>=30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            break
    return scores

scores = ddpg()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.show()
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig('training.png')

env.close()
