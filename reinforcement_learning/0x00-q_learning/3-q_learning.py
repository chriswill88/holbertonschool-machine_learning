#!/usr/bin/env python3
"""This module contains the function for task 3"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """this function uses epsilon-greedy to determine the next action"""
    if np.random.uniform(0, 1) < epsilon:
        # Explore: random action
        choice = np.random.randint(0, Q.shape[1])
    else:
        # Exploit: select action with max value (Future Reward)
        choice = np.argmax(Q[state])
    return choice


def train(
    env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1,
        min_epsilon=0.1, epsilon_decay=0.05):
    """This function trains Q-learning algorithm"""
    total_rewards = []
    maxeps = epsilon
    for i_episode in range(episodes):
        curr_rew = 0
        state = env.reset()
        for t in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, done, info = env.step(action)
            if reward == 0 and done is True:
                reward = -1
            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[new_state, :]) - Q[state, action])

            state = new_state
            curr_rew += reward
            if done:
                total_rewards.append(curr_rew)
                break
        epsilon = min_epsilon + \
            (maxeps - min_epsilon) * np.exp(-epsilon_decay*i_episode)
    return Q, total_rewards
