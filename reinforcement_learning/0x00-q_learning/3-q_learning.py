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
    epi = epsilon
    total_rewards = []
    for i_episode in range(episodes):
        state = env.reset()

        for t in range(max_steps):
            # env.render()
            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, done, info = env.step(action)
            if reward == 0 and done is True:
                reward = -1
            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[new_state, :]) - Q[state, action])
            state = new_state
            if done:
                total_rewards.append(reward)
                # print("Episode finished after {} timesteps".format(t+1))k
                if (epsilon - epsilon_decay) > min_epsilon:
                    epsilon -= epsilon_decay
                else:
                    epsilon = min_epsilon
                break
    return Q, total_rewards
