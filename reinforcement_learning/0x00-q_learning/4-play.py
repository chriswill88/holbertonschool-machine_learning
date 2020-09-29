#!/usr/bin/env python3
"""This module contains the function for task 4"""
import numpy as np


def play(env, Q, max_steps=100):
    """play - plays an episode"""
    tot_reward = []
    state = env.reset()
    env.render()
    for step in range(max_steps):
        action = np.argmax(Q[state])
        state, reward, done, info = env.step(action)
        if reward == 0 and done is True:
            reward = -1
        env.render()
        if done:
            return reward
