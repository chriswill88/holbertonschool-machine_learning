#!/usr/bin/env python3
"""This module contains the function for task 2"""
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
