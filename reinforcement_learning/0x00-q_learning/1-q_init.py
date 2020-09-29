#!/usr/bin/env python3
"""this model contains the function for task 1"""
import numpy as np


def q_init(env):
    """q_init - intitializes q table"""
    return np.zeros((env.observation_space.n, env.action_space.n))
