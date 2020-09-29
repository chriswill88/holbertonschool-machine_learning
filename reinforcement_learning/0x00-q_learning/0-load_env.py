#!/usr/bin/env python3
"""this module contains a function for task 0"""
import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """creates the frozen lake enviroment"""
    env = gym.make(
        'FrozenLake-v0', desc=desc, map_name=map_name, is_slippery=is_slippery)
    return env
