#!/usr/bin/env python3
"""
that can display a game played by the agent trained by train.py:
    Your script should load the policy network saved in policy.h5
    Your agent should use the GreedyQPolicy
"""
import keras as K
import numpy as np
from PIL import Image

import h5py
import gym

from rl.core import Processor
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy

from keras import layers
from keras.optimizers import Adam


# enviroment
env = gym.make('BreakoutNoFrameskip-v4')
state = env.reset()
actions = env.action_space.n


# preprocressor class
class AtariProcessor(Processor):
    """this class prepocesses data"""
    def process_observation(self, observation):
        """This function preprocess the state"""
        obs = observation[:, :, 0]
        prep = np.array(Image.fromarray(obs).resize((84, 84)).convert('L'))
        return prep.astype('uint8')

    def process_state_batch(self, batch):
        """This function preprocesses the state of a batch"""
        return batch.astype('uint8') / 255

    def process_reward(self, reward):
        """This function processes the reward"""
        return np.clip(reward, -1, 1.)


model = K.models.load_model('policy.h5')
memory = SequentialMemory(limit=1000000, window_length=4)
policy = GreedyQPolicy()
stateprocess = AtariProcessor()
dqn = DQNAgent(
    model=model, nb_actions=actions, memory=memory,
    policy=policy, processor=stateprocess)
dqn.compile(optimizer=Adam(lr=.00025, clipnorm=1.0), metrics=['mae'])

dqn.test(env, nb_episodes=1, visualize=True)
