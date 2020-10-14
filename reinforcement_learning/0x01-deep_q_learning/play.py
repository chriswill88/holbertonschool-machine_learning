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


def create_q_model(actions):
    """This function creates a convolution neural network"""
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=(4, 84, 84))
    layer0 = layers.Permute((2, 3, 1))(inputs)
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(layer0)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)
    layer4 = layers.Flatten()(layer3)
    layer5 = layers.Dense(512, activation="relu")(layer4)
    action = layers.Dense(actions, activation="linear")(layer5)
    return K.Model(inputs=inputs, outputs=action)


model = create_q_model(actions)
memory = SequentialMemory(limit=1000000, window_length=4)

policy = GreedyQPolicy()

stateprocess = AtariProcessor()

dqn = DQNAgent(
    model=model, nb_actions=actions, memory=memory,
    policy=policy, processor=stateprocess)
dqn.compile(optimizer=Adam(lr=.00025, clipnorm=1.0), metrics=['mae'])

dqn.load_weights('policy.h5')

dqn.test(env, nb_episodes=5, visualize=True)
