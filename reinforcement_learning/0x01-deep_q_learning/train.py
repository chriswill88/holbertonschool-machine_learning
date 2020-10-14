#!/usr/bin/env python3
"""
This model utilizes keras, keras-rl, and gym to train an agent that can
play Atari’s Breakout
"""
from PIL import Image
import keras as K
import numpy as np
import gym

from keras import layers
from keras.optimizers import Adam

from rl.core import Processor
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory
from rl.agents.dqn import DQNAgent


# enviroment
env = gym.make('BreakoutNoFrameskip-v4')
state = env.reset()
actions = env.action_space.n


class AtariProcessor(Processor):
    def process_observation(self, observation):
        obs = observation[:, :, 0]
        prep = np.array(Image.fromarray(obs).resize((84, 84)).convert('L'))
        return prep.astype('uint8')

    def process_state_batch(self, batch):
        return batch.astype('uint8') / 255

    def process_reward(self, reward):
        return np.clip(reward, -1, 1.)

# Downgrade to tensorflow 1.14 before attempting to run


def create_q_model(actions):
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=(4, 84, 84))
    layer0 = layers.Permute((2, 3, 1))(inputs)
    # Convolutions on the frames on the screen
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(layer0)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)
    layer4 = layers.Flatten()(layer3)
    layer5 = layers.Dense(512, activation="relu")(layer4)
    action = layers.Dense(actions, activation="linear")(layer5)
    return K.Model(inputs=inputs, outputs=action)


model = create_q_model(actions)
model.summary()

memory = SequentialMemory(limit=1000000, window_length=4)
policy = LinearAnnealedPolicy(
    EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1,
    value_test=.05, nb_steps=1000000)

stateprocess = AtariProcessor()

dqn = DQNAgent(
    model=model, nb_actions=actions, memory=memory,
    nb_steps_warmup=35, target_model_update=1e-2, policy=policy,
    processor=stateprocess, enable_double_dqn=True)

dqn.compile(optimizer=Adam(lr=.00025, clipnorm=1.0), metrics=['mae'])
dqn.fit(env, nb_steps=1750000)
dqn.save_weights('policy.h5', overwrite=True)
model.save("my_model.h5")
# dqn.test(env, nb_episodes=5, visualize=True)
