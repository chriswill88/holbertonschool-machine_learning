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

from rl.callbacks import Callback, ModelIntervalCheckpoint
from rl.core import Processor
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory
from rl.agents.dqn import DQNAgent

from os import path

# enviroment setup
env = gym.make('BreakoutNoFrameskip-v4')
state = env.reset()
actions = env.action_space.n


# Custom callback function
class ModelIntervalCheck(Callback):
    """
    ModelIntervalCheck: is a callback class
    """
    def __init__(self, filepath, interval, verbose=0, kmodel=None):
        """
        This callback will allow the model to be save every x steps
        @filepath: the filepath
        @intervals: the intervals
        @verbose: wheather the message prints or not
        @kmodel: the keras model used
        """
        self.filepath = filepath
        self.interval = interval
        self.verbose = verbose
        self.kmodel = kmodel
        self.total_steps = 0

    def on_step_end(self, step, logs={}):
        """ Save weights at interval steps during training """
        self.total_steps += 1
        if self.total_steps % self.interval != 0:
            # Nothing to do.
            return

        filepath = self.filepath.format(step=self.total_steps, **logs)
        if self.verbose > 0:
            print('\nStep {}: saving kmodel to {}'.format(
                self.total_steps, filepath))
        self.kmodel.save(filepath)


# Preprocesser class
class AtariProcessor(Processor):
    """prepocesses data"""
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


# model used
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


# This will automatically use a saved model!
if path.exists("policy.h5"):
    print("Using saved model!")
    model = K.models.load_model('policy.h5')
else:
    print("Using new model!")
    model = create_q_model(actions)

# setting up the DQN agent and keras-rl stuff
memory = SequentialMemory(limit=1000000, window_length=4)
policy = LinearAnnealedPolicy(
    EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1,
    value_test=.05, nb_steps=850000)
stateprocess = AtariProcessor()
dqn = DQNAgent(
    model=model, nb_actions=actions, memory=memory,
    nb_steps_warmup=35, target_model_update=1e-2, policy=policy,
    processor=stateprocess, enable_double_dqn=True)
dqn.compile(
    optimizer=Adam(lr=.00025, clipnorm=1.0),
    metrics=['mae', 'accuracy'])
dqn.fit(env, nb_steps=1750000, callbacks=[
    ModelIntervalCheck('policy.h5', 1000, 1, model)], visualize=True)

# Saving the policy network
model.save("policy.h5")
