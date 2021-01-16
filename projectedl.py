#!pip install keras-rl
#!pip install box2d-py
#!pip install gym[Box_2D]
from __future__ import division
from PIL import Image
import numpy as np
import gym
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam
import keras.backend as K
import keras

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

"""
Carreguem l'enviroment i n'extraiem el núm d'accions i espai d'observació

4 opcions => Esquerra, Dreta, Main i None

8 Space => Pos.x.y, Vel.x.y, Angle, Vel_angular, Contacte amb les potes
"""

ENV_NAME = 'LunarLander-v2'

env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

print(env.action_space.n)
print(env.observation_space.shape)

"""
Declaració del model. 3 capes fully-connected.
"""
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

"""
Reservar memoria i agafar un unic estat.

Política BoltzmannQPolicy => 
  · S'aplica Softmax a les estimacions de les accions
  · L'acció aparentment optima és la més possible de ser escollida
  · Millor respecte greedy perque no considera d'igual manera les opcions considerades no optimes
  · D'aquesta manera s'ignoren accions sub-optimes
  
Testejats diferents Learning rates i agafat el més optim

"""

memory = SequentialMemory(limit=1000000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=0.0015), metrics=['mae'])

"""
Entrenament per 150000 steps
"""
a = dqn.fit(env, nb_steps=150000, visualize=False, verbose=2)

"""
Carregar pesos, cuidado que et carregues l'entrenament
"""

weights_filename = 'dqn64_LunarLander-v2_weights.h5f'.format('LunarLander-v2')

dqn.load_weights(weights_filename)

"""
Test per 20 epochs
"""
dqn.test(env, nb_episodes=20, visualize=False)

import matplotlib.pyplot as plt

plt.plot([199.09,217.98,233.922,225.90,220.99,245.82,236.89,262.95,221.20,241.72], label='Escollit')
plt.plot([235.44,244.76,-94.505,248.86,265.25,228.75,202.80,256.86,239.59,-85.32], label='4 estats 3 capes')
plt.plot([242.257,182.120,-208.038,-433.513,-119.047,206.689,-167.636,173.352,229.998,-181.344], label='4 estats 4 capes')
