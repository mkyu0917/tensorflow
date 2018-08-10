import gym
import numpy  as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr

#https://gist.github.com/stober/194351

def rargmax(vector):
    """Argmax that chooses randomly among eligible maximum indices"""
    m = np.amax(vector)
    indices = np.nonzero(vector ==m)[0]
    return pr.choice(indices)


register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy'
)