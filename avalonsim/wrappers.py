import gym
from avalonsim.env import Action
import random


class RandomEnemyWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        enemy_action = Action(random.choice(range(4)))
        return self.env.step([action, enemy_action])