import gymnasium as gym
from gymnasium.spaces import Discrete
from avalonsim.env import Action
import random


class RandomEnemyWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        enemy_action = Action(random.choice(range(4)))
        return self.env.step([action, enemy_action])


class InvertRewardWrapper(gym.RewardWrapper):
    def reward(self, reward):
        return - reward


class NoTurnaroundWrapper(gym.ActionWrapper):

    def reverse_action(self, action):
        pass

    def __init__(self, env):
        super().__init__(env)
        self.action_space = Discrete(env.unwrapped.action_space.n - 1)
        self.mapping = []

    def action(self, action):
        for i in range(len(action)):
            if action[i] >= Action.REVERSE_FACING:
                action[i] += 1
        return action
