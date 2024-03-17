from avalonsim import Action
import random
import gym
from argparse import ArgumentParser

if __name__ == "__main__":

    parser = ArgumentParser()
    args = parser.parse_args()

    env = gym.make('Avalon-v1')

    import pygame

    running = True

    state = env.reset()
    print(state)

    rgb = env.render(mode='human')
    random.seed(42)

    player_wins = 0
    trials = 0
    player_health = 0
    done = False
    actions = []

    while running:
        actions = [Action.ATTACK, Action.ATTACK]

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

        if len(actions) == 2:
            state, reward, done, info = env.step(actions)
            rgb = env.render(mode='human')

            if reward >= 1.:
                player_wins += 1
                trials += 1
                player_health += env.state.agents[0].hp

        if done:
            state = env.reset()
            rgb = env.render(mode='human')
            done = False
            trajectory = []
            print(player_wins, trials, player_health/trials)

    pygame.quit()
