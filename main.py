from avalonsim import Action
import random
import gymnasium as gym
from argparse import ArgumentParser


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--enemy', action='store_true')
    args = parser.parse_args()

    env = gym.make('Avalon-v1', render_mode='human')

    import pygame

    running = True

    state, info = env.reset()
    print(state)

    rgb = env.render()
    random.seed(42)

    trajectory = []
    done = False

    while running:
        for event in pygame.event.get():

            actions = []

            if event.type == pygame.KEYDOWN:
                random_action = Action(random.choice(range(4)))
                if event.key == pygame.K_a:
                    actions = [Action.BACKWARD, random_action]
                elif event.key == pygame.K_d:
                    if event.mod & pygame.KMOD_SHIFT:
                        actions = [Action.SPRINT_FORWARD, random_action]
                    elif event.mod & pygame.KMOD_CTRL:
                        actions = [Action.DODGEROLL_FORWARD, random_action]
                    else:
                        actions = [Action.FORWARD, random_action]
                elif event.key == pygame.K_SPACE:
                    actions = [Action.ATTACK, random_action]
                elif event.key == pygame.K_s:
                    actions = [Action.NOOP, random_action]
                elif event.key == pygame.K_t:
                    actions = [Action.HEALTH_POT, random_action]
                elif event.key == pygame.K_q:
                    actions = [Action.REVERSE_FACING, random_action]

                elif event.key == pygame.K_t:
                    print([[s[0].name, s[1].name] for s in trajectory])
                else:
                    break

                if args.enemy:
                    actions = list(reversed(actions))

            if event.type == pygame.QUIT:
                running = False

            if len(actions) == 2:
                print([a.name for a in actions])
                trajectory += [actions]
                state, reward, done, trunc, info = env.step(actions)
                print(state, reward, done)

                rgb = env.render()

            if done:
                print([[s[0].name, s[1].name] for s in trajectory])
                state, info = env.reset()
                rgb = env.render()
                done = False
                trajectory = []

    pygame.quit()
