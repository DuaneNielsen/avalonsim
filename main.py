from avalonsim import Action
import random
import gym

if __name__ == "__main__":

    env = gym.make('Avalon-v1')

    import pygame

    running = True

    state = env.reset()
    print(state)

    rgb = env.render(mode='human')
    random.seed(42)

    trajectory = []
    done = False

    while running:
        for event in pygame.event.get():

            actions = []

            if event.type == pygame.KEYDOWN:
                enemy_action = Action(random.choice(range(4)))
                if event.key == pygame.K_a:
                    actions = [Action.BACKWARD, enemy_action]
                elif event.key == pygame.K_d:
                    actions = [Action.FORWARD, enemy_action]
                elif event.key == pygame.K_SPACE:
                    actions = [Action.ATTACK, enemy_action]
                elif event.key == pygame.K_s:
                    actions = [Action.NOOP, enemy_action]
                elif event.key == pygame.K_t:
                    print([[s[0].name, s[1].name] for s in trajectory])
                else:
                    break

            if event.type == pygame.QUIT:
                running = False

            if len(actions) == 2:
                print([a.name for a in actions])
                trajectory += [actions]
                state, reward, done, info = env.step(actions)
                print(state, reward, done)

                rgb = env.render(mode='human')

            if done:
                print([[s[0].name, s[1].name] for s in trajectory])
                state = env.reset()
                rgb = env.render(mode='human')
                done = False
                trajectory = []

    pygame.quit()
