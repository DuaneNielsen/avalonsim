from avalonsim import Weapon, Agent, Direction, RangeFinder, CollisionLayer, Env, Action
import random
import cv2

if __name__ == "__main__":

    sword = Weapon(damage=10, shot_speed=0.5, time_to_live=0.04, cooldown_time=0.1, shot_width=0.01, windup_time=0.1, recovery_time=0.04)
    bow = Weapon(damage=3, shot_speed=0.5, time_to_live=1., cooldown_time=0.3, windup_time=0.3)
    player = Agent(pos=0.1, facing=Direction.EAST, collision_layer=CollisionLayer.PLAYER, shot_collision_layer=CollisionLayer.PLAYER_SHOTS)
    player.weapon = sword
    player.add_vertex(RangeFinder(player.weapon.range))
    player.add_vertex(RangeFinder(-player.weapon.range))
    enemy = Agent(pos=0.9, facing=Direction.WEST)
    enemy.weapon = bow
    enemy.add_vertex(RangeFinder(enemy.weapon.range))
    enemy.add_vertex(RangeFinder(-enemy.weapon.range))
    map = [player, enemy]
    env = Env(map)

    import pygame

    running = True

    state = env.reset()
    print(state)

    rgb = env.render(mode='rgb')
    cv2.imshow("screen", rgb)
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

                rgb = env.render(mode='rgb')
                cv2.imshow("screen", rgb)

            if done:
                print([[s[0].name, s[1].name] for s in trajectory])
                state = env.reset()
                rgb = env.render(mode='rgb')
                cv2.imshow("screen", rgb)
                done = False
                trajectory = []

    pygame.quit()
