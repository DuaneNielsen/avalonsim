from avalonsim import Weapon, Agent, Direction, RangeFinder, CollisionLayer, Env, Action
import random


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
    env = Env(map, state_format="dict")

    import pygame
    from math import floor, ceil
    from copy import deepcopy

    pygame.init()
    screen_width, screen_height = 1200, 400
    screen_border_width, screen_border_height = 50, 50

    screen = pygame.display.set_mode((screen_width, screen_height))
    fps = 50
    speed = 4


    def to_screen(*args):
        """
        convert cartesians to screen co-ordinates
        :param args: an XY or a XYWH in normalized cartesian co-ordinates and length on the interval 0..1
        :return: an XY or YYWH in screen co-ordinates
        """
        x_scale, y_scale = screen_width - screen_border_width * 2, screen_height - screen_border_height * 2
        if len(args) == 2:
            x, y = args[0], args[1]

            x = x_scale * x + screen_border_width
            y = y_scale * (1 - y) + screen_border_height
            return x, y

        if len(args) == 4:
            x, y, w, h = args[0], args[1], args[2], args[3]

            x = x_scale * x + screen_border_width
            y = y_scale * (1 - y - h) + screen_border_height
            w = x_scale * w
            h = y_scale * h
            return x, y, w, h

        assert False, "must be a co-ordinate XY or rectangle XYWH"


    def draw_rect(base, height, width, color):
        y = 0.0
        x, y, width, height = to_screen(base.pos, y, width, height)
        x -= width / 2
        color = pygame.Color(color)
        bar = pygame.Rect(x, y, width, height)
        pygame.draw.rect(screen, color, bar)


    def draw(state):

        screen.fill((0, 0, 0))

        for static in state.statics:
            draw_rect(static, 1., static.width, "grey")

        for agent in state.agents:
            if agent.collision_layer == CollisionLayer.PLAYER:
                color = "blue"
            elif agent.collision_layer == CollisionLayer.ENEMY:
                color = "darkorchid"
            else:
                color = "green"

            draw_rect(agent, 0.6 * agent.hp / agent.hp_max, agent.width, color)

        for shot in state.shots:
            if shot.collision_layer == CollisionLayer.PLAYER_SHOTS:
                draw_rect(shot, 0.4, shot.width, "lightgoldenrod1")
            if shot.collision_layer == CollisionLayer.ENEMY_SHOTS:
                draw_rect(shot, 0.4, shot.width, "red")

        for i, agent in enumerate(state.agents):
            if agent.weapon:
                if agent.weapon.on_cooldown:
                    color = pygame.Color("red")
                else:
                    color = pygame.Color("green")
                x = 0.4 + i * 0.2
                x, y, width, height = to_screen(x, 0.9, 0.05, 0.05)
                bar = pygame.Rect(x, y, width, height)
                pygame.draw.rect(screen, color, bar)

        pygame.display.flip()
        pygame.time.wait(floor(100/speed/fps))


    running = True

    state = env.reset()
    print(state)

    draw(state)
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
                    actions = [Action.PASS, enemy_action]
                elif event.key == pygame.K_t:
                    print([[s[0].name, s[1].name] for s in trajectory])
                else:
                    break

            if event.type == pygame.QUIT:
                running = False

            if len(actions) == 2:
                print([a.name for a in actions])
                trajectory += [actions]
                state, reward, done, info = env.step(actions, render=True)
                print(state, reward, done)

                for dt in range(ceil(info['dt'] * fps)):
                    for key, item in info['initial_state'].items():
                        info['initial_state'][key].pos += info['initial_state'][key].vel / fps
                        draw(info['initial_state'])
                draw(state)

            if done:
                print([[s[0].name, s[1].name] for s in trajectory])
                state = env.reset()
                draw(state)
                done = False
                trajectory = []

    pygame.quit()
