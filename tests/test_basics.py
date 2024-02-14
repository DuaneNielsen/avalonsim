from main import Wall, Agent, Weapon
from main import Action, Direction
from main import CL_PLAYER, CL_ENEMY, CL_PLAYER_SHOTS, CL_ENEMY_SHOTS
from main import Env
from main import TimerQueue, Timer
from main import close
import random
from main import collision_handler

def test_base():
    agent1 = Agent(pos=0.1)
    agent2 = Agent(pos=0.2)
    width = agent1.width
    agent1.vel = 0.1
    agent2.vel = -0.1
    assert agent1.moving_in_direction_of(agent2)
    assert agent2.moving_in_direction_of(agent1)
    assert not agent1.moving_same_direction(agent2)
    assert not agent2.moving_same_direction(agent1)

    agent1 = Agent(pos=0.1)
    agent2 = Agent(pos=0.2)
    agent1.vel = 0.1
    agent2.vel = 0.1

    assert agent1.moving_in_direction_of(agent2)
    assert not agent2.moving_in_direction_of(agent1)
    assert agent1.moving_same_direction(agent2)
    assert agent2.moving_same_direction(agent1)

    agent1 = Agent(pos=0.1 - width)
    agent2 = Agent(pos=0.1 + width)
    agent1.vel = 0.1
    agent2.vel = 0.0

    assert agent1.moving_in_direction_of(agent2)
    assert not agent2.moving_in_direction_of(agent1)
    assert not agent1.moving_same_direction(agent2)
    assert not agent2.moving_same_direction(agent1)


def test_walls_and_simple_movement():
    print()

    map = [Agent(pos=0.5, facing=Direction.EAST, collision_layer=CL_PLAYER)]
    env = Env(map)
    state = env.reset()
    assert state.agents[0].pos == 0.5

    action = Action.FORWARD
    print(action)
    state, reward, done, info = env.step([action])
    print(state)
    assert close(state.agents[0].pos, 1.0 - state.agents[0].width / 2 - state.statics[1].width / 2)
    assert state.agents[0].vel == 0.

    action = Action.FORWARD
    print(action)
    state, reward, done, info = env.step([action])
    assert state.agents[0].vel == 0.
    assert close(state.agents[0].pos, 1.0 - state.agents[0].width / 2 - state.statics[1].width / 2)

    action = Action.BACKWARD
    print(action)
    state, reward, done, info = env.step([action])
    print(state)
    assert close(state.agents[0].pos, state.agents[0].width / 2 + state.statics[1].width / 2)
    assert state.agents[0].vel == 0.0

    action = Action.REVERSE_FACING
    print(action)
    state, reward, done, info = env.step([action])
    print(state)
    assert close(state.agents[0].pos, state.agents[0].width / 2 + state.statics[1].width / 2)
    assert state.agents[0].vel == 0.0

    action = Action.FORWARD
    print(action)
    state, reward, done, info = env.step([action])
    print(state)
    assert close(state.agents[0].pos, state.agents[0].width / 2 + state.statics[1].width / 2)
    assert state.agents[0].vel == 0.0

    action = Action.BACKWARD
    print(action)
    state, reward, done, info = env.step([action])
    print(state)
    assert state.agents[0].vel == 0.0
    assert close(state.agents[0].pos, 1.0 - state.agents[0].width / 2 - state.statics[1].width / 2)

    action = Action.BACKWARD
    print(action)
    state, reward, done, info = env.step([action])
    print(state)
    assert close(state.agents[0].pos, 1.0 - state.agents[0].width / 2 - state.statics[1].width / 2)
    assert state.agents[0].vel == 0.0


def test_agent_agent_collision_simple():
    # two agents meet halfway
    agent1 = Agent(pos=0.1, facing=Direction.EAST, collision_layer=CL_PLAYER)
    agent2 = Agent(pos=0.9, facing=Direction.WEST, collision_layer=CL_ENEMY)
    map = [agent1, agent2]
    env = Env(map)
    state = env.reset()

    actions = [Action.FORWARD, Action.FORWARD]
    print(actions)
    state, reward, done, info = env.step(actions)
    print(state)
    assert close(state.agents[0].pos, (0.5 - state.agents[0].width / 2))
    assert state.agents[0].vel == 0.
    assert close(state.agents[1].pos, (0.5 + state.agents[0].width / 2))
    assert state.agents[1].vel == 0.

    actions = [Action.BACKWARD, Action.BACKWARD]
    print(actions)
    state, reward, done, info = env.step(actions)
    print(state)
    assert close(state.agents[0].pos, 0.0 + state.statics[0].width / 2 + state.agents[0].width / 2)
    assert state.agents[0].vel == 0.0
    assert close(state.agents[1].pos, 1.0 - state.statics[1].width / 2 - state.agents[1].width / 2)
    assert state.agents[1].vel == 0.0


def test_agent_agent_collision_chase():
    # two agents meet halfway
    agent1 = Agent(pos=0.1, facing=Direction.EAST, collision_layer=CL_PLAYER)
    agent2 = Agent(pos=0.15, walk_speed=0.05, facing=Direction.EAST, collision_layer=CL_ENEMY)
    agent2.pos += agent2.width / 2
    map = [agent1, agent2]
    env = Env(map)
    state = env.reset()
    state, reward, done, info = env.step([Action.FORWARD, Action.FORWARD])
    print(state)
    assert close(state.agents[0].pos, 0.2 - agent1.width)
    assert close(state.agents[1].pos, 0.2)
    assert state.agents[0].vel == 0.05
    assert state.agents[1].vel == 0.05


def test_agent_agent_collision_wait():
    agent1 = Agent(pos=0.1, facing=Direction.EAST, collision_layer=CL_PLAYER)
    agent2 = Agent(pos=0.5, walk_speed=0.05, facing=Direction.WEST, collision_layer=CL_ENEMY)
    map = [agent1, agent2]
    env = Env(map)
    state = env.reset()
    state, reward, done, info = env.step([Action.PASS, Action.FORWARD])

    print(state)
    assert state.agents[0].pos == 0.1
    assert state.agents[0].vel == 0.

    # if you were not moving on the collision you stay put
    assert close(state.agents[1].pos, 0.1 + state.agents[1].width)
    assert state.agents[1].vel == 0.


def test_shoot_wall():
    print()
    player = Agent(pos=0.5, facing=Direction.EAST, collision_layer=CL_PLAYER, shot_collision_layer=CL_PLAYER_SHOTS)
    sword = Weapon(damage=10, shot_speed=100, time_to_live=0.01, action_blocking=True)
    player.weapon = sword

    map = [player]
    env = Env(map)
    state = env.reset()
    assert state.agents[0].pos == 0.5
    assert len(state.dynamics) == 1
    state, reward, done, info = env.step([Action.ATTACK])

    # the shot should hit the wall and vanish
    assert len(state.dynamics) == 1


def test_sword():
    print()
    player = Agent(pos=0.4, facing=Direction.EAST, collision_layer=CL_PLAYER, shot_collision_layer=CL_PLAYER_SHOTS)
    sword = Weapon(damage=10, shot_speed=100, time_to_live=0.01, action_blocking=True)
    player.weapon = sword
    enemy = Agent(pos=0.6, facing=Direction.WEST, collision_layer=CL_ENEMY, shot_collision_layer=CL_ENEMY_SHOTS)
    sword = Weapon(damage=10, shot_speed=100, time_to_live=0.01, action_blocking=True)
    enemy.weapon = sword
    map = [player, enemy]
    env = Env(map)
    state = env.reset()
    assert state.agents[0].pos == 0.4
    assert state.agents[1].pos == 0.6
    assert len(state.dynamics) == 2
    state, reward, done, info = env.step([Action.ATTACK, Action.ATTACK])
    print(state)

    # the shot should hit the agents
    assert len(state.dynamics) == 2
    assert len(state.shots) == 0
    assert state.agents[1].hp == 90
    assert state.agents[0].hp == 90
    print(state)


def test_timer():
    tq = TimerQueue()
    timer1, timer2, timer3 = Timer(), Timer(), Timer()
    timer1.t = 1.
    timer2.t = 2.
    timer3.t = 3.

    assert tq.is_empty()

    tq.push(timer3)
    assert tq.peek().t == 3.
    assert not tq.is_empty()

    tq.push(timer1)
    assert tq.peek().t == 1.

    tq.push(timer2)
    assert tq.peek().t == 1.

    assert tq.pop().t == 1.
    assert tq.peek().t == 2.

    assert tq.pop().t == 2.
    assert tq.peek().t == 3.

    assert tq.pop().t == 3.
    assert tq.peek() is None
    assert tq.is_empty()


def test_wall_hack():
    sword = Weapon(damage=10, shot_speed=100, time_to_live=0.01, action_blocking=True)
    player = Agent(pos=0.1, facing=Direction.EAST, collision_layer=CL_PLAYER, shot_collision_layer=CL_PLAYER_SHOTS)
    player.weapon = sword
    enemy = Agent(pos=0.9)
    enemy.weapon = sword
    map = [player, enemy]
    env = Env(map)
    state = env.reset()

    actions = [['FORWARD', 'PASS'], ['FORWARD', 'PASS']]

    for agent1_action, agent2_action in actions[:-1]:
        action = [Action[agent1_action], Action[agent2_action]]
        print(action)
        state, _, _, _ = env.step(action)
        print(state)

        for i, item in enumerate(state.dynamics):
            # no wall hacking
            assert state.get_sorted_collision_map()[1].pos <= item.pos <= state.get_sorted_collision_map()[-2].pos

    agent1_action, agent2_action = actions[-1]
    action = [Action[agent1_action], Action[agent2_action]]
    print(action)
    state, _, _, _ = env.step(action)
    print(state)

    for i, item in enumerate(state.dynamics):
        # no wall hacking
        cmap = state.get_sorted_collision_map()
        assert cmap[1].pos <= item.faces[0].pos <= cmap[-2].pos
        assert cmap[1].pos <= item.faces[1].pos <= cmap[-2].pos

    for i, item in enumerate(state.dynamics):
        for j, next_item in enumerate(state.dynamics):
            if i != j:
                assert item.pos != next_item.pos

def test_shot_penetrate_rules():
    sword = Weapon(damage=10, shot_speed=100, time_to_live=0.01, action_blocking=True)
    player = Agent(pos=0.1, facing=Direction.EAST, collision_layer=CL_PLAYER, shot_collision_layer=CL_PLAYER_SHOTS)
    player.weapon = sword
    enemy = Agent(pos=0.9)
    enemy.weapon = sword
    map = [player, enemy]
    env = Env(map)
    state = env.reset()

    actions = [['FORWARD', 'PASS'], ['FORWARD', 'PASS'], ['FORWARD', 'BACKWARD'], ['FORWARD', 'FORWARD'], ['FORWARD', 'FORWARD'], ['FORWARD', 'FORWARD'], ['FORWARD', 'PASS'], ['FORWARD', 'PASS'], ['FORWARD', 'ATTACK']]

    for agent1_action, agent2_action in actions[:-1]:
        action = [Action[agent1_action], Action[agent2_action]]
        print(action)
        state, _, _, _ = env.step(action)
        print(state)

        for i, item in enumerate(state.dynamics):
            # no wall hacking
            assert state.get_sorted_collision_map()[1].pos <= item.pos <= state.get_sorted_collision_map()[-2].pos

    agent1_action, agent2_action = actions[-1]
    action = [Action[agent1_action], Action[agent2_action]]
    print(action)
    state, _, _, _ = env.step(action)
    print(state)

    for i, item in enumerate(state.dynamics):
        # no wall hacking
        cmap = state.get_sorted_collision_map()
        if not cmap[1].pos <= item.faces[0].pos <= cmap[-2].pos:
            print(item)
        assert cmap[1].pos <= item.faces[0].pos <= cmap[-2].pos

        if not cmap[1].pos <= item.faces[1].pos <= cmap[-2].pos:
            print(item)
        assert cmap[1].pos <= item.faces[1].pos <= cmap[-2].pos

    for i, item in enumerate(state.dynamics):
        for j, next_item in enumerate(state.dynamics):
            if i != j:
                assert item.pos != next_item.pos




def test_random_seed_42():
    random.seed(42)
    sword = Weapon(damage=10, shot_speed=100, time_to_live=0.01, action_blocking=True)
    player = Agent(pos=0.1, facing=Direction.EAST, collision_layer=CL_PLAYER, shot_collision_layer=CL_PLAYER_SHOTS)
    player.weapon = sword
    enemy = Agent(pos=0.9)
    enemy.weapon = sword
    map = [player, enemy]
    env = Env(map)
    state = env.reset()

    actions = []

    for _ in range(100):
        action = [Action.FORWARD, Action(random.choice(range(4)))]
        print(action)
        actions += [action]
        state, _, _, _ = env.step(action)
        cmap = state.get_sorted_collision_map()
        west_wall, east_wall = cmap[1].pos, cmap[-2].pos
        print(state)

        for i, item in enumerate(state.dynamics):
            # no wall hacking
            if not (state.get_sorted_collision_map()[1].pos <= item.pos <= state.get_sorted_collision_map()[-2].pos):
                print(item)
                print([[action[0].name, action[1].name] for action in actions])

            cmap = state.get_sorted_collision_map()
            assert west_wall <= item.faces[0].pos <= east_wall
            assert west_wall <= item.faces[1].pos <= east_wall


        # enhance this to check for overlapping objects
        # objects should not be at the same position if they can collide
        for i, item in enumerate(state.dynamics):
            for j, next_item in enumerate(state.dynamics):
                if i != j:
                    if item.pos == next_item.pos and (collision_handler.can_collide(item, next_item) or collision_handler.can_collide(next_item, item)):
                        print([[action[0].name, action[1].name] for action in actions])
                        print(item, item.collision_layer, next_item, next_item.collision_layer)
                        assert item.pos != next_item.pos
