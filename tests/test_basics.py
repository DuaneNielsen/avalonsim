from main import Wall, Agent, Weapon
from main import Action, Direction
from main import CL_PLAYER, CL_ENEMY, CL_PLAYER_SHOTS, CL_ENEMY_SHOTS
from main import Env
from timer import TimerQueue, Timer
from main import close, between, overlap
import random
from main import collision_handler
from main import State
from main import VelFacePathIter
from main import get_contact_groups
from main import DirectedVertexBodyIter, Vertex

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


def test_overlap():
    width = Agent().width
    agent1 = Agent(pos=0.1)
    agent2 = Agent(pos=0.1 + width + 1e-7)
    assert between(agent1, 0.1 - width / 2 + 1e-3)
    assert not between(agent1, 0.1 - width / 2)
    assert between(agent1, 0.1 + width / 2 - 1e-3)
    assert not between(agent1, 0.1 + width / 2)

    print()
    print(agent1.faces, agent2.faces)
    assert not overlap(agent1, agent2)
    assert not overlap(agent2, agent1)
    agent2.pos -= 1e-7
    assert overlap(agent1, agent2)
    assert overlap(agent2, agent1)

    agent1 = Agent(pos=0.9945007000000003)
    agent2 = Agent(pos=0.9945014000000001)
    assert overlap(agent1, agent2)


def test_iterator():
    sword = Weapon(damage=10, shot_speed=0.5, time_to_live=0.04, cooldown_time=0.1, shot_width=0.01, windup_time=0.1, recovery_time=0.04)
    bow = Weapon(damage=3, shot_speed=0.7, time_to_live=1, cooldown_time=0.3, windup_time=0.3)
    player = Agent(pos=0.1, facing=Direction.EAST, collision_layer=CL_PLAYER, shot_collision_layer=CL_PLAYER_SHOTS)
    player.weapon = sword
    enemy = Agent(pos=0.9, facing=Direction.WEST)
    enemy.weapon = bow
    player.add_vertex(Vertex(-0.5))
    player.add_vertex(Vertex(0.3))
    enemy.add_vertex(Vertex(-0.5))
    enemy.add_vertex(Vertex(0.3))
    west_wall, east_wall = Wall(0, Direction.EAST), Wall(1., Direction.WEST)
    sorted_map = [west_wall, player, enemy, east_wall]
    player.vel = 0.2
    enemy.vel = -0.2

    order = []

    print('')

    for vertex, face in DirectedVertexBodyIter(sorted_map, 1):
        order += [(vertex, face)]

    assert len(order) == 4

    assert order[0][0].pos == 0.3 + player.pos
    assert order[1][0].pos == 0.3 + player.pos
    assert order[2][0].pos == -0.5 + player.pos
    assert order[3][0].pos == -0.5 + player.pos

    assert order[0][1].pos == enemy.pos - enemy.width / 2
    assert order[1][1].pos == east_wall.pos - east_wall.width / 2
    assert order[2][1].pos == enemy.pos - enemy.width / 2
    assert order[3][1].pos == east_wall.pos - east_wall.width / 2

    order = []

    for vertex, face in DirectedVertexBodyIter(sorted_map, 2):
        order += [(vertex, face)]

    assert len(order) == 4

    assert order[0][0].pos == enemy.pos - 0.5
    assert order[1][0].pos == enemy.pos - 0.5
    assert order[2][0].pos == enemy.pos + 0.3
    assert order[3][0].pos == enemy.pos + 0.3

    assert order[0][1].pos == player.pos + player.width / 2
    assert order[1][1].pos == west_wall.pos + west_wall.width / 2
    assert order[2][1].pos == player.pos + player.width / 2
    assert order[3][1].pos == west_wall.pos + west_wall.width / 2


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
    expected_pos = 1.0 - state.agents[0].width / 2 - state.statics[1].width / 2
    assert close(state.agents[0].pos, expected_pos)
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
    expected_pos = state.agents[0].width / 2 + state.statics[1].width / 2
    assert close(state.agents[0].pos, expected_pos)
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
    exp_pos = 0.0 + state.statics[0].width / 2 + state.agents[0].width / 2
    assert close(state.agents[0].pos, exp_pos)
    assert state.agents[0].vel == 0.0
    exp_pos = 1.0 - state.statics[1].width / 2 - state.agents[1].width / 2
    assert close(state.agents[1].pos, exp_pos)
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
    timer1, timer2, timer3 = Timer(1.), Timer(2.), Timer(3.)

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

    actions = [['FORWARD', 'PASS'], ['FORWARD', 'PASS'], ['FORWARD', 'BACKWARD'], ['FORWARD', 'FORWARD'],
               ['FORWARD', 'FORWARD'], ['FORWARD', 'FORWARD'], ['FORWARD', 'PASS'], ['FORWARD', 'PASS'],
               ['FORWARD', 'ATTACK']]

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
    def verify(state):
        for i, item in enumerate(state.dynamics):

            if not west_wall <= item.faces[0].pos <= east_wall:
                print("face penetrated_wall", item.faces[0], "west_pos", west_wall, "east_pos", east_wall)
            if not west_wall <= item.faces[1].pos <= east_wall:
                print("face penetrated_wall", item.faces[1], "west_pos", west_wall, "east_pos", east_wall)

            assert west_wall <= item.pos <= east_wall
            assert west_wall <= item.faces[0].pos <= east_wall
            assert west_wall <= item.faces[1].pos <= east_wall

        # enhance this to check for overlapping objects
        # objects should not be at the same position if they can collide
        map = state.get_sorted_base_map()
        for i, item in enumerate(map):
            for j, next_item in enumerate(map):
                if i != j:
                    can_constrain = collision_handler.can_constrain(item, next_item) \
                                    or collision_handler.can_constrain(next_item, item)
                    if overlap(item, next_item) and can_constrain:
                        print([[action[0].name, action[1].name] for action in actions])
                        print("OVERLAP", item, item.collision_layer, next_item, next_item.collision_layer)
                        assert False

                    if item.pos == next_item.pos and can_constrain:
                        print([[action[0].name, action[1].name] for action in actions])
                        print("EQUAL", item, item.collision_layer, next_item, next_item.collision_layer)
                        assert False

    random.seed(42)
    sword = Weapon(damage=10, shot_speed=100, time_to_live=0.00001, action_blocking=True)
    player = Agent(pos=0.1, facing=Direction.EAST, collision_layer=CL_PLAYER, shot_collision_layer=CL_PLAYER_SHOTS)
    player.weapon = sword
    enemy = Agent(pos=0.9)
    enemy.weapon = sword
    map = [player, enemy]
    env = Env(map)
    state = env.reset()

    actions = []

    for i in range(400):
        action = [Action.FORWARD, Action(random.choice(range(4)))]
        print(action)
        actions += [action]
        state, _, _, _ = env.step(action)
        cmap = state.get_sorted_collision_map()
        west_wall, east_wall = cmap[1].pos, cmap[-2].pos
        print(i, state)
        verify(state)

    action = [Action.FORWARD, Action(random.choice(range(4)))]
    print(action)
    actions += [action]
    state, _, _, _ = env.step(action)
    cmap = state.get_sorted_collision_map()
    west_wall, east_wall = cmap[1].pos, cmap[-2].pos
    print(state)
    verify(state)


def test_no_action():
    sword = Weapon(damage=10, shot_speed=100, time_to_live=0.01, action_blocking=True)
    bow = Weapon(damage=3, shot_speed=0.3, time_to_live=0.5)
    player = Agent(pos=0.1, facing=Direction.EAST, collision_layer=CL_PLAYER, shot_collision_layer=CL_PLAYER_SHOTS)
    player.weapon = sword
    enemy = Agent(pos=0.9)
    enemy.weapon = sword
    map = [player, enemy]
    env = Env(map)
    actions = [
        ['FORWARD', 'PASS'],
        ['FORWARD', 'PASS'],
        ['FORWARD', 'BACKWARD'],
        ['FORWARD', 'FORWARD'],
        ['FORWARD', 'FORWARD'],
        ['FORWARD', 'FORWARD'],
        ['BACKWARD', 'PASS'],
        ['BACKWARD', 'PASS'],
        ['BACKWARD', 'ATTACK'],
        ['BACKWARD', 'PASS'],
        ['FORWARD', 'PASS']]

    def encode(a):
        return Action[a[0]], Action[a[1]]

    actions = [encode(a) for a in actions]

    print('')
    for action in actions:
        print([a.name for a in action])
        state, _, _, _ = env.step(action)
        print(state)


def test_directional_iterator():
    sword = Weapon(damage=10, shot_speed=100, time_to_live=0.01, action_blocking=True)
    bow = Weapon(damage=3, shot_speed=0.3, time_to_live=0.5)
    player = Agent(pos=0.1, facing=Direction.EAST, collision_layer=CL_PLAYER, shot_collision_layer=CL_PLAYER_SHOTS)
    player.weapon = sword
    player.vel = 0.1
    enemy = Agent(pos=0.9)
    enemy.weapon = sword
    enemy.vel = - 0.1
    map = [Wall(0, Direction.EAST), player, enemy, Wall(1., Direction.WEST)]
    state_map = State(map).get_sorted_base_map()

    print('')

    for item, item_on_path in VelFacePathIter(state_map, 0):
        assert False

    for item, item_on_path in VelFacePathIter(state_map, 1):
        assert item.side == item_on_path.side

    for item, item_on_path in VelFacePathIter(state_map, 2):
        assert item.side == item_on_path.side

    for item, item_on_path in VelFacePathIter(state_map, 3):
        assert False


def test_contact_groups():
    sword = Weapon(damage=10, shot_speed=100, time_to_live=0.01, action_blocking=True)
    bow = Weapon(damage=3, shot_speed=0.3, time_to_live=0.5)
    player = Agent(pos=0.1, facing=Direction.EAST, collision_layer=CL_PLAYER, shot_collision_layer=CL_PLAYER_SHOTS)
    width = player.width
    player.weapon = sword
    player.vel = 0.1
    enemy = Agent(pos=0.9)
    enemy.weapon = sword
    enemy.vel = - 0.1
    map = [Wall(0, Direction.EAST), player, enemy, Wall(1., Direction.WEST)]

    state = State(map)
    sorted_map = state.get_sorted_base_map()
    # contact_groups = get_contact_groups(sorted_map)
    # assert len(contact_groups) == 0

    enemy.pos = player.pos + player.width
    contact_groups = get_contact_groups(sorted_map)
    assert len(contact_groups) == 1
    assert len(contact_groups[0]) == 2
    assert player.id == contact_groups[0][0].id
    assert enemy.id == contact_groups[0][1].id

    player1 = Agent(pos=0.1, facing=Direction.EAST, collision_layer=CL_PLAYER, shot_collision_layer=CL_PLAYER_SHOTS)
    enemy1 = Agent(pos=0.1 + width)
    player2 = Agent(pos=0.1 + 2 * width, facing=Direction.EAST, collision_layer=CL_PLAYER,
                    shot_collision_layer=CL_PLAYER_SHOTS)
    enemy2 = Agent(pos=0.1 + 4 * width)
    player3 = Agent(pos=0.1 + 5 * width, facing=Direction.EAST, collision_layer=CL_PLAYER,
                    shot_collision_layer=CL_PLAYER_SHOTS)

    map = [Wall(0, Direction.EAST), player1, enemy1, player2, enemy2, player3, Wall(1., Direction.WEST)]

    state = State(map)
    sorted_map = state.get_sorted_base_map()
    contact_groups = get_contact_groups(sorted_map)
    assert len(contact_groups) == 2
    assert contact_groups[0][0].id == player1.id
    assert contact_groups[0][1].id == enemy1.id
    assert contact_groups[0][2].id == player2.id
    assert contact_groups[1][0].id == enemy2.id
    assert contact_groups[1][1].id == player3.id


def test_attackfest():
    sword = Weapon(damage=10, shot_speed=100, time_to_live=0.01, action_blocking=True)
    bow = Weapon(damage=3, shot_speed=0.3, time_to_live=0.5)
    player = Agent(pos=0.1, facing=Direction.EAST, collision_layer=CL_PLAYER, shot_collision_layer=CL_PLAYER_SHOTS)
    player.weapon = sword
    enemy = Agent(pos=0.9, facing=Direction.WEST)
    enemy.weapon = sword
    map = [player, enemy]
    env = Env(map)

    actions = [['FORWARD', 'PASS'], ['ATTACK', 'PASS'], ['ATTACK', 'BACKWARD'], ['ATTACK', 'FORWARD'],
               ['ATTACK', 'FORWARD'], ['ATTACK', 'FORWARD'], ['ATTACK', 'PASS'], ['ATTACK', 'PASS'],
               ['ATTACK', 'ATTACK'], ['ATTACK', 'PASS'], ['ATTACK', 'PASS'], ['ATTACK', 'PASS'], ['ATTACK', 'FORWARD'],
               ['ATTACK', 'FORWARD']]

    def encode(a):
        return Action[a[0]], Action[a[1]]

    actions = [encode(a) for a in actions]

    def verify(state):
        cmap = state.get_sorted_collision_map()
        west_wall, east_wall = cmap[1].pos, cmap[-2].pos
        shots = state.dynamics[2:]
        if len(state.shots) > 0:
            if state.shots[0].collision_layer == CL_ENEMY_SHOTS:
                assert state.agents[0].faces[1].pos <= state.shots[0].faces[0].pos
                assert state.agents[1].pos >= state.shots[0].faces[1].pos

            if state.shots[0].collision_layer == CL_PLAYER_SHOTS:
                assert state.agents[0].pos <= state.shots[0].faces[0].pos
                assert state.agents[1].faces[0].pos >= state.shots[0].faces[1].pos

        print("SHOTS", shots)
        pass

    print('')

    actions = actions[:10]

    for i, action in enumerate(actions[:-1]):
        print(i, action)
        state, _, _, _ = env.step(action)
        print(i, state)
        verify(state)

    action = actions[-1]
    print(action)
    state, _, _, _ = env.step(actions[-1])
    print(state)
    verify(state)
