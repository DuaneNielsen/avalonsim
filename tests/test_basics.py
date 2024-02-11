from main import Wall, Agent, Weapon
from main import ACTION_FORWARD, ACTION_BACKWARD, ACTION_ATTACK, ACTION_PASS
from main import FACING_FORWARD, FACING_BACKWARD
from main import CL_PLAYER, CL_ENEMY, CL_PLAYER_SHOTS, CL_ENEMY_SHOTS
from main import EPS_DIST
from main import Env
from main import near
from main import TimerQueue, Timer
import random


def test_walls_and_simple_movement():
    print()

    map = [Wall(pos=0), Wall(pos=1.), Agent(pos=0.5, facing=FACING_FORWARD, collision_layer=CL_PLAYER)]
    env = Env(map)
    state = env.reset()
    assert state.agents[0].pos == 0.5
    print("FORWARD")
    state, reward, done, info = env.step([ACTION_FORWARD])
    print(state)
    assert near(state.agents[0].pos, 1.0 - EPS_DIST)
    assert state.agents[0].vel == 0.
    print("BACKWARD")
    state, reward, done, info = env.step([ACTION_BACKWARD])
    print(state)
    assert near(state.agents[0].pos, 0.0 + EPS_DIST)
    assert state.agents[0].vel == 0.0


def test_two_creatures_and_walls():
    agent1 = Agent(pos=0.5, facing=FACING_FORWARD, collision_layer=CL_PLAYER)
    agent2 = Agent(pos=0.5, facing=FACING_FORWARD, collision_layer=CL_PLAYER)
    map = [Wall(pos=0), Wall(pos=1.), agent1, agent2]
    env = Env(map)
    state = env.reset()
    assert state.agents[0].pos == 0.5
    assert state.agents[1].pos == 0.5
    print("FORWARD")
    state, reward, done, info = env.step([ACTION_FORWARD, ACTION_FORWARD])
    print(state)
    assert near(state.agents[0].pos, 1.0 - EPS_DIST)
    assert state.agents[0].vel == 0.
    assert near(state.agents[1].pos, 1.0 - EPS_DIST)
    assert state.agents[1].vel == 0.
    print("BACKWARD")
    state, reward, done, info = env.step([ACTION_BACKWARD, ACTION_BACKWARD])
    print(state)
    assert near(state.agents[0].pos,  EPS_DIST)
    assert state.agents[0].vel == 0.0
    assert near(state.agents[1].pos,  EPS_DIST)
    assert state.agents[1].vel == 0.0


def test_creatures_and_walls_reverse_facing():
    agent1 = Agent(pos=0.5, facing=FACING_BACKWARD, collision_layer=CL_PLAYER)
    agent2 = Agent(pos=0.5, facing=FACING_BACKWARD, collision_layer=CL_PLAYER)
    map = [Wall(pos=0), Wall(pos=1.), agent1, agent2]
    env = Env(map)
    state = env.reset()
    assert state.agents[0].pos == 0.5
    assert state.agents[1].pos == 0.5
    print("FORWARD")
    state, reward, done, info = env.step([ACTION_FORWARD, ACTION_FORWARD])
    print(state)
    assert near(state.agents[0].pos,  EPS_DIST)
    assert state.agents[0].vel == 0.
    assert near(state.agents[1].pos,  EPS_DIST)
    assert state.agents[1].vel == 0.
    print("BACKWARD")
    state, reward, done, info = env.step([ACTION_BACKWARD, ACTION_BACKWARD])
    print(state)
    assert near(state.agents[0].pos, 1.0 - EPS_DIST)
    assert state.agents[0].vel == 0.0
    assert near(state.agents[1].pos, 1.0 - EPS_DIST)
    assert state.agents[1].vel == 0.0


def test_double_walls_and_simple_movement():
    print()
    map = [Wall(pos=0), Wall(pos=0), Wall(pos=1.), Wall(pos=1.), Agent(pos=0.5, facing=FACING_FORWARD, collision_layer=CL_PLAYER)]
    env = Env(map)
    state = env.reset()
    assert state.agents[0].pos == 0.5
    print("FORWARD")
    state, reward, done, info = env.step([ACTION_FORWARD])
    print(state)
    assert near(state.agents[0].pos, 1.0 - EPS_DIST)
    assert state.agents[0].vel == 0.
    print("BACKWARD")
    state, reward, done, info = env.step([ACTION_BACKWARD])
    print(state)
    assert near(state.agents[0].pos,  EPS_DIST)
    assert state.agents[0].vel == 0.0


def test_double_walls_reverse_facing():
    map = [Wall(pos=0), Wall(pos=0), Wall(pos=1.), Wall(pos=1.), Agent(pos=0.5, facing=FACING_BACKWARD, collision_layer=CL_PLAYER)]
    env = Env(map)
    state = env.reset()
    assert state.agents[0].pos == 0.5
    print("FORWARD")
    state, reward, done, info = env.step([ACTION_FORWARD])
    print(state)
    assert near(state.agents[0].pos,  EPS_DIST)
    assert state.agents[0].vel == 0.
    print("BACKWARD")
    state, reward, done, info = env.step([ACTION_BACKWARD])
    print(state)
    assert near(state.agents[0].pos, 1.0 - EPS_DIST)
    assert state.agents[0].vel == 0.0


def test_agent_agent_collision_simple():
    # two agents meet halfway
    agent1 = Agent(pos=0., facing=FACING_FORWARD, collision_layer=CL_PLAYER)
    agent2 = Agent(pos=1., facing=FACING_BACKWARD, collision_layer=CL_ENEMY)
    map = [Wall(pos=0), Wall(pos=1.), agent1, agent2]
    env = Env(map)
    state = env.reset()
    assert state.agents[0].pos == 0.
    assert state.agents[1].pos == 1.
    print("FORWARD")
    state, reward, done, info = env.step([ACTION_FORWARD, ACTION_FORWARD])
    print(state)
    assert near(state.agents[0].pos, 0.5 - EPS_DIST)
    assert state.agents[0].vel == 0.
    assert near(state.agents[1].pos, 0.5 + EPS_DIST)
    assert state.agents[1].vel == 0.
    print("BACKWARD")
    state, reward, done, info = env.step([ACTION_BACKWARD, ACTION_BACKWARD])
    print(state)
    assert near(state.agents[0].pos,  EPS_DIST)
    assert state.agents[0].vel == 0.0
    assert near(state.agents[1].pos, 1.0 - EPS_DIST)
    assert state.agents[1].vel == 0.0


def test_agent_agent_collision_chase():
    # two agents meet halfway
    agent1 = Agent(pos=0., facing=FACING_FORWARD, collision_layer=CL_PLAYER)
    agent2 = Agent(pos=0.05, walk_speed=0.05, facing=FACING_FORWARD, collision_layer=CL_ENEMY)
    map = [Wall(pos=0), Wall(pos=1.), agent1, agent2]
    env = Env(map)
    state = env.reset()
    state, reward, done, info = env.step([ACTION_FORWARD, ACTION_FORWARD])
    print(state)
    assert near(state.agents[0].pos, 0.1 - EPS_DIST)
    assert state.agents[0].vel == 0.
    assert near(state.agents[1].pos, 0.1 + EPS_DIST)
    assert state.agents[1].vel == 0.

    # two agents meet halfway
    agent1 = Agent(pos=1., walk_speed=0.1, facing=FACING_BACKWARD, collision_layer=CL_PLAYER)
    agent2 = Agent(pos=0.95, walk_speed=0.05, facing=FACING_BACKWARD, collision_layer=CL_ENEMY)
    map = [Wall(pos=0), Wall(pos=1.), agent1, agent2]
    env = Env(map)
    state = env.reset()
    state, reward, done, info = env.step([ACTION_FORWARD, ACTION_FORWARD])
    print(state)
    assert near(state.agents[0].pos, 0.9 + EPS_DIST)
    assert state.agents[0].vel == 0.
    assert near(state.agents[1].pos, 0.9 - EPS_DIST)
    assert state.agents[1].vel == 0.


def test_agent_agent_collision_wait():
    agent1 = Agent(pos=0.1, facing=FACING_FORWARD, collision_layer=CL_PLAYER)
    agent2 = Agent(pos=0.5, walk_speed=0.05, facing=FACING_BACKWARD, collision_layer=CL_ENEMY)
    map = [Wall(pos=0), Wall(pos=1.), agent1, agent2]
    env = Env(map)
    state = env.reset()
    state, reward, done, info = env.step([ACTION_PASS, ACTION_FORWARD])
    print(state)
    assert near(state.agents[0].pos,  0.1)
    assert state.agents[0].vel == 0.
    #if you were not moving on the collision you stay put
    assert near(state.agents[1].pos, 0.1 + EPS_DIST)
    assert state.agents[1].vel == 0.


def test_3_agent_collision():
    agent1 = Agent(pos=0., facing=FACING_FORWARD, collision_layer=CL_PLAYER)
    agent2 = Agent(pos=0., facing=FACING_FORWARD, collision_layer=CL_PLAYER)
    agent3 = Agent(pos=1., facing=FACING_BACKWARD, collision_layer=CL_ENEMY)
    map = [Wall(pos=0), Wall(pos=1.), agent1, agent2, agent3]
    env = Env(map)
    state = env.reset()
    assert state.agents[0].pos == 0.
    assert state.agents[1].pos == 0.
    assert state.agents[2].pos == 1.
    print("FORWARD")
    state, reward, done, info = env.step([ACTION_FORWARD, ACTION_FORWARD, ACTION_FORWARD])
    print(state)
    assert near(state.agents[0].pos, 0.5 - EPS_DIST)
    assert state.agents[0].vel == 0.
    assert near(state.agents[1].pos, 0.5 - EPS_DIST)
    assert state.agents[1].vel == 0.
    assert near(state.agents[2].pos, 0.5 + EPS_DIST)
    assert state.agents[2].vel == 0.


def test_spawn_shot():
    print()
    player = Agent(pos=0.5, facing=FACING_FORWARD, collision_layer=CL_PLAYER, shot_collision_layer=CL_PLAYER_SHOTS)
    sword = Weapon(damage=10, shot_speed=100, time_to_live=0.01, action_blocking=True)
    player.weapon = sword

    map = [Wall(pos=0), Wall(pos=1.), player]
    env = Env(map)
    state = env.reset()
    assert state.agents[0].pos == 0.5
    assert len(state.dynamics) == 1
    state, reward, done, info = env.step([ACTION_ATTACK])
    # the shot should hit the wall
    assert len(state.dynamics) == 2
    assert near(state.dynamics[1].pos,  1. - EPS_DIST)
    assert state.dynamics[1].vel == 0.
    print(state)


def test_sword():
    print()
    player = Agent(pos=0.4, facing=FACING_FORWARD, collision_layer=CL_PLAYER, shot_collision_layer=CL_PLAYER_SHOTS)
    sword = Weapon(damage=10, shot_speed=100, time_to_live=0.01, action_blocking=True)
    player.weapon = sword
    enemy = Agent(pos=0.6, facing=FACING_BACKWARD, collision_layer=CL_ENEMY, shot_collision_layer=CL_ENEMY_SHOTS)
    sword = Weapon(damage=10, shot_speed=100, time_to_live=0.01, action_blocking=True)
    enemy.weapon = sword
    map = [Wall(pos=0), Wall(pos=1.), player, enemy]
    env = Env(map)
    state = env.reset()
    assert state.agents[0].pos == 0.4
    assert state.agents[1].pos == 0.6
    assert len(state.dynamics) == 2
    state, reward, done, info = env.step([ACTION_ATTACK, ACTION_ATTACK])
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


def test_random_seed_42():

    random.seed(42)
    sword = Weapon(damage=10, shot_speed=100, time_to_live=0.01, action_blocking=True)
    player = Agent(facing=FACING_FORWARD, collision_layer=CL_PLAYER, shot_collision_layer=CL_PLAYER_SHOTS)
    player.weapon = sword
    enemy = Agent(pos=1.)
    enemy.weapon = sword
    map = [Wall(pos=0), Wall(pos=1.), player, enemy]
    env = Env(map)
    state = env.reset()

    for _ in range(10):
        action = [ACTION_FORWARD, random.choice(range(4))]
        print(action)
        state, _, _, _ = env.step(action)
        print(state)

        for key, item in state.items():

            # no wall hacking
            assert 0. <= item.pos <= 1.
