from math import inf
import uuid
from enum import IntEnum
from copy import deepcopy
import numpy as np
from avalonsim.timer import TimerQueue, Timer
from avalonsim.render import start_screen, draw_rect, to_screen
import gymnasium as gym
import pygame
from math import floor, ceil
import copy
from typing import Optional


def print(*args):
    pass


class CollisionLayer(IntEnum):
    WALLS = 0
    PLAYER = 1
    ENEMY = 2
    PLAYER_SHOTS = 3
    ENEMY_SHOTS = 4
    RANGEFINDER = 5


class Direction(IntEnum):
    EAST = 1
    WEST = -1


class FaceDirection(IntEnum):
    FRONT = 0
    BACK = 1


class AgentState(IntEnum):
    READY = 0
    WINDUP = 1
    RECOVERY = 2


def reverse_facing(facing):
    return Direction.EAST if facing == facing.WEST else Direction.WEST


class Action(IntEnum):
    NOOP = 0
    FORWARD = 1
    BACKWARD = 2
    ATTACK = 3
    REVERSE_FACING = 4


def sign(x):
    return (x > 0) - (x < 0) if x != 0 else 0


def next_id(length=8):
    while True:
        uid = str(uuid.uuid4())[:length]  # Generate UUID and truncate it
        yield uid


def lookahead(list, i):
    if i + 1 < len(list):
        return list[i + 1]
    else:
        return None


class Face:
    def __init__(self, parent, pos, side):
        self.parent = parent
        self.pos = pos
        self.side = side
        self.collision_layer = parent.collision_layer

    def same_parent(self, face):
        return self.parent.id == face.parent.id

    def __repr__(self):
        return f"{self.parent.__class__.__name__} {self.parent.id} {self.side.name} pos: {self.pos} vel: {self.parent.vel}"


def close(a, b, tol=1e-6):
    return abs(a - b) < tol


def between(base, pos):
    faces = sorted(base.faces, key=lambda x: x.pos)
    return faces[0].pos < pos < faces[1].pos


def overlap(base1, base2):
    return between(base1, base2.faces[0].pos) or between(base1, base2.faces[1].pos)


class Vertex:
    def __init__(self, pos):
        self.parent = None
        self._pos = pos
        self.collision_layer = None

    @property
    def pos(self):
        return self.parent.pos + self._pos

    def __repr__(self):
        return f"{self.__class__.__name__} {self.pos}"


class RangeFinder(Vertex):
    def __init__(self, pos):
        super().__init__(pos)
        self.collision_layer = CollisionLayer.RANGEFINDER


class Body:
    def __init__(self):
        self.id = None
        self.pos = 0
        self.vel = 0
        self.width = 0.001
        self.facing = Direction.EAST
        self.delete = False
        self.collision_layer = ""
        self._vertices = []

    @property
    def faces(self):
        if self.facing == Direction.EAST:
            return [
                Face(self, self.pos - self.width / 2, FaceDirection.BACK),
                Face(self, self.pos + self.width / 2, FaceDirection.FRONT)
            ]
        else:
            return [
                Face(self, self.pos - self.width / 2, FaceDirection.FRONT),
                Face(self, self.pos + self.width / 2, FaceDirection.BACK)
            ]

    def add_vertex(self, vertex):
        vertex.parent = self
        self._vertices += [vertex]
        self._vertices.sort(key=lambda vertex: vertex.pos)

    @property
    def vertices(self):
        return self._vertices

    def direction(self, other):
        return Direction.WEST if other.pos < self.pos else Direction.EAST

    def moving_in_direction_of(self, other):
        return sign(self.pos - other.pos) * self.vel < 0.

    def moving_same_direction(self, other):
        return sign(self.vel) == sign(other.vel)

    def __repr__(self):
        return f"{self.__class__.__name__} {self.id} face: {self.facing.name} pos: {self.pos} vel: {self.vel}"

    def as_numpy(self):
        return np.array([self.pos - self.width/2, self.pos + self.width/2, self.vel])


class Static(Body):
    def __init__(self):
        super().__init__()


class Dynamic(Body):
    def __init__(self):
        super().__init__()


class Wall(Static):
    def __init__(self, pos, facing):
        super().__init__()
        self.pos = pos
        self.facing = facing
        self.collision_layer = CollisionLayer.WALLS
        self.width = 0.001

    def __repr__(self):
        return str(self.__class__) + " face:" + str(self.facing) + " pos: " + str(self.pos)


class Agent(Dynamic):
    def __init__(self, pos=0., facing=Direction.WEST, walk_speed=0.1, hp_max=100, collision_layer=CollisionLayer.ENEMY,
                 shot_collision_layer=CollisionLayer.ENEMY_SHOTS):
        super().__init__()
        self.pos = pos
        self.facing = facing
        self.walk_speed = walk_speed
        self.hp = hp_max
        self.hp_max = hp_max
        self.weapon = None
        self.action_ready = True  # able to take an action
        self.collision_layer = collision_layer
        self.shot_collision_layer = shot_collision_layer
        self.width = 0.01
        self.state = AgentState.READY

    def as_numpy(self):
        return np.array([self.pos - self.width/2, self.pos + self.width/2, self.vel, self.hp / self.hp_max, self.state])


class Shot(Dynamic):
    def __init__(self, facing, pos, vel, damage, collision_layer, width):
        super().__init__()
        self.facing = facing
        self.pos = pos
        self.vel = vel
        self.damage = damage
        self.collision_layer = collision_layer
        self.width = width
        self.timer = None

    def as_numpy(self):
        return np.array([self.pos, self.vel, self.width, self.damage/100])


body_size_registry = {
    CollisionLayer.WALLS: Wall(0., FaceDirection.FRONT).as_numpy().shape,
    CollisionLayer.PLAYER: Agent().as_numpy().shape,
    CollisionLayer.ENEMY: Agent().as_numpy().shape,
    CollisionLayer.PLAYER_SHOTS: Shot(Direction.EAST, 0., 0., 10, CollisionLayer.PLAYER_SHOTS, 0.05).as_numpy().shape,
    CollisionLayer.ENEMY_SHOTS: Shot(Direction.EAST, 0., 0., 10, CollisionLayer.PLAYER_SHOTS, 0.05).as_numpy().shape
}


class CollisionHandler:
    def __init__(self):
        self.handlers = {}

    def add_handler(self, layer_name1, layer_name2, callback, can_constrain=False):
        self.handlers[(layer_name1, layer_name2)] = callback, can_constrain

    def can_collide(self, obj1, obj2):
        key = obj1.collision_layer, obj2.collision_layer
        if key in self.handlers:
            return True
        else:
            return False

    def can_constrain(self, obj1, obj2):
        can_constrain = False
        layer_name1 = obj1.collision_layer
        layer_name2 = obj2.collision_layer
        handler = self.handlers.get((layer_name1, layer_name2))
        if handler:
            _, can_constrain = handler
        return can_constrain

    def handle_collision(self, obj1, obj2):
        layer_name1 = obj1.collision_layer
        layer_name2 = obj2.collision_layer
        handler = self.handlers.get((layer_name1, layer_name2))
        if handler:
            handler_cb, _ = handler
            handler_cb.collide(obj1, obj2)


class Collision:
    def can_collide(self, face1, face2):
        return True

    # effects are applied to objects the moment they collide
    def collide(self, object, other):
        pass


class StaticCollision(Collision):

    def collide(self, static_object, other):
        other.parent.vel = 0


static_stop = StaticCollision()


class DynamicCollision(Collision):
    def collide(self, face, target_frace):
        if face.parent.moving_same_direction(target_frace.parent):
            target_frace.parent.vel = min(face.parent.vel, target_frace.parent.vel)
        else:
            target_frace.parent.vel = 0


dynamic_stop = DynamicCollision()


class ApplyAndDeleteShot(Collision):

    def collide(self, face, target_face):
        target_face.parent.hp -= face.parent.damage
        face.parent.vel = 0.
        face.parent.delete = True


apply_damage_and_delete = ApplyAndDeleteShot()


class DeleteSelf(Collision):
    def collide(self, face, target_face):
        face.parent.vel = 0.
        face.parent.delete = True


delete_self = DeleteSelf()


class RangeFinderCollision(Collision):
    pass


rangefinder = RangeFinderCollision()

collision_handler = CollisionHandler()
collision_handler.add_handler(CollisionLayer.PLAYER, CollisionLayer.ENEMY, dynamic_stop, can_constrain=True)
collision_handler.add_handler(CollisionLayer.ENEMY, CollisionLayer.PLAYER, dynamic_stop, can_constrain=True)
collision_handler.add_handler(CollisionLayer.WALLS, CollisionLayer.PLAYER, static_stop, can_constrain=True)
collision_handler.add_handler(CollisionLayer.WALLS, CollisionLayer.ENEMY, static_stop, can_constrain=True)
collision_handler.add_handler(CollisionLayer.PLAYER_SHOTS, CollisionLayer.WALLS, delete_self)
collision_handler.add_handler(CollisionLayer.ENEMY_SHOTS, CollisionLayer.WALLS, delete_self)
collision_handler.add_handler(CollisionLayer.ENEMY_SHOTS, CollisionLayer.PLAYER, apply_damage_and_delete)
collision_handler.add_handler(CollisionLayer.PLAYER_SHOTS, CollisionLayer.ENEMY, apply_damage_and_delete)
collision_handler.add_handler(CollisionLayer.RANGEFINDER, CollisionLayer.PLAYER, rangefinder)
collision_handler.add_handler(CollisionLayer.RANGEFINDER, CollisionLayer.ENEMY, rangefinder)


class ShotTimer(Timer):
    def __init__(self, t, shot):
        super().__init__(t)
        self.shot = shot

    def on_expire(self, env):
        self.shot.delete = True


class WindupTimer(Timer):
    def __init__(self, t, weapon, agent):
        super().__init__(t)
        self.weapon = weapon
        self.agent = agent

    def on_expire(self, env):
        shot = self.agent.weapon.shoot(env.t, env.timers, self.agent.pos, self.agent.facing,
                                       self.agent.shot_collision_layer)
        env.state.append(shot)
        if self.agent.weapon.recovery_time > 0.:
            self.agent.state = AgentState.RECOVERY
            timer = RecoveryTimer(env.t + self.weapon.recovery_time, self.agent)
            env.timers.push(timer)
        else:
            self.agent.state = AgentState.READY


class RecoveryTimer(Timer):
    def __init__(self, t, agent):
        super().__init__(t)
        self.agent = agent

    def on_expire(self, env):
        self.agent.state = AgentState.READY


class WeaponCooldownTimer(Timer):
    def __init__(self, t, weapon):
        super().__init__(t)
        self.weapon = weapon
        self.weapon.on_cooldown = True

    def on_expire(self, env):
        print(f"Timer cooldown {self.weapon.on_cooldown}")
        self.weapon.on_cooldown = False


class MoveTimer(Timer):
    def __init__(self, t, timer_q, agent):
        super().__init__(t)
        self.agent = agent

    def on_expire(self, env):
        self.agent.vel = 0


class Weapon:
    def __init__(self, damage=10, shot_speed=0.1, time_to_live=0., windup_time=0., cooldown_time=0.0, recovery_time=0.,
                 shot_width=0.005):
        self.shot_speed = shot_speed
        self.damage = damage
        self.windup_time = windup_time
        self.recovery_time = recovery_time
        self.cooldown_time = cooldown_time
        self.on_cooldown = False
        self.ttl = time_to_live
        self.time_alive = 0.
        self.shot_width = shot_width

    @property
    def range(self):
        return self.shot_speed * self.ttl + self.shot_width / 2

    def shoot(self, t, timer_q, pos, direction, collision_layer):
        shot = Shot(direction, pos, self.shot_speed * direction, self.damage, collision_layer, self.shot_width)
        timer = ShotTimer(t + self.ttl, shot)
        shot.timer = timer
        timer_q.push(timer)
        return shot


def dt_to_collision(pos0, vel0, pos1, vel1):
    relative_velocity = vel0 - vel1
    if relative_velocity == 0:
        return inf
    else:
        return (pos1 - pos0) / relative_velocity


def face_map(map):
    face_map = []
    for base in map:
        face_map += base.faces
    return face_map


class State:
    def __init__(self, base_list=None):
        """

        :param state_config: a dictionary with an entry for each collision layer that specifices a maximum
        amount of bodies
        :param base_list:
        """
        self._state = {}
        self.state_config = {
            CollisionLayer.WALLS: 2,
            CollisionLayer.PLAYER: 1,
            CollisionLayer.ENEMY: 1,
            CollisionLayer.PLAYER_SHOTS: 3,
            CollisionLayer.ENEMY_SHOTS: 3
        }
        if base_list is not None:
            for item in base_list:
                self.append(item)

    def get_sorted_collision_map(self):
        return face_map(sorted(self._state.values(), key=lambda x: x.pos))

    def get_sorted_base_map(self):
        return sorted(self._state.values(), key=lambda x: x.pos)

    @property
    def agents(self):
        return list(filter(lambda x: isinstance(x, Agent), self._state.values()))

    @property
    def statics(self):
        return list(filter(lambda x: isinstance(x, Static), self._state.values()))

    @property
    def dynamics(self):
        return list(filter(lambda x: isinstance(x, Dynamic), self._state.values()))

    @property
    def shots(self):
        return list(filter(lambda x: isinstance(x, Shot), self._state.values()))

    def collision_layer(self, collision_layer):
        return list(filter(lambda x: x.collision_layer == collision_layer, self._state.values()))

    @property
    def marked_for_deletion(self):
        return list(filter(lambda key: self._state[key].delete, self._state.keys()))

    def append(self, item):
        key = next(next_id())
        item.id = key
        self._state[key] = item
        return key

    def remove(self, item):
        del self._state[item.id]

    def items(self):
        return self._state.items()

    def __copy__(self):
        copy_state = State()
        for key, value in self.items():
            print(value)
            copy_state.append(copy.copy(value))
        return copy_state

    def __len__(self):
        return len(self._state)

    def __getitem__(self, key):
        return self._state[key]

    def __delitem__(self, key):
        del self._state[key]

    def __repr__(self):
        return str(self.get_sorted_collision_map())

    def as_numpy(self):

        np_arrays = []
        for layer, length in self.state_config.items():
            size = body_size_registry[layer][0]
            array = np.zeros(length * size)
            bodies = self.collision_layer(layer)
            for i in range(min(length, len(bodies))):
                array[i * size:i*size+size] = bodies[i].as_numpy()
            np_arrays.append(array)

        return np.concatenate(np_arrays).astype(np.float32)


class VelFacePathIter:
    def __init__(self, sorted_base_map, i):
        self.base_map = sorted_base_map
        self.i = i
        self.base = self.base_map[i]

    def __iter__(self):
        return self

    def __next__(self):
        if self.base.vel == 0.:
            raise StopIteration()
        self.i += sign(self.base.vel)
        if 0 <= self.i < len(self.base_map):
            face_idx = 1 if self.base.vel > 0 else 0
            return self.base.faces[face_idx], self.base_map[self.i].faces[1 - face_idx]
        else:
            raise StopIteration()


class VelBasePathIter:
    def __init__(self, sorted_base_map, i):
        self.base_map = sorted_base_map
        self.i = i
        self.base = self.base_map[i]

    def __iter__(self):
        return self

    def __next__(self):
        if self.base.vel == 0.:
            raise StopIteration()
        self.i += sign(self.base.vel)
        if 0 <= self.i < len(self.base_map):
            return self.base_map[self.i]
        else:
            raise StopIteration()


class DirectedVertexBodyIter:
    def __init__(self, sorted_body_map, i, reverse=False, check_zero_vel=False):
        self.body_map = sorted_body_map
        self.body = sorted_body_map[i]
        self.direction = sign(sorted_body_map[i].vel)
        if check_zero_vel and self.direction == 0:
            self.direction = -1 if reverse else 1
        self.face_idx = 0 if self.body.vel > 0 else 1

        self.iterator = []
        if self.direction > 0 ^ reverse:
            for v in reversed(list(range(len(self.body.vertices)))):
                for j in list(range(i + 1, len(sorted_body_map))):
                    self.iterator += [(v, j)]
        else:
            for v in range(len(self.body.vertices)):
                for j in reversed(list(range(0, i))):
                    self.iterator += [(v, j)]
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.direction == 0:
            raise StopIteration()
        if self.i < len(self.iterator):
            v, j = self.iterator[self.i]
            vertex, face = self.body.vertices[v], self.body_map[j].faces[self.face_idx]
            self.i += 1
            return vertex, face
        else:
            raise StopIteration()


class AdjacentPairs:
    def __init__(self, list):
        self.list = list
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.i + 1 < len(self.list):
            x, x_adj = self.list[self.i], self.list[self.i + 1]
            self.i += 1
            return x, x_adj
        else:
            raise StopIteration()


def get_contact_groups(sorted_map):
    contact_sets = []
    adjacent_set = False

    for i in range(len(sorted_map) - 1):
        x = sorted_map[i]
        for j in range(i + 1, len(sorted_map)):
            x_adj = sorted_map[j]
            can_constrain = collision_handler.can_constrain(x, x_adj) or collision_handler.can_constrain(x_adj, x)
            if can_constrain:
                if close(x.faces[1].pos, x_adj.faces[0].pos):
                    if adjacent_set:
                        contact_sets[-1].append(x_adj)
                    else:
                        contact_sets.append([x, x_adj])
                        adjacent_set = True
                else:
                    adjacent_set = False
                break

    return contact_sets


def contains(list, basetype):
    for i in range(len(list)):
        if isinstance(list[i], basetype):
            return i
    return None


def draw(screen, state):
    screen.fill((0, 0, 0))

    for static in state.statics:
        draw_rect(screen, static, 1., static.width, "grey")

    for agent in state.agents:
        if agent.collision_layer == CollisionLayer.PLAYER:
            color = "blue"
        elif agent.collision_layer == CollisionLayer.ENEMY:
            color = "darkorchid"
        else:
            color = "green"

        draw_rect(screen, agent, 0.6 * agent.hp / agent.hp_max, agent.width, color)

    for shot in state.shots:
        if shot.collision_layer == CollisionLayer.PLAYER_SHOTS:
            draw_rect(screen, shot, 0.4, shot.width, "lightgoldenrod1")
        if shot.collision_layer == CollisionLayer.ENEMY_SHOTS:
            draw_rect(screen, shot, 0.4, shot.width, "red")

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



class Env(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
        "render_speed": 4,
    }

    def __init__(self, map, render_mode: Optional[str]=None, state_format="numpy"):

        # make the axis finite by placing walls at each end
        self._map = [Wall(0., Direction.EAST), Wall(1.0, Direction.WEST)] + map
        self.state = State(deepcopy(self._map))
        self.t = 0.
        self.timers = TimerQueue(self)
        self.state_format = state_format
        self._seed = 0
        self.render_mode = render_mode if render_mode is not None else 'human'
        self.render_fps = self.metadata['render_fps']
        self.render_speed = self.metadata['render_speed']

        # check for overlaps
        base_map = self.state.get_sorted_base_map()
        for i in range(len(base_map) - 1):
            if overlap(base_map[i], base_map[i + 1]):
                assert False, "Overlaps detected in map"

        self.observation_space = gym.spaces.Box(-0.1, 1.1, shape=self.state.as_numpy().shape)
        self.action_space = gym.spaces.Discrete(len(Action))

        # for rendering the environment
        self.screen = None
        self.initial_state = None
        self.dt = None


    def get_action_meanings(self):
        return [a.name for a in Action]

    def reset(self, seed=None, options=None):
        self.state = State(deepcopy(self._map))
        self.t = 0.
        self.dt = 0.
        if self.state_format == "numpy":
            return self.state.as_numpy(), {}
        else:
            return self.state, {}

    def step(self, actions):

        done = False
        reward = 0.

        for agent, action in zip(self.state.agents, actions):
            if agent.state == AgentState.READY:
                if action == Action.FORWARD:
                    agent.vel = agent.walk_speed * agent.facing
                elif action == Action.BACKWARD:
                    agent.vel = - agent.walk_speed * agent.facing
                elif action == Action.ATTACK:
                    print(abs(self.state.agents[0].pos - self.state.agents[1].pos) - agent.width)
                    print(agent.weapon.range)
                    if abs(self.state.agents[0].pos - self.state.agents[1].pos) - agent.width <= agent.weapon.range:
                        agent.vel = 0.
                        if agent.weapon is not None:
                            if not agent.weapon.on_cooldown:
                                agent.state = AgentState.WINDUP
                                timer = WindupTimer(self.t + agent.weapon.windup_time, agent.weapon, agent)
                                self.timers.push(timer)
                    else:
                        agent.vel = agent.walk_speed * agent.facing
                elif action == Action.REVERSE_FACING:
                    agent.facing = reverse_facing(agent.facing)

        print(f'TIMERS: {self.timers}')

        dt = inf
        next_timer = None
        # if timers are set get the soonest one as a candidate for next event
        if not self.timers.is_empty():
            dt = self.timers.peek().t - self.t
            print(f'next_timer {self.timers.peek()} dt: {dt}')
            next_timer = dt

        sorted_map = self.state.get_sorted_base_map()

        # stop objects moving through each other using contact constraints
        contact_groups = get_contact_groups(sorted_map)
        for group in contact_groups:
            for i in range(len(group) - 1):
                left_x, right_x = group[i], group[i + 1]
                if sign(left_x.vel) - sign(right_x.vel) == 2:
                    left_x.vel = 0.
                    right_x.vel = 0.

        for group in contact_groups:
            for i, element in enumerate(group):
                vel_dir = sign(element.vel)
                speed = abs(element.vel)
                for next_in_path in VelBasePathIter(group, i):
                    speed = min(speed, abs(next_in_path.vel))
                element.vel = speed * vel_dir

        self.initial_state = copy.copy(self.state)

        # for objects not in contact, compute future collisions and find the nearest in time
        next_collision = None

        for i in range(len(sorted_map)):
            for left_x, right_x in VelFacePathIter(sorted_map, i):
                if collision_handler.can_collide(left_x, right_x) or collision_handler.can_collide(right_x, left_x):
                    if not close(left_x.pos, right_x.pos):
                        dt_adj = dt_to_collision(left_x.pos, left_x.parent.vel, right_x.pos, right_x.parent.vel)
                        if dt_adj > 0 and dt_adj != inf:
                            # print("fc", (dt_adj, dt), left_x, right_x)
                            if dt_adj < dt:
                                next_collision = left_x, right_x
                            dt = min(dt, dt_adj)
                            break

        # vertex system, probably should upgrade the previous collision finder loop to use this instead
        for i in range(len(sorted_map)):
            for vertex, face in DirectedVertexBodyIter(sorted_map, i, check_zero_vel=True):
                if collision_handler.can_collide(vertex, face):
                    if not close(vertex.pos, face.pos):
                        dt_adj = dt_to_collision(vertex.pos, vertex.parent.vel, face.pos, face.parent.vel)
                        if dt_adj > 0 and dt_adj != inf:
                            if dt_adj < dt:
                                next_collision = vertex, face
                            dt = min(dt, dt_adj)
                            break

            for vertex, face in DirectedVertexBodyIter(sorted_map, i, reverse=True, check_zero_vel=True):
                print(vertex, face)
                if collision_handler.can_collide(vertex, face):
                    if not close(vertex.pos, face.pos):
                        dt_adj = dt_to_collision(vertex.pos, vertex.parent.vel, face.pos, face.parent.vel)
                        if dt_adj > 0 and dt_adj != inf:
                            if dt_adj < dt:
                                next_collision = vertex, face
                            dt = min(dt, dt_adj)
                            print(vertex, face, dt)
                            break

        # if dt is inf all the objects are stationary
        # or there are only two objects moving away from each other that will never intersect
        # assuming you have walls at both ends, then we cannot be in the latter, so just return the current state

        truncated = False
        info = {'t': self.t, 'dt': 0, 'initial_state': self.initial_state}

        if dt == inf:
            print(dt, "EVENT: NO COLLISION")
            if self.state_format == "numpy":
                return self.state.as_numpy(), reward, done, truncated, info

            else:
                return self.state, reward, done, truncated, info
        else:
            if dt == next_timer:
                print(dt, "EVENT: TIMER", self.timers.peek())
            else:
                print(dt, "EVENT: COLLISION", next_collision)

        # update positions and move time forward
        for _, left_x in self.state.items():
            left_x.pos += left_x.vel * dt
        self.t += dt
        self.dt = dt

        collision_map = self.state.get_sorted_collision_map()
        collisions = []

        # now fire any timer events
        if not self.timers.is_empty():
            while self.t == self.timers.peek().t:
                self.timers.pop().on_expire(self)
                if self.timers.is_empty():
                    break

        # cancel any move timers
        for timer in self.timers.queue:
            if isinstance(timer, MoveTimer):
                timer.cancel()

        # compute collision events
        for i in range(len(collision_map) - 1):

            # it's possible to have simultaneous collisions
            for j in range(i + 1, len(collision_map)):
                left_x, right_x = collision_map[i], collision_map[j]
                can_collide = collision_handler.can_collide(left_x.parent,
                                                            right_x.parent) or collision_handler.can_collide(
                    right_x.parent, left_x.parent)
                if not left_x.same_parent(right_x) and can_collide:
                    # print('checking', left_x, right_x, close(left_x.pos, right_x.pos))
                    if close(left_x.pos, right_x.pos):
                        collisions.append((left_x, right_x))

                    # check if the next object is at the same position
                    if lookahead(collision_map, j):
                        if lookahead(collision_map, j).pos != right_x.pos:
                            # no more objects at this position, so we are done
                            break

        # run the collisions
        for left_x, x_other in collisions:
            collision_handler.handle_collision(left_x, x_other)
            collision_handler.handle_collision(x_other, left_x)

        # delete stuff marked for deletion
        for key in list(self.state.marked_for_deletion):
            item = self.state[key]
            if isinstance(item, Shot):
                item.timer.cancel()
            del self.state[key]

        # check and resolve overlaps that could be caused by fp numerical errors
        sorted_map = self.state.get_sorted_base_map()
        for group in get_contact_groups(sorted_map):
            contains_wall = contains(group, Wall)
            if contains_wall:
                if contains_wall > 0:
                    for left_x, right_x in AdjacentPairs(list(reversed(group))):
                        right_x.pos = left_x.pos - left_x.width / 2 - right_x.width / 2
            else:
                for left_x, right_x in AdjacentPairs(group):
                    right_x.pos = left_x.pos + left_x.width / 2 + right_x.width / 2

        for i, agent in enumerate(self.state.agents):
            if agent.collision_layer == CollisionLayer.PLAYER:
                if agent.hp <= 0:
                    reward, done = -1., True
                elif agent.hp < self.initial_state.agents[i].hp:
                    reward = -0.01
            elif agent.collision_layer == CollisionLayer.ENEMY:
                if agent.hp <= 0:
                    reward, done = 1., True
                elif agent.hp < self.initial_state.agents[i].hp:
                    reward = 0.01

        info = {'t': self.t, 'dt': dt, 'initial_state': self.initial_state}

        if self.state_format == "numpy":
            return self.state.as_numpy(), reward, done, truncated, info

        else:
            return self.state, reward, done, truncated, info

    def render(self):
        if self.screen is None:
            self.screen = start_screen()
        draw(self.screen, self.state)
        pygame.time.wait(floor(100 / self.render_speed / self.render_fps))

        if self.render_mode == "human":
            if self.initial_state is not None:
                for dt in range(ceil(self.dt * self.render_fps)):
                    for key, item in self.initial_state.items():
                        self.initial_state[key].pos += self.initial_state[key].vel / self.render_fps
                        draw(self.screen, self.initial_state)
                        pygame.time.wait(floor(100/self.render_fps))

        draw(self.screen, self.state)

        if self.render_mode == 'rgb_array':
            return np.swapaxes(pygame.surfarray.array3d(self.screen), 0, 1)

    def seed(self, seed):
        self._seed = seed

