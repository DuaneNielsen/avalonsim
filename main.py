from math import inf
import uuid
import random
from enum import Enum, IntEnum
from collections import deque

CL_WALLS = "walls"
CL_PLAYER = "team_player"
CL_ENEMY = "team_enemy"
CL_PLAYER_SHOTS = "player_shots"
CL_ENEMY_SHOTS = "enemy_shots"


class Direction(IntEnum):
    EAST = 1
    WEST = -1


class FaceDirection(IntEnum):
    FRONT = 0
    BACK = 1


def reverse_facing(facing):
    return Direction.EAST if facing == facing.WEST else Direction.WEST


class Action(IntEnum):
    PASS = 0
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
    if i+1 < len(list):
        return list[i+1]
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


class Base:
    def __init__(self):
        self.id = None
        self.pos = 0
        self.vel = 0
        self.width = 0.001
        self.facing = Direction.EAST
        self.delete = False
        self.collision_layer = ""

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

    def direction(self, other):
        return Direction.WEST if other.pos < self.pos else Direction.EAST

    def moving_in_direction_of(self, other):
        return sign(self.pos - other.pos) * self.vel < 0.

    def moving_same_direction(self, other):
        return sign(self.vel) == sign(other.vel)

    def __repr__(self):
        return f"{self.__class__.__name__} {self.id} face: {self.facing.name} pos: {self.pos} vel: {self.vel}"


class Static(Base):
    def __init__(self):
        super().__init__()


class Dynamic(Base):
    def __init__(self):
        super().__init__()


class Wall(Static):
    def __init__(self, pos, facing):
        super().__init__()
        self.pos = pos
        self.facing = facing
        self.collision_layer = CL_WALLS
        self.width = 0.001

    def __repr__(self):
        return str(self.__class__) + " face:" + str(self.facing) + " pos: " + str(self.pos)


class Agent(Dynamic):
    def __init__(self, pos=0., facing=Direction.WEST, walk_speed=0.1, hp_max=100, collision_layer=CL_ENEMY,
                 shot_collision_layer=CL_ENEMY_SHOTS):
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


class Shot(Dynamic):
    def __init__(self, facing, pos, vel, damage, collision_layer, expiry_time):
        super().__init__()
        self.facing = facing
        self.pos = pos
        self.vel = vel
        self.damage = damage
        self.collision_layer = collision_layer
        self.width = 0.001
        self.timer = ShotTimer(expiry_time, self)


class RangeFinder(Dynamic):
    def __init__(self):
        super().__init__()
        self.width = 0.001


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

collision_handler = CollisionHandler()
collision_handler.add_handler(CL_PLAYER, CL_ENEMY, dynamic_stop, can_constrain=True)
collision_handler.add_handler(CL_ENEMY, CL_PLAYER, dynamic_stop, can_constrain=True)
collision_handler.add_handler(CL_WALLS, CL_PLAYER, static_stop, can_constrain=True)
collision_handler.add_handler(CL_WALLS, CL_ENEMY, static_stop, can_constrain=True)
collision_handler.add_handler(CL_PLAYER_SHOTS, CL_WALLS, delete_self)
collision_handler.add_handler(CL_ENEMY_SHOTS, CL_WALLS, delete_self)
collision_handler.add_handler(CL_ENEMY_SHOTS, CL_PLAYER, apply_damage_and_delete)
collision_handler.add_handler(CL_PLAYER_SHOTS, CL_ENEMY, apply_damage_and_delete)


class TimerQueue:
    def __init__(self):
        self.queue = []

    def push(self, timer):
        self.queue.append(timer)
        self.queue.sort(key=lambda x: x.t)

    def pop(self):
        if self.is_empty():
            raise IndexError("pop from an empty priority queue")
        return self.queue.pop(0)  # Pop and return the item with the highest priority

    def peek(self):
        if self.is_empty():
            return None
        return self.queue[0]  # Return the item with the highest priority

    def is_empty(self):
        return len(self.queue) == 0

    def cancel(self, id):
        for i, timer in enumerate(self.queue):
            if timer.id == id:
                del self.queue[i]

    def __repr__(self):
        return str(self.queue)


timers = TimerQueue()


class Timer:
    def __init__(self, t):
        """
        :param t: the absolute time at which the timer will expire

        To use just create the object, it will automatically be added to the timer queue
        """
        self.t = t
        self.id = next(next_id())
        timers.push(self)

    def on_expire(self):
        """
        called if timer expires
        """
        pass

    def cancel(self):
        """
        cancels this timer
        """
        timers.cancel(self.id)

    def __repr__(self):
        return f'{self.__class__.__name__}(t={self.t}, id={self.id})'


class ShotTimer(Timer):
    def __init__(self, t, shot):
        super().__init__(t)
        self.shot = shot

    def on_expire(self):
        self.shot.delete = True


class WeaponCooldownTimer(Timer):
    def __init__(self, t, weapon):
        super().__init__(t)
        self.weapon = weapon
        self.weapon.on_cooldown = True

    def on_expire(self):
        print(f"Timer cooldown {self.weapon.on_cooldown}")
        self.weapon.on_cooldown = False


class Weapon:
    def __init__(self, damage=10, shot_speed=0.1, time_to_live=0., windup_time=0., cooldown_time=0.0, action_blocking=False):
        self.shot_speed = shot_speed
        self.damage = damage
        self.windup_time = windup_time
        self.cooldown_time = cooldown_time
        self.on_cooldown = False
        self.action_blocking = action_blocking
        self.ttl = time_to_live
        self.time_alive = 0.

    def shoot(self, t, pos, direction, collision_layer):
        WeaponCooldownTimer(t + self.cooldown_time, self)
        return Shot(direction, pos, self.shot_speed * direction, self.damage, collision_layer, t + self.ttl)


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
        self._state = {}
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

    def __len__(self):
        return len(self._state)

    def __getitem__(self, key):
        return self._state[key]

    def __delitem__(self, key):
        del self._state[key]

    def __repr__(self):
        return str(self.get_sorted_collision_map())


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
            return self.base.faces[face_idx], self.base_map[self.i].faces[1-face_idx]
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


class AdjacentPairs:
    def __init__(self, list):
        self.list = list
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.i + 1 < len(self.list):
            x, x_adj = self.list[self.i], self.list[self.i+1]
            self.i += 1
            return x, x_adj
        else:
            raise StopIteration()


def get_contact_groups(sorted_map):
    contact_sets = []
    adjacent_set = False

    for i in range(len(sorted_map) - 1):
        x = sorted_map[i]
        for j in range(i+1, len(sorted_map)):
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


class Env:
    def __init__(self, map):

        # make the axis finite by placing walls at each end
        self._map = [Wall(0., Direction.EAST), Wall(1.0, Direction.WEST)] + map
        self.state = State(self._map)
        self.t = 0.

        # check for overlaps
        base_map = self.state.get_sorted_base_map()
        for i in range(len(base_map) - 1):
            if overlap(base_map[i], base_map[i + 1]):
                assert False, "Overlaps detected in map"

    def reset(self):
        self.state = State(self._map)
        self.t = 0.
        return self.state

    def step(self, actions, render=False):

        for agent, action in zip(self.state.agents, actions):
            if action == Action.FORWARD:
                agent.vel = agent.walk_speed * agent.facing
            elif action == Action.BACKWARD:
                agent.vel = - agent.walk_speed * agent.facing
            elif action == Action.ATTACK:
                agent.vel = 0.
                if agent.weapon is not None:
                    if not agent.weapon.on_cooldown:
                        shot = agent.weapon.shoot(self.t, agent.pos, agent.facing, agent.shot_collision_layer)
                        self.state.append(shot)
            elif action == Action.REVERSE_FACING:
                agent.facing = reverse_facing(agent.facing)

        print(f'TIMERS: {timers}')

        initial_state = None
        if render:
            initial_state = deepcopy(self.state)

        dt = inf
        next_timer = None
        # if timers are set get the soonest one as a candidate for next event
        if not timers.is_empty():
            dt = timers.peek().t - self.t
            print(f'next_timer {timers.peek()} dt: {dt}')
            next_timer = dt

        sorted_map = self.state.get_sorted_base_map()

        # stop objects moving through each other using contact constraints
        contact_groups = get_contact_groups(sorted_map)
        for group in contact_groups:
            for i in range(len(group)-1):
                left_x, right_x = group[i], group[i+1]
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

        # if dt is inf all the objects are stationary
        # or there are only two objects moving away from each other that will never intersect
        # assuming you have walls at both ends, then we cannot be in the latter, so just return the current state

        if dt == inf:
            print(dt, "EVENT: NO COLLISION")
            return self.state, 0., False, {'t': self.t, 'dt': 0, 'initial_state': initial_state}
        else:
            if dt == next_timer:
                print(dt, "EVENT: TIMER", timers.peek())
            else:
                print(dt, "EVENT: COLLISION", next_collision)

        # update positions and move time forward
        for _, left_x in self.state.items():
            left_x.pos += left_x.vel * dt
        self.t += dt

        collision_map = self.state.get_sorted_collision_map()
        collisions = []

        # now fire any timer events
        if not timers.is_empty():
            while self.t == timers.peek().t:
                timers.pop().on_expire()
                if timers.is_empty():
                    break

        # compute collision events
        for i in range(len(collision_map) - 1):

            # it's possible to have simultaneous collisions
            for j in range(i + 1, len(collision_map)):
                left_x, right_x = collision_map[i], collision_map[j]
                can_collide = collision_handler.can_collide(left_x.parent, right_x.parent) or collision_handler.can_collide(right_x.parent, left_x.parent)
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

        return self.state, 0., False, {'t': self.t, 'dt': dt, 'initial_state': initial_state}


if __name__ == "__main__":

    sword = Weapon(damage=10, shot_speed=10, time_to_live=0.05, cooldown_time=0.1)
    bow = Weapon(damage=3, shot_speed=0.7, time_to_live=1, cooldown_time=0.3)
    player = Agent(pos=0.1, facing=Direction.EAST, collision_layer=CL_PLAYER, shot_collision_layer=CL_PLAYER_SHOTS)
    player.weapon = bow
    enemy = Agent(pos=0.9, facing=Direction.WEST)
    enemy.weapon = sword
    map = [player, enemy]
    env = Env(map)

    import pygame
    from math import floor, ceil
    from copy import deepcopy

    pygame.init()
    screen_width, screen_height = 600, 400
    screen_border_width, screen_border_height = 50, 50

    screen = pygame.display.set_mode((screen_width, screen_height))
    fps = 50
    speed = 2


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


    def draw_rect(base, height, color):
        y, width = 0.0, 0.05
        x, y, width, height = to_screen(base.pos, y, width, height)
        x -= width / 2
        color = pygame.Color(color)
        bar = pygame.Rect(x, y, width, height)
        pygame.draw.rect(screen, color, bar)


    def draw(state):

        screen.fill((0, 0, 0))

        for static in state.statics:
            draw_rect(static, 1., "grey")

        for agent in state.agents:
            if agent.collision_layer == CL_PLAYER:
                draw_rect(agent, 0.6, "blue")
            if agent.collision_layer == CL_ENEMY:
                draw_rect(agent, 0.6, "darkorchid")

        for shot in state.shots:
            if shot.collision_layer == CL_PLAYER_SHOTS:
                draw_rect(shot, 0.4, "lightgoldenrod1")
            if shot.collision_layer == CL_ENEMY_SHOTS:
                draw_rect(shot, 0.4, "red")

        for i, agent in enumerate(state.agents):
            if agent.weapon:
                if agent.weapon.on_cooldown:
                    color = pygame.Color("blue")
                else:
                    color = pygame.Color("lightblue")
                x = 0.4 + i * 0.2
                x, y, width, height = to_screen(x, 0.9, 0.05, 0.05)
                bar = pygame.Rect(x, y, width, height)
                pygame.draw.rect(screen, color, bar)

        pygame.display.update()
        pygame.time.wait(floor(100/speed/fps))


    running = True

    state = env.reset()
    print(state)

    draw(state)
    random.seed(42)

    trajectory = []

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
                print(state)

                for dt in range(ceil(info['dt'] * fps)):
                    for key, item in info['initial_state'].items():
                        info['initial_state'][key].pos += info['initial_state'][key].vel / fps
                        draw(info['initial_state'])
                draw(state)

    pygame.quit()
