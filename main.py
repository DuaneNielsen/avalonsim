from math import inf
import uuid
import random
from enum import Enum, IntEnum

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


class Face:
    def __init__(self, parent, pos, side):
        self.parent = parent
        self.pos = pos
        self.side = side
        self.collision_layer = parent.collision_layer

    def same_parent(self, face):
        return self.parent.id == face.parent.id

    def __repr__(self):
        return str(self.parent.__class__) + " " + str(self.side) + " pos: " + str(self.pos) + " vel: " + str(self.parent.vel)


def close(a, b, tol=1e-7):
    return abs(a - b) < tol


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
        return str(self.__class__) + " face:" + str(self.facing) + " pos: " + str(self.pos) + " vel: " + str(self.vel)


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
    def __init__(self, facing, pos, vel, damage, collision_layer):
        super().__init__()
        self.facing = facing
        self.pos = pos
        self.vel = vel
        self.damage = damage
        self.collision_layer = collision_layer
        self.width = 0.001


class RangeFinder(Dynamic):
    def __init__(self):
        super().__init__()
        self.width = 0.001


class CollisionHandler:
    def __init__(self):
        self.handlers = {}

    def add_handler(self, layer_name1, layer_name2, callback):
        self.handlers[(layer_name1, layer_name2)] = callback

    def can_collide(self, obj1, obj2):
        key = obj1.collision_layer, obj2.collision_layer
        if key in self.handlers:
            return self.handlers[key].can_collide(obj1, obj2)
        else:
            return False

    def apply_contact_constraint(self, obj1, obj2):
        layer_name1 = obj1.collision_layer
        layer_name2 = obj2.collision_layer
        handler = self.handlers.get((layer_name1, layer_name2))
        if handler:
            handler.contact_constraint(obj1, obj2)

    def handle_collision(self, obj1, obj2):
        layer_name1 = obj1.collision_layer
        layer_name2 = obj2.collision_layer
        handler = self.handlers.get((layer_name1, layer_name2))
        if handler:
            handler.effect(obj1, obj2)


class Collision:
    def can_collide(self, face1, face2):
        return True

    # effects are applied to objects the moment they collide
    def effect(self, object, other):
        pass

    # contact constraints are applied to action when objects are touching
    def contact_constraint(self, object, other):
        pass


class StaticCollision(Collision):

    def effect(self, static_object, other):
        other.parent.vel = 0

    def contact_constraint(self, static_object, other):
        if other.parent.moving_in_direction_of(static_object.parent):
            other.parent.vel = 0


static_stop = StaticCollision()


class DynamicCollision(Collision):
    def effect(self, face, target_frace):
        if face.parent.moving_same_direction(target_frace.parent):
            target_frace.parent.vel = min(face.parent.vel, target_frace.parent.vel)
        else:
            target_frace.parent.vel = 0

    def contact_constraint(self, face, target_face):
        if face.parent.moving_same_direction(target_face.parent):
            face.parent.vel = min(face.parent.vel, target_face.parent.vel)
        elif face.parent.moving_in_direction_of(target_face):
            face.parent.vel = 0


dynamic_stop = DynamicCollision()


class ApplyAndDeleteShot(Collision):

    def effect(self, face, target_face):
        target_face.parent.hp -= face.parent.damage
        face.parent.vel = 0.
        face.parent.delete = True


apply_damage_and_delete = ApplyAndDeleteShot()


class DeleteSelf(Collision):
    def effect(self, face, target_face):
        face.parent.vel = 0.
        face.parent.delete = True


delete_self = DeleteSelf()


collision_handler = CollisionHandler()
collision_handler.add_handler(CL_PLAYER, CL_ENEMY, dynamic_stop)
collision_handler.add_handler(CL_ENEMY, CL_PLAYER, dynamic_stop)
collision_handler.add_handler(CL_WALLS, CL_PLAYER, static_stop)
collision_handler.add_handler(CL_WALLS, CL_ENEMY, static_stop)
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


class Timer:
    def __init__(self):
        self.t = 0.
        self.id = next(next_id())

    def on_expire(self):
        pass

    def __repr__(self):
        return f'Timer({self.t}, {self})'


class ShotTimer(Timer):
    def __init__(self, t, shot):
        super().__init__()
        self.t = t
        self.shot = shot

    def on_expire(self):
        self.shot.delete = True


class Weapon:
    def __init__(self, damage=10, shot_speed=0.1, time_to_live=0., windup_time=0., action_blocking=False):
        self.shot_speed = shot_speed
        self.damage = damage
        self.windup_time = windup_time
        self.action_blocking = action_blocking
        self.ttl = time_to_live
        self.time_alive = 0.


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
        return sorted(face_map(self._state.values()), key=lambda x: x.pos)

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


class Env:
    def __init__(self, map):

        # make the axis finite by placing walls at each end
        self._map = [Wall(0., Direction.EAST), Wall(1.0, Direction.WEST)] + map
        self.state = State(map)
        self.timers = TimerQueue()

        self.t = 0.

    def reset(self):
        self.state = State(self._map)
        self.t = 0.
        return self.state

    def step(self, actions):

        for agent, action in zip(self.state.agents, actions):
            if action == Action.FORWARD:
                agent.vel = agent.walk_speed * agent.facing
            elif action == Action.BACKWARD:
                agent.vel = - agent.walk_speed * agent.facing
            elif action == Action.ATTACK:
                agent.vel = 0.
                if agent.weapon is not None:
                    shot = Shot(agent.facing, agent.pos, agent.weapon.shot_speed * agent.facing, agent.weapon.damage, agent.shot_collision_layer)
                    self.timers.push(ShotTimer(self.t + agent.weapon.ttl, shot))
                    self.state.append(shot)
            elif action == Action.REVERSE_FACING:
                agent.facing = reverse_facing(agent.facing)

        dt = inf

        # if timers are set get the soonest one as a candidate for next event
        if not self.timers.is_empty():
            dt = self.timers.peek().t - self.t

        collision_map = self.state.get_sorted_collision_map()

        # check adjacent objects for future collisions and find the next one
        for i in range(len(collision_map) - 1):
            x, x_adj_pos = collision_map[i], collision_map[i + 1].pos

            # keep checking until you have a collision
            for j in range(i + 1, len(collision_map)):
                x_adj = collision_map[j]

                # check if this pair can interact with each other
                if not x.same_parent(x_adj):
                    if collision_handler.can_collide(x, x_adj) or collision_handler.can_collide(x_adj, x):

                        # if objects are in contact already apply contact constraints
                        if close(x.pos, x_adj.pos):
                            collision_handler.apply_contact_constraint(x, x_adj)
                            collision_handler.apply_contact_constraint(x_adj, x)
                        else:  # they are not in contact, so compute time to collide
                            dt_adj = dt_to_collision(x.pos, x.parent.vel, x_adj.pos, x_adj.parent.vel)
                            if dt_adj > 0 and dt_adj != inf:
                                print("collision", x, x_adj, dt_adj)
                                dt = min(dt, dt_adj)
                                break

        # if dt is inf all the objects are stationary
        # or there are only two objects moving away from each other that will never intersect
        # assuming you have walls at both ends, then we cannot be in the latter, so just return the current state
        if dt == inf:
            return self.state, 0., False, {}

        # update positions and move time forward
        for _, x in self.state.items():
            x.pos += x.vel * dt
        self.t += dt

        collision_map = self.state.get_sorted_collision_map()
        collisions = []

        # now fire any timer events
        if not self.timers.is_empty():
            while dt + self.t == self.timers.peek().t:
                self.timers.pop().on_expire()
                if self.timers.is_empty():
                    break

        # compute collision events
        for i in range(len(collision_map) - 1):

            # it's possible to have simultaneous collisions
            for j in range(i + 1, len(collision_map)):
                x, x_adj = collision_map[i], collision_map[j]
                if close(x.pos, x_adj.pos):
                    if not x.same_parent(x_adj):
                        collisions.append((x, x_adj))
                else:
                    # no more objects at this position
                    break

        # run the collisions
        for x, x_other in collisions:
            collision_handler.handle_collision(x, x_other)
            collision_handler.handle_collision(x_other, x)

        # delete stuff marked for deletion
        for key in list(self.state.marked_for_deletion):
            del self.state[key]

        return self.state, 0., False, {}


if __name__ == "__main__":

    sword = Weapon(damage=10, shot_speed=100, time_to_live=0.01, action_blocking=True)
    bow = Weapon(damage=3, shot_speed=0.3, time_to_live=0.5)
    player = Agent(facing=Direction.EAST, collision_layer=CL_PLAYER, shot_collision_layer=CL_PLAYER_SHOTS)
    player.weapon = sword
    enemy = Agent(pos=1.)
    enemy.weapon = sword
    map = [Wall(0, Direction.EAST), Wall(1., Direction.WEST), player, enemy]
    env = Env(map)

    import pygame

    pygame.init()
    screen_width, screen_height = 600, 400
    screen_border_width, screen_border_height = 50, 50

    screen = pygame.display.set_mode((screen_width, screen_height))


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
            draw_rect(static, 1., "green")

        for agent in state.agents:
            draw_rect(agent, 0.6, "blue")

        for shot in state.shots:
            draw_rect(shot, 0.4, "red")

        pygame.display.update()
        pygame.time.wait(200)


    running = True

    state = env.reset()
    draw(state)
    random.seed(42)

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
                else:
                    actions = [Action.PASS, enemy_action]

            if event.type == pygame.QUIT:
                running = False

            if len(actions) == 2:
                print(actions)
                state, reward, done, info = env.step(actions)
                print(state)
                draw(state)

    pygame.quit()
