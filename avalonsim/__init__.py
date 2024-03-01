from .env import Env, Weapon, Agent, Direction, FaceDirection, RangeFinder, CollisionLayer, Action, Wall
from .env import close, between, overlap
from gym.envs.registration import register, EnvSpec

sword = Weapon(damage=8, shot_speed=2.0, time_to_live=0.03, shot_width=0.01, windup_time=0.05,
               recovery_time=0.04)
bow = Weapon(damage=16, shot_speed=0.3, time_to_live=2., windup_time=0.3)
player = Agent(pos=0.1, facing=Direction.EAST, collision_layer=CollisionLayer.PLAYER,
               shot_collision_layer=CollisionLayer.PLAYER_SHOTS)
player.weapon = sword
player.add_vertex(RangeFinder(player.weapon.range))
player.add_vertex(RangeFinder(-player.weapon.range))
enemy = Agent(pos=0.9, facing=Direction.WEST)
enemy.weapon = bow
enemy.add_vertex(RangeFinder(enemy.weapon.range))
enemy.add_vertex(RangeFinder(-enemy.weapon.range))
map = [player, enemy]

register(
    id='Avalon-v1',
    entry_point='avalonsim.env:Env',
    max_episode_steps=2000,
    reward_threshold=1.,
    kwargs={'map': map}
)