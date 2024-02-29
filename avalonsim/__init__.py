from .env import Env, Weapon, Agent, Direction, FaceDirection, RangeFinder, CollisionLayer, Action, Wall
from .env import close, between, overlap
from gym.envs.registration import register, EnvSpec

sword = Weapon(damage=10, shot_speed=0.5, time_to_live=0.04, cooldown_time=0.1, shot_width=0.01,
               windup_time=0.1,
               recovery_time=0.04)
bow = Weapon(damage=3, shot_speed=0.5, time_to_live=1., cooldown_time=0.3, windup_time=0.3)
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
    max_episode_steps=10000,
    reward_threshold=1.,
    kwargs={'map': map}
)