from .env import Env, Weapon, Agent, Direction, FaceDirection, RangeFinder, CollisionLayer, Action, Wall, Items
from .env import close, between, overlap
from gymnasium.envs.registration import register


sword = Weapon(damage=8, shot_speed=1.0, time_to_live=0.01, shot_width=0.01, windup_time=0.05,
               recovery_time=0.04)
bow = Weapon(damage=25, shot_speed=0.3, time_to_live=2., windup_time=1.8)
player = Agent(pos=0.1, facing=Direction.EAST, walk_speed=0.05,
               collision_layer=CollisionLayer.PLAYER,
               shot_collision_layer=CollisionLayer.PLAYER_SHOTS)
player.weapon = sword
player.add_vertex(RangeFinder(player.weapon.range))
player.add_vertex(RangeFinder(-player.weapon.range))
player.inventory += Items.HEALTH_POT
enemy = Agent(pos=0.9, facing=Direction.WEST, walk_speed=0.05)
enemy.weapon = bow
enemy.add_vertex(RangeFinder(enemy.weapon.range))
enemy.add_vertex(RangeFinder(-enemy.weapon.range))
enemy.inventory += Items.HEALTH_POT
map = [player, enemy]


register(
    id='Avalon-v1',
    entry_point='avalonsim.env:Env',
    max_episode_steps=200,
    reward_threshold=1.,
    kwargs={'map': map}
)