from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.trajectories import TimeStep
import random
from bb import BB
from player import Player
class PlayerPolicy:
  def __init__(self, env, player:Player):
    self.env = env
    self.player = player

  def action(self, ts : TimeStep):
    action_id = self.player.findMove(self.env)
    return PolicyStep(action_id, (), ())

class MultiPlayerPolicy:
  def __init__(self, env, players:list[Player]):
    self.env = env
    self.players = players
    self.current_player = 0

  def action(self, ts : TimeStep):
    action_id = self.players[self.current_player].findMove(self.env)
    self.current_player += 1
    self.current_player %= len(self.players)

    return PolicyStep(action_id, (), ())

class PlayerRandomPolicy(PlayerPolicy):

  def action(self, ts : TimeStep):
    bb_current = self.env._compute_current_BB()
    free_cols = PlayerRandomPolicy.free_cols(bb_current)
    random.shuffle(free_cols)
    return PolicyStep(free_cols[0], (), ())

  def free_cols(bb : BB):
    free_cols = []
    for col in range(BB.NB_COLS):
      if bb.get(BB.NB_ROWS-1, col) == 0:
        free_cols.append(col)
    return free_cols