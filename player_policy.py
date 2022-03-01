from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.trajectories import TimeStep
import random
from bb import BB
from player import Player
import numpy as np
class PlayerPolicy:
  def __init__(self, player:Player):
    self.player = player

  def __str__(self):
      return f'Policy for {str(self.player)}'

  def action(self, ts : TimeStep):
    action_id = self.player.findMove(ts)
    return PolicyStep(action_id, (), ())

class PlayerRandomPolicy():

  def action(self, ts : TimeStep):
    free_cols = PlayerRandomPolicy.free_cols(ts)
    random.shuffle(free_cols)
    return PolicyStep(free_cols[0], (), ())

  def free_cols(ts:TimeStep):
    # convert observation to a couple of BBs
    obs = ts.observation
    while len(obs.shape) > 3:
        obs = obs[0]

    return np.where(obs[BB.NB_ROWS-1,:,3]==0)[0]