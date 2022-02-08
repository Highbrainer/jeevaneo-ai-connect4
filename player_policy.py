from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.trajectories import TimeStep
from player import Player
class PlayerPolicy:
  def __init__(self, env, player:Player):
    self.env = env
    self.player = player

  def action(self, ts : TimeStep):
    action_id = self.player.findMove(self.env)
    return PolicyStep(action_id, (), ())