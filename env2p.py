from env import REWARD
from env import MyPuissance4Env

from tf_agents.trajectories import time_step as ts

class TwoPlayerPyEnv(MyPuissance4Env):
  def __init__(self, player2_policy = None, inverse_observations_for_player2=False):
        super(TwoPlayerPyEnv, self).__init__()
        self.player2_policy = player2_policy
        self.inverse_observations_for_player2 = inverse_observations_for_player2
  
  def _step(self, action):
    # player 1
    #print("Player1 about to play", action)
    time_step = super()._step(action)
    #print("Player1 played", action)

    #print(time_step)

    if not self.player2_policy is None and not time_step.is_last():
      # player 2 !
      # print("Player2 thinking...")

      if self.inverse_observations_for_player2:
        t = self._inverse(time_step)  
      else :
        t = time_step     

      #print(t)
      action2 = self.player2_policy.action(t)
      
      #print("Player2 about to play", action2)
      time_step = super()._step(action2.action)
      #print("Player2 played", action, "and got ", time_step.reward.numpy()[0])
      if time_step.reward == REWARD.BAD_MOVE:
        return ts.termination(time_step.observation, REWARD.OTHER_FAILED)

    return time_step