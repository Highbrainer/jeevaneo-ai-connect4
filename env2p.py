from env import REWARD
from env import MyPuissance4Env

from tf_agents.trajectories import time_step as ts

class TwoPlayerPyEnv(MyPuissance4Env):
  def __init__(self, player2_policy = None, player2_policies=[], inverse_observations_for_player2=False):
        super(TwoPlayerPyEnv, self).__init__()
        self.player2_policies = player2_policies
        if not player2_policy is None:
          self.player2_policies.append(player2_policy)
        self.inverse_observations_for_player2 = inverse_observations_for_player2
        self.current_policy=0

  def _step(self, action):
      
    # player 1
    #print("Player1 about to play", action)
    time_step = super()._step(action)
    #print("Player1 played", action)

    #print(time_step)

    if len(self.player2_policies)>0 and not time_step.is_last():
        # player 2 !
        # print("Player2 thinking...")

        if self.inverse_observations_for_player2:
          t = self._inverse(time_step)  
        else :
          t = time_step     

        #print(t)
        action2 = self.player2_policies[self.current_policy].action(t)
        
        #print("Player2 about to play", action2)
        time_step = super()._step(action2.action)
        #print("Player2 played", action, "and got ", time_step.reward.numpy()[0])
        if time_step.reward == REWARD.BAD_MOVE:
          print("BAD MOVE from player 2 !????", time_step)
          time_step = ts.termination(time_step.observation, REWARD.OTHER_FAILED)
    

    if time_step.is_last() :
      # time to change policy !
      nb_policies = len(self.player2_policies)
      if nb_policies >0 :
        self.current_policy += 1
        self.current_policy %= nb_policies
        #print(f"ENV switching to policy #{self.current_policy}")

    return time_step