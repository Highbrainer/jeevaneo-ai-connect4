﻿from numpy.testing._private.utils import assert_equal
import unittest

from env import REWARD
from env import MyPuissance4Env

from env2p import TwoPlayerPyEnv

import numpy as np

import tensorflow as tf
import tf_agents
from tf_agents.environments import tf_py_environment
from tf_agents.trajectories import StepType
from tf_agents.trajectories.policy_step import PolicyStep

NB_ROWS=6
NB_COLS=7

def step_name(timestep):
  if timestep.step_type == StepType.FIRST:
    return "FIRST"
  if timestep.step_type == StepType.MID:
    return "MID"
  if timestep.step_type == StepType.LAST:
    return "LAST"
  return "???!!!???"
class MyPuissanc4Test(unittest.TestCase):
  def test_reset(self):
    py_env = TwoPlayerPyEnv()
    env = tf_py_environment.TFPyEnvironment(py_env)
    timestep = env.reset()
    self.assertEqual(timestep.step_type.numpy(), StepType.FIRST)
    self.assertEqual(timestep.reward.shape, (1))
    self.assertEqual(timestep.reward, 0)
    self.assertEqual(timestep.discount.shape, (1))
    self.assertEqual(timestep.discount, 1)
    self.assertEqual(timestep.observation.shape, (1,NB_ROWS,NB_COLS,1))
    self.assertEqual(timestep.observation.numpy().min(), 0)
    self.assertEqual(timestep.observation.numpy().max(), 0)

    timestep = env.step(0)
    self.assertEqual(timestep.step_type.numpy(), StepType.MID)
    timestep = env.step(1)
    self.assertEqual(timestep.observation.numpy().max(), 1.0)
    self.assertEqual(timestep.step_type.numpy(), StepType.MID)
    timestep = env.reset()
    self.assertEqual(timestep.step_type.numpy(), StepType.FIRST)
    self.assertEqual(timestep.reward, 0)
    self.assertEqual(timestep.discount, 1)
    self.assertEqual(timestep.observation.shape, (1,NB_ROWS,NB_COLS,1))
    self.assertEqual(timestep.observation.numpy().min(), 0)
    self.assertEqual(timestep.observation.numpy().max(), 0)

  def test_player2_fails(self):

    class FailingPolicy():      
      def action(self, ts):
        return PolicyStep(action=0)

    py_env = TwoPlayerPyEnv(player2_policy=FailingPolicy())
    env = tf_py_environment.TFPyEnvironment(py_env)
    timestep = env.reset()
    timestep = env.step(0)
    timestep = env.step(0)
    timestep = env.step(0)
    timestep = env.step(1)
    self.assertEqual(timestep.is_last(), True)



  def test_step_reward(self):
    env = tf_py_environment.TFPyEnvironment(TwoPlayerPyEnv())
    timestep = env.reset()
    timestep = env.step(0)
    self.assertEqual(timestep.reward, REWARD.GOOD_MOVE)
    timestep = env.step(0)
    self.assertEqual(timestep.reward, REWARD.GOOD_MOVE)
    timestep = env.step(0)
    self.assertEqual(timestep.reward, REWARD.GOOD_MOVE*2)
    timestep = env.step(0)
    self.assertEqual(timestep.reward, REWARD.GOOD_MOVE*2)
    timestep = env.step(0)
    self.assertEqual(timestep.reward, REWARD.GOOD_MOVE*3)

  def test_step_simple(self):
    py_env = TwoPlayerPyEnv()
    env = tf_py_environment.TFPyEnvironment(py_env)
    timestep = env.reset()
    self.assertEqual(timestep.step_type, StepType.FIRST)
    timestep = env.step(0)
    self.assertEqual(timestep.reward, REWARD.GOOD_MOVE)
    self.assertEqual(timestep.discount, np.array(0.95, dtype=np.float32))
    self.assertEqual(timestep.observation.shape, (1, NB_ROWS,NB_COLS,1))
    self.assertEqual(timestep.observation.numpy().min(), 0)
    self.assertEqual(timestep.observation.numpy().max(), 0.5)
    self.assertEqual(timestep.observation.numpy()[0][0][0], 0.5)
    self.assertEqual(timestep.step_type, StepType.MID)
    
    timestep = env.step(0)
    self.assertEqual(timestep.reward, REWARD.GOOD_MOVE)
    self.assertEqual(timestep.discount, np.array(0.95, dtype=np.float32))
    self.assertEqual(timestep.observation.shape, (1, NB_ROWS,NB_COLS,1))
    self.assertEqual(timestep.observation.numpy().min(), 0)
    self.assertEqual(timestep.observation.numpy().max(), 1.0)
    self.assertEqual(timestep.observation.numpy()[0][0][0], 0.5)
    self.assertEqual(timestep.observation[0][1][0][0], 1)
    self.assertEqual(timestep.observation[0][0][0+NB_COLS-1], 0)
    self.assertEqual(timestep.step_type, StepType.MID)

    for _ in range(4):
      timestep = env.step(0)
      self.assertEqual(timestep.reward, REWARD.GOOD_MOVE*((_//2)+2))
      self.assertEqual(timestep.step_type, StepType.MID)

    # BAD MOVE !
    timestep = env.step(0)
    self.assertEqual(timestep.reward, REWARD.BAD_MOVE)
    self.assertEqual(timestep.step_type, StepType.LAST)

    # FULL !
    timestep = env.reset()
    nb = 0
    for row in range(NB_ROWS):
      for col in range(NB_COLS):    
        actid = (col + row//3)%NB_COLS
        if nb > 20:
          actid = NB_COLS - actid -1 
        if nb > 26 and nb<34:
          actid %= NB_COLS-2
        if nb == 39:
          actid = 6
        if nb == 40:
          actid = 5
        if nb == 41:
          actid = 6
        timestep = env.step(actid)
        nb += 1

    self.assertEqual(timestep.step_type, StepType.LAST)
    self.assertEqual(timestep.reward, REWARD.DRAW)

    # WIN TODO
    # 2-2-2
    # 1-1-1-1
    env.reset()
    for _ in range(3):
      timestep = env.step(_)
      timestep = env.step(_)
      self.assertEqual(timestep.reward, REWARD.GOOD_MOVE*(1+_))
      self.assertEqual(timestep.step_type, StepType.MID)
    
    timestep = env.step(3)
    self.assertEqual(timestep.reward, REWARD.WIN)
    self.assertEqual(timestep.step_type, StepType.LAST)

    # LOSE TODO
    # 1
    # 1
    # 2-2-2-2
    # 1 1 1 2
    env.reset()
    for _ in range(3):
      timestep = env.step(_)
      timestep = env.step(_)
      self.assertEqual(timestep.step_type, StepType.MID)
       
    timestep = env.step(0)
    self.assertEqual(timestep.reward, REWARD.GOOD_MOVE*(3+1))
    self.assertEqual(timestep.step_type, StepType.MID)
    timestep = env.step(3)
    self.assertEqual(timestep.reward, REWARD.GOOD_MOVE*(3+1))
    self.assertEqual(timestep.step_type, StepType.MID)
    timestep = env.step(0)
    self.assertEqual(timestep.reward, REWARD.GOOD_MOVE*(3+2))
    self.assertEqual(timestep.step_type, StepType.MID)
  
    timestep = env.step(3)
    self.assertEqual(timestep.reward, REWARD.LOST)
    self.assertEqual(timestep.step_type, StepType.LAST)

  def test_inverse_py(self):
    env= MyPuissance4Env()
    timestep=env.reset()
    timestep=env.step(0)
    timestep=env.step(0)
    timestep=env.step(1)
    timestep=env.step(1)
    inversed = env._inverse(timestep)

    self.assertEqual(timestep.observation[0][0][0], 1)
    self.assertEqual(timestep.observation[0][1][0], 1)
    self.assertEqual(timestep.observation[1][0][0], 0.5)
    self.assertEqual(timestep.observation[1][1][0], 0.5)
    for col in range(0,7):
      for row in range(0, 6):
        if not (col, row) in [(0,0), (0,1), (1,0), (1,1)]:
          self.assertEqual(timestep.observation[row][col][0], 0.0)
    
    self.assertEqual(inversed.observation[0][0][0], 1)
    self.assertEqual(inversed.observation[0][1][0], 1)
    self.assertEqual(inversed.observation[1][0][0], 0.5)
    self.assertEqual(inversed.observation[1][1][0], 0.5)
    for col in range(0,7):
      for row in range(0, 6):
        if not (col, row) in [(0,0), (0,1), (1,0), (1,1)]:
          self.assertEqual(inversed.observation[row][col][0], 0.0)

unittest.main(argv=['bidon'], exit=False)