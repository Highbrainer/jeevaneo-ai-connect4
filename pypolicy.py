﻿from tf_agents.policies.tf_policy import TFPolicy
class PyPolicy:
  def __init__(self, tfpolicy:TFPolicy):
    self.tf_policy=tfpolicy
      
  def action(self, timestep : TimeStep):
    ps = self.tf_policy.action(TimeStep(step_type=[timestep.step_type], reward=[timestep.reward], discount=[timestep.discount], observation=[timestep.observation]))
    return PolicyStep(ps.action[0], ps.state, ps.info)