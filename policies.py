import json
import os
import random

from typing import Optional

import tensorflow as tf
from numpy import size
from tensorflow.python.training.tracking.tracking import AutoTrackable
from tf_agents.environments import tf_py_environment
from tf_agents.policies import policy_saver, py_policy, categorical_q_policy
from tf_agents.policies.py_tf_policy import PyTFPolicy
from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.policies.tf_py_policy import TFPyPolicy
from tf_agents.trajectories import time_step as ts, policy_step
from tf_agents.trajectories import TimeStep
from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.typing import types

import numpy as np
from tf_agents.utils import common

from env import MyPuissance4Env


class ZZZPyPolicy(py_policy.PyPolicy):
    def __init__(self, delegate, name='unnamed'):
        # super(PyPolicy, self).__init()
        self.tf_delegate = delegate
        self.name = name

    def __str__(self):
        return f'PyPolicy wrapper for policy "{self.name}"'

    def _action(self, timestep: TimeStep, state=None):
        ps = self.tf_delegate.action(
            TimeStep(step_type=[timestep.step_type], reward=[timestep.reward], discount=[timestep.discount],
                     observation=[timestep.observation]))
        return PolicyStep(ps.action[0].numpy(), ps.state, ps.info)


class ZZZTFPolicy(TFPolicy):
    def __init__(self, delegate, time_step_spec, action_spec, name='unnamed'):
        super(ZZZTFPolicy, self).__init__(time_step_spec=time_step_spec, action_spec=action_spec)
        self.delegate = delegate
        self.label = name

    def __str__(self):
        return f'TFPolicy wrapper for policy "{self.label}"'

    def _action(self, time_step: ts.TimeStep,
                policy_state: types.NestedTensor,
                seed: Optional[types.Seed] = None) -> policy_step.PolicyStep:
        return self.delegate.action(time_step, policy_state)


class PolicyManager:
    def __init__(self, root_dir: str, py_env: MyPuissance4Env, alpha: float = 0.75):
        self.root_dir = root_dir
        self.pyenv = py_env
        self.alpha = alpha
        self.policies = {}
        self.ranks = []
        self.ranks_file = os.path.join(self.root_dir, 'ranks.json')
        self.load_policies()

    def load_policies(self):
        self.policies = {'random': RandomPyPolicy(self.pyenv)}
        self.ranks = ['random']
        if not os.path.isdir(self.root_dir):
            print("Creating policy dir :", self.root_dir)
            os.makedirs(self.root_dir)
        for dir in os.listdir(self.root_dir):
            if os.path.isdir(os.path.join(self.root_dir, dir)):
                self.load_policy(dir)
        self._unpersist_ranks()
        # FIXME should check cross integrity (is every dir mentioned in ranks ? Has every rank an existing, valid dir ?

    def _unpersist_ranks(self):
        if not os.path.isfile(self.ranks_file):
            # file does not exist - no big deal unless we already have more than 0 or 1 policy...
            if len(self.policies) > 1:
                print('WARNING ranks.json is missing - policies will be taken in random order !')
            # update sorted_policies
            self.ranks = [id for id in self.policies.keys()]
            # save it for next time !
            self._persist_ranks()
        else:
            # read ranks.json
            in_file = open(self.ranks_file, 'r')
            self.ranks = json.load(in_file)
            in_file.close()
        # ensure random is listed
        if not 'random' in self.ranks:
            self.ranks.append('random')

    def _persist_ranks(self):
        # save as ranks.json
        out_file = open(self.ranks_file, 'w')
        json.dump(self.ranks, out_file)
        out_file.close()

    def load_policy(self, id: str):
        print(f"Loading policy {id}...")
        saved_model = tf.saved_model.load(os.path.join(self.root_dir, id))
        # tfpol = ZZZTFPolicy(saved_model, time_step_spec=self.pyenv.time_step_spec(), action_spec=self.pyenv.action_spec(), name = id)

        pol = ZZZPyPolicy(saved_model, name=id)
        self.policies[id] = pol
        return pol

    ### registers the given policy and adds it as the first rank
    def register_champion_policy(self, tf_policy: TFPolicy, id: str):
        # generate a unique yet simple id if not provided
        if id is None:
            id = self._generate_next_policy_id(id)

        tf_policy_saver = policy_saver.PolicySaver(tf_policy)
        tf_policy_saver.save(os.path.join(self.root_dir, id))

        self.load_policy(id)

        self.ranks.append(id)

        self._persist_ranks()

        return id

    def _generate_next_policy_id(self, id):
        nb_pols = len(self.policies)
        while id is None or self.policies.keys().__contains__(id):
            nb_pols += 1
            id = f'policy{nb_pols:03d}'
        return id

    def get_champion(self):
        if len(self.policies) < 1:
            raise "Cannot get champion - no policy ais available !"
        id = self.ranks[-1]
        return id, self.policies[id]

    def pick_non_champion(self, nb=1):
        if len(self.ranks) < 2:
            return []
        nb_pols = len(self.ranks)
        ids = random.sample(self.ranks[:-1], min(nb_pols - 1, nb))
        indices = np.random.triangular(0, nb_pols - 1, nb_pols - 1, size=nb)
        loosers = [(id, self.policies[id]) for id in ids]
        return loosers

    def pick_policy(self):
        if len(self.ranks) < 2 or random.random() < self.alpha:
            id, pol = self.get_champion()
        else:
            id, pol = self.pick_non_champion()[0]
        return id, pol


class SinglePlayerPyEnv(MyPuissance4Env):
    def __init__(self, inverse_observations_for_player2=True):
        super(SinglePlayerPyEnv, self).__init__()
        self.inverse_observations_for_player2 = inverse_observations_for_player2
        self._select_policy()

    def __str__(self):
        return f'{type(self).__name__}(using policy {self.current_policy_id})'

    def _select_policy(self):
        raise 'Implement me by extending my class !'
        self.current_policy_id, self.current_policy = "###NOT INITIALIZED###", None
        return self.current_policy_id, self.current_policy

    def _step(self, action):

        # player 1
        # print("Player1 about to play", action)
        time_step = super()._step(action)
        # print("Player1 played", action, "and got", time_step.reward)

        # print(time_step)

        if not time_step.is_last():
            # player 2 !
            # print("Player2 thinking...")

            if self.inverse_observations_for_player2:
                t = self._inverse(time_step)
            else:
                t = time_step

                # print(t)
            action2 = self.current_policy.action(t)

            # print("Player2 about to play", action2)
            time_step = super()._step(action2.action)
            # print("Player2 played", action, "and got", time_step.reward)
            # print("Player2 played", action, "and got ", time_step.reward.numpy()[0])
            if time_step.reward == self.REWARD_BAD_MOVE:
                # print("BAD MOVE from player 2 !????", time_step)
                time_step = ts.termination(time_step.observation, self.REWARD_OTHER_FAILED)
            else:
                # inverse the reward as player 1 will receive player 2 's
                time_step = TimeStep(step_type=time_step.step_type,
                                     reward=-time_step.reward,
                                     discount=time_step.discount,
                                     observation=time_step.observation)
        else:
            # time to change policy !
            self._select_policy()
            # print(f"ENV switching to policy #{self.current_policy_id}")
        # print("last ?", time_step.is_last())
        return time_step


class SinglePlayerMonoPolicyPyEnv(SinglePlayerPyEnv):
    def __init__(self, policy, inverse_observations_for_player2=True):
        self.policy = policy
        super(SinglePlayerMonoPolicyPyEnv, self).__init__(inverse_observations_for_player2)

    def _select_policy(self):
        self.current_policy_id, self.current_policy = str(self.policy), self.policy
        return self.current_policy_id, self.current_policy


### This env will simply delegate player2 decisions to its manager's "champion" policy most of the time.
### The rest of time, it will pick a random policy among those available in the manager.
### "Most of the time" is parametrized by the alpha parameter.
### alpha=0.75 means it will choose the champion 75 times out of 100.
### a triangular random distribution is used so that "stronger policies" are more likely to be selected.
### if inverse_observations_for_player2 is True, observations are inversed (player 1 and 2 are switched)
### so that player2's policy plays as if it was player 1...
class SinglePlayerMultiPolicyPyEnv(SinglePlayerPyEnv):
    def __init__(self, policy_manager_root_dir, alpha=0.75, inverse_observations_for_player2=True):
        self.policy_manager_root_dir = policy_manager_root_dir
        self.policy_manager = None  # will be initialized after self as it takes self as a constructor param.
        self.alpha = alpha
        super(SinglePlayerMultiPolicyPyEnv, self).__init__(inverse_observations_for_player2)

    def _select_policy(self):
        if self.policy_manager is None:
            self.policy_manager = PolicyManager(root_dir=self.policy_manager_root_dir, py_env=self, alpha=self.alpha)

        id, pol = self.policy_manager.pick_policy()
        self.current_policy_id, self.current_policy = id, pol
        return id, pol


### A policy that chooses a random action among the valid ones
class RandomPyPolicy(py_policy.PyPolicy):

    def __init__(self, py_env: MyPuissance4Env):
        super(RandomPyPolicy, self).__init__(py_env.time_step_spec(), py_env.action_spec())

    def __str__(self):
        return f'Random Policy()'

    def _action(self, ts: TimeStep, state: None):
        free_cols = RandomPyPolicy.free_cols(ts)
        random.shuffle(free_cols)
        return PolicyStep(free_cols[0], (), ())

    def free_cols(ts: TimeStep):
        # available cols have a 0 in the upper cell's fourth layer/element
        obs = ts.observation
        while len(obs.shape) > 3:
            obs = obs[0]

        return [col for col, bit in enumerate(obs[-1, :, 3]) if bit == 0]
