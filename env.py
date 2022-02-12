from collections.abc import Iterable
from functools import lru_cache
import numpy as np
from pyglet_utils import Label
from bb import BB

#import tensorflow as tf
import tf_agents
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import TimeStep

from gym.envs.classic_control import rendering
import pyglet.gl

DISCOUNT = 0.95

EMPTY = np.float32(0.0)
PLAYER1 = np.float32(0.5)
PLAYER2 = np.float32(1.0)

# Colors
BLUE = (0, 0.1, 0.8)
BLACK = (0, 0, 0)
WHITE = (1, 1, 1)
GREEN = (0, 1, 0)
YELLOW = (0.8, 1, 0)
RED = (1, 0, 0)

# Display params
CELL_SIZE = 6 * 16
RADIUS = 32
SPACE = (CELL_SIZE - 2 * RADIUS) // 2
FOOTER_HEIGHT = 6 * 16

# Params
COLOR_PLAYER1 = RED
COLOR_PLAYER2 = YELLOW
COLOR_EMPTY = BLACK
COLORS = [COLOR_PLAYER1, COLOR_PLAYER2, COLOR_EMPTY]


class REWARD:
    DRAW = 0
    WIN = 100000
    LOST = -WIN
    BAD_MOVE = -101000 #-MyPuissance4Env.ESTIMATE_4 #np.float32(-0.95 * 1000)
    OTHER_FAILED = 99000


class MyPuissance4Env(py_environment.PyEnvironment):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, nb_rows=6, nb_cols=7):
        super().__init__()

        self.nb_rows = nb_rows
        self.nb_cols = nb_cols

        self.BOARD_WIDTH = self.nb_cols * CELL_SIZE
        self.BOARD_HEIGHT = self.nb_rows * CELL_SIZE

        self._action_spec = array_spec.BoundedArraySpec(shape=(),
                                                        dtype=np.int32,
                                                        minimum=0,
                                                        maximum=6,
                                                        name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self.nb_rows, self.nb_cols, 4),
            dtype=np.float32,
            minimum=0.0,
            maximum=1.0,
            name='observation')
        self._time_step_observation = self._observation_spec
        self._time_step_step_type_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, name='step_type', minimum=0)
        self._time_step_discount_spec = array_spec.BoundedArraySpec(
            shape=(),
            dtype=np.float32,
            name='discount',
            minimum=0.0,
            maximum=1.0)
        self._time_step_reward_spec = array_spec.BoundedArraySpec(
            shape=(),
            dtype=np.float32,
            name='reward',
            minimum=REWARD.BAD_MOVE,
            maximum=REWARD.WIN)
        self._time_step_spec = TimeStep(self._time_step_step_type_spec,
                                        self._time_step_reward_spec,
                                        self._time_step_discount_spec,
                                        self._time_step_observation)

        # to comply with PyEnvironment from tensorflow's tf-agents
        self.handle_auto_reset: bool = True
        self._parallel_execution: bool = False

        #        # Define action and observation space
        #        # They must be gym.spaces objects
        #        # Example when using discrete actions:
        #        self.action_space = spaces.Discrete(nb_cols)
        #        # Example for using image as input:
        #        self.observation_space = spaces.Box(low=0,
        #                                            high=2,
        #                                            shape=(6, 7),
        #                                            dtype=np.uint8)
        #
        #        self.reward_range = (REWARD.LOST, REWARD.WIN)
        #        self.action_space = spaces.Box(low=np.array([0]),
        #                                       high=np.array([Board.WIDTH]),
        #                                       dtype=np.uint8)
        #        self.observation_space = spaces.Box(low=0,
        #                                            high=2,
        #                                            shape=(6, 7),
        #                                            dtype=np.uint8)
        self.reset()

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def time_step_spec(self):
        return self._time_step_spec

    def current_time_step(self):
        return self._current_time_step

    def _isColumnAlreadyFull(self, col):
        return self._compute_current_BB().get(self.nb_rows - 1, col)

    def _step(self, action:int):
        whoseTurn = self.whoseTurn
        tstep = self._inner_step(action)
        self.cumulated_rewards[whoseTurn] += tstep.reward
        return tstep

    def _compute_current_BB(self):
        bb_current = 0
        for bb_player in self.bb_players:
            bb_current |= bb_player.bb
        return self.new_BB(initial=bb_current)

    def print_bb(self):
      current = self._compute_current_BB()
      for row in reversed(range(current.nb_rows)):
        line = ""
        for col in range(current.nb_cols):
          if current.get(row, col) == 0:
            line += '_ '
          else:
            for i, bb_player in enumerate(self.bb_players):
              if bb_player.get(row, col) == 1:
                line += str(i+1) + ' '
        print(line)
      print("--")

    def _inner_step(self, action:int):      
        self.current_step += 1

        if self._episode_ended:
            #print("AUTO RESET !", self.current_step)
            self._reset()

        ok = not self._isColumnAlreadyFull(action)

        if not ok:
            #print("BAD MOVE !", self.current_step)
            self._episode_ended = True
            self.winner = (self.whoseTurn + 1) % 2
            self.whoseTurn = 2
            #print("\n\n\n@@@@@@@@@@ BAD MOVE !!!!! @@@@@@@@@@\n\n\n")
            return ts.termination(self._state, REWARD.BAD_MOVE)

        # take action
        bb_current = self._compute_current_BB()
        for targetRow in range(self.nb_rows):
            if bb_current.get(targetRow, action) == 0:
                self.bb_players[self.whoseTurn].set(targetRow, action)
                #actualize the current view
                bb_current = self._compute_current_BB()
                break

        # done status
        if self.bb_players[self.whoseTurn].hasFour():
            self.winner = self.whoseTurn
            self._episode_ended = True
        elif bb_current.isFull():
            self.winner = 2
            self._episode_ended = True

        # observation
        self._state = self._compute_state()
        self.whoseTurn = self.current_step % 2

        reward = MyPuissance4Env.estimate(self.bb_players[0], self.bb_players[1])

        if self._episode_ended:
            return ts.termination(self._state, reward)

        return ts.transition(self._state, reward=reward, discount=DISCOUNT)

    def _computeColor(self, cell):
        index = 0 if cell[0] == 1 else 1 if cell[1] == 1 else 2
        return COLORS[index]

    def _compute_state(self):
        #4 LAYERS  R/G/B/A corresponding to P1/P2/empty/non-empty
        _state = np.zeros((self.nb_rows, self.nb_cols, 4), dtype=np.float32)        
        for row in range(self.nb_rows):
            for col in range(self.nb_cols):
                p1 = self.bb_players[0].get(row, col)
                p2 = self.bb_players[1].get(row, col)
                not_empty = p1 | p2
                _state[row][col] = np.array([p1, p2, not not_empty, not_empty]) * 1.0 # as floats
        return _state

    def new_BB(self, initial=0):
        return BB(nb_rows=self.nb_rows, nb_cols=self.nb_cols, initial=initial)

    def _reset(self):
        self.viewer = None
        self.current_step = 0
        self._episode_ended = False
        self.cumulated_rewards = [0.0, 0.0, 0.0]
        #self.bb_player1 = BB()
        #self.bb_player2 = BB()
        self.winner = 2
        self.whoseTurn = 0
        self.bb_players = [self.new_BB(), self.new_BB()]
        self._state = self._compute_state()
        return ts.restart(self._state)

    def render(self, mode="rgb_array", close=False):
        screen_width = self.BOARD_WIDTH
        screen_height = self.BOARD_HEIGHT + FOOTER_HEIGHT

        obs = self._state

        background_color = (BLUE if not self._episode_ended else
                            COLORS[self.winner])
        if self.viewer is None:

            self.viewer = rendering.Viewer(screen_width, screen_height)
            pyglet.gl.glEnable(pyglet.gl.GL_LINE_SMOOTH)
            pyglet.gl.glHint(pyglet.gl.GL_LINE_SMOOTH_HINT,
                             pyglet.gl.GL_NICEST)

            #print("Background :", background_color)

            self.bg = rendering.FilledPolygon([
                (0, 0),
                (screen_width, 0),
                (screen_width, screen_height),
                (0, screen_height),
            ],
                                              #color=background_color,
                                              )
            self.viewer.add_geom(self.bg)

            self.cumul_reward_label = Label("Cumulated reward :",
                                            x=5,
                                            y=FOOTER_HEIGHT - 10,
                                            font_size=16)
            self.viewer.add_geom(self.cumul_reward_label)

            self.cumul_reward_1_label = Label("0.00",
                                              x=250,
                                              y=FOOTER_HEIGHT - 10,
                                              font_size=16)
            r, g, b = COLORS[1]
            self.cumul_reward_1_label.set_color(r, g, b)
            self.viewer.add_geom(self.cumul_reward_1_label)

            self.cumul_reward_2_label = Label("0.00",
                                              x=320,
                                              y=FOOTER_HEIGHT - 10,
                                              font_size=16)
            r, g, b = COLORS[2]
            self.cumul_reward_2_label.set_color(r, g, b)
            self.viewer.add_geom(self.cumul_reward_2_label)

            self.reward_label = Label("Reward",
                                      x=5,
                                      y=FOOTER_HEIGHT - 42,
                                      font_size=16)
            self.viewer.add_geom(self.reward_label)

            self.nextplayer_label = Label("Next player :",
                                          x=5,
                                          y=FOOTER_HEIGHT - 74,
                                          font_size=16)
            self.viewer.add_geom(self.nextplayer_label)

            self.nextplayer_indicator = rendering.make_circle(RADIUS / 3)
            r, g, b = COLORS[self.whoseTurn]
            self.nextplayer_indicator.set_color(r, g, b)
            self.nextplayer_indicator.add_attr(
                rendering.Transform(translation=(200, FOOTER_HEIGHT - 74)))
            self.viewer.add_geom(self.nextplayer_indicator)

            self.step_type_label = Label("",
                                         x=300,
                                         y=FOOTER_HEIGHT - 16,
                                         font_size=32,
                                         color=(255, 128, 0, 255))
            self.viewer.add_geom(self.step_type_label)

            self.winner_label = Label("",
                                      x=300,
                                      y=FOOTER_HEIGHT - 58,
                                      font_size=32)
            self.viewer.add_geom(self.winner_label)

            self.cells = []
            for col in range(0, self.nb_cols):
                column = []
                self.cells.append(column)
                for row in reversed(range(self.nb_rows)):
                    #print("Go", col, row, "=>", row+col*Board.HEIGHT)
                    #print(obs)
                    cell = rendering.make_circle(RADIUS)
                    r, g, b = self._computeColor(obs[row, col])
                    cell.set_color(r, g, b)
                    #self.cells[row][col] = cell
                    column.append(cell)
                    cell.add_attr(
                        rendering.Transform(
                            translation=(SPACE + RADIUS + CELL_SIZE * col,
                                         FOOTER_HEIGHT + self.BOARD_HEIGHT -
                                         (SPACE + RADIUS + CELL_SIZE * row))))
                    self.viewer.add_geom(cell)

    #if self.state is None:
    #    return None

    # Edit the colours !

        self.reward_label_text = f'Last reward : {self._current_time_step.reward:.2f}'
        self.cumul_reward_1_label.text = f'{self.cumulated_rewards[1]:.2f}'
        self.cumul_reward_2_label.text = f'{self.cumulated_rewards[2]:.2f}'

        r, g, b = COLORS[self.whoseTurn]
        self.nextplayer_indicator.set_color(r, g, b)

        if self._current_time_step.is_last():
            if self.winner == 2:
                self.winner_label.text = "IT's A DRAW"
            else:
                if self._current_time_step.reward == REWARD.BAD_MOVE:
                    r, g, b = COLORS[self.whoseTurn]
                    self.step_type_label.set_color(r, g, b)
                    self.step_type_label.text = "BAD MOVE !"
                r, g, b = COLORS[self.whoseTurn]
                self.winner_label.set_color(r, g, b)
                self.winner_label.text = f'PLAYER {self.winner+1} WINS'

        r, g, b = background_color
        self.bg.set_color(r, g, b)
        for col in range(0, self.nb_cols):
            for row in range(self.nb_rows):
                #print("Editing ", row, col, len(self.cells), len(self.cells[col]))
                cell = self.cells[col][row]
                r, g, b = self._computeColor(obs[row, col])
                cell.set_color(r, g, b)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def _inverse(self, timestep: TimeStep):
        if hasattr(timestep.observation.shape, 'rank'):
            outer_rank = tf_agents.utils.nest_utils.get_outer_rank(
                timestep.observation, self._observation_spec)
            batch_squash = tf_agents.networks.utils.BatchSquash(outer_rank)
            observations = tf.nest.map_structure(batch_squash.flatten,
                                                 timestep.observation)
            new_obs = np.copy(observations.numpy())
            for obs in new_obs:
                MyPuissance4Env._inplace_inverse(new_obs)
        else:
            new_obs = np.copy(timestep.observation)
            MyPuissance4Env._inplace_inverse(new_obs)

        new_ts = TimeStep(step_type=timestep.step_type,
                          reward=timestep.reward,
                          discount=timestep.discount,
                          observation=new_obs)
        return new_ts

    # inverse P1 and P2
    def _inplace_inverse(obs: np.array):
        for row in obs:
            for cell in row:
                c0 = cell[0]
                c1 = cell[1]
                cell[0]=c1
                cell[1]=c0


    ESTIMATE_4 = REWARD.WIN
    ESTIMATE_3 = 1000
    ESTIMATE_2 = 10
    ESTIMATE_1 = 1

    @lru_cache(maxsize=1*1024*1024)
    def estimate(bb1: BB, bb2: BB) -> int:

        bb_current = BB(initial=bb1.bb | bb2.bb)
        # draw
        if (bb_current.isFull()):
            #debug("ESTIMATION: DRAW !")
            # bb_current.printBB()
            return REWARD.DRAW

        #debug(f'Player 1 ({bb1.bb}):')
        estimation_p1 = MyPuissance4Env.estimate_player(bb1, bb2)
        #debug(f'Player 2 ({bb2.bb}):')
        estimation_p2 = -MyPuissance4Env.estimate_player(bb2, bb1)
        estimation = estimation_p1 + estimation_p2
        #debug(f'ESTIMATION: P1:{estimation_p1}\tP2:{estimation_p2}\tGLOBAL:{estimation}\t')
        return estimation

    def estimate_player(bb_estimated: BB, bb_other: BB) -> int:
        # 4
        if bb_estimated.hasFour():
            #debug(f' Found 4 !')
            return MyPuissance4Env.ESTIMATE_4

        estimation = 0

        empty = ~(bb_estimated.bb | bb_other.bb)

        # three vertically, with an empty cell above
        # player 1
        vthrees = ((bb_estimated.bb & bb_estimated.bb << 1) &
                   ((bb_estimated.bb & bb_estimated.bb << 1) << 1))
        vthrees_with_empty_above = (vthrees << 1) & empty
        nb_vthrees = BB(initial=vthrees_with_empty_above).count()
        #debug(f' Found {nb_vthrees} vertical 3s')
        estimation += nb_vthrees * MyPuissance4Env.ESTIMATE_3

        # two vertically, with an empty cell above
        # player 1
        vtwos = (bb_estimated.bb & bb_estimated.bb << 1) & ~vthrees
        vtwos_with_empty_above = (vtwos << 1) & empty
        nb_vtwos = BB(initial=vtwos_with_empty_above).count()
        #debug(f' Found {nb_vtwos} vertical 2s')
        estimation += nb_vtwos * MyPuissance4Env.ESTIMATE_2

        SHIFT = bb_estimated.size

        # three horizontally
        hthrees = ((bb_estimated.bb & bb_estimated.bb << SHIFT) &
                   ((bb_estimated.bb & bb_estimated.bb << SHIFT) << SHIFT))
        # three with a free slot on their right
        hthrees_with_empty_right = (hthrees << SHIFT) & empty
        nb_hthrees_r = BB(initial=hthrees_with_empty_right).count()
        #debug(f' Found {nb_hthrees_r} horizontal 3s with right free space')
        estimation += nb_hthrees_r * MyPuissance4Env.ESTIMATE_3
        # three with a free slot on their left
        hthrees_with_empty_left = (hthrees >> (2 * SHIFT)) & empty
        nb_hthrees_l = BB(initial=hthrees_with_empty_left).count()
        #debug(f' Found {nb_hthrees_l} horizontal 3s with left free space')
        estimation += nb_hthrees_l * MyPuissance4Env.ESTIMATE_3

        # two horizontally
        htwos = bb_estimated.bb & (bb_estimated.bb << SHIFT) & ~hthrees
        # two with a free slot on their right
        htwos_with_empty_right = ((htwos) << SHIFT) & empty
        nb_htwos_r = BB(initial=htwos_with_empty_right).count()
        #debug(f' Found {nb_htwos_r} horizontal 2s with right free space')
        estimation += nb_htwos_r * MyPuissance4Env.ESTIMATE_2
        # two with a free slot on their left
        htwos_with_empty_left = (htwos >> (2 * SHIFT)) & empty
        nb_htwos_l = BB(initial=htwos_with_empty_left).count()
        #debug(f' Found {nb_htwos_l} horizontal 2s with left free space')
        estimation += nb_htwos_l * MyPuissance4Env.ESTIMATE_2

        # one
        ones = bb_estimated.bb & (
            bb_estimated.bb << SHIFT) & ~htwos & ~vtwos & ~hthrees & ~vthrees
        # one with a free slot above
        ones_with_empty_above = (ones << 1) & empty
        nb_ones_a = BB(initial=ones_with_empty_above).count()
        #debug(f' Found {nb_ones_a} slots with a free space above')
        estimation += nb_ones_a * MyPuissance4Env.ESTIMATE_1

        # one with a free slot on their right
        ones_with_empty_right = (ones << SHIFT) & empty
        nb_ones_r = BB(initial=ones_with_empty_right).count()
        #debug(f' Found {nb_ones_r} slots with a free space on the right')
        estimation += nb_ones_r * MyPuissance4Env.ESTIMATE_1

        # one with a free slot on their left
        ones_with_empty_left = (ones >> SHIFT) & empty
        nb_ones_l = BB(initial=ones_with_empty_left).count()

        estimation += nb_ones_l * MyPuissance4Env.ESTIMATE_1
        #debug(f' Found {nb_ones_l} slots with a free space on the left')

        # diag down,left>up,right
        dupthrees = (bb_estimated.bb & (bb_estimated.bb <<
                                        (SHIFT + 1))) & ((bb_estimated.bb &
                                                          (bb_estimated.bb <<
                                                           (SHIFT + 1))) <<
                                                         (SHIFT + 1))
        # diag up with a free fourth slot on their upper right
        dupthrees_with_empty_up_right = (dupthrees << (SHIFT + 1)) & empty
        nb_dup_r = BB(initial=dupthrees_with_empty_up_right).count()
        #debug(f' Found {nb_dup_r} up right diags with a free space on the top right')
        estimation += nb_dup_r * MyPuissance4Env.ESTIMATE_3

        # diag up with a free fourth slot on their bottom left
        dupthrees_with_empty_bottom_left = (dupthrees >>
                                            (3 * (SHIFT + 1))) & empty
        nb_dup_l = BB(initial=dupthrees_with_empty_bottom_left).count()
        #debug(f' Found {nb_dup_l} up right diags with a free space on the bottom left')
        estimation += nb_dup_l * MyPuissance4Env.ESTIMATE_3

        # diag up,left>bottom,right
        ddownthrees = (bb_estimated.bb & (bb_estimated.bb <<
                                          (SHIFT - 1))) & ((bb_estimated.bb &
                                                            (bb_estimated.bb <<
                                                             (SHIFT - 1))) <<
                                                           (SHIFT - 1))
        # diag down with a free fourth slot on their bottom right
        ddownthrees_with_empty_bottom_right = (ddownthrees <<
                                               (SHIFT - 1)) & empty
        nb_ddown_r = BB(initial=ddownthrees_with_empty_bottom_right).count()
        #debug(f' Found {nb_ddown_r} down left diags with a free space on the bottom right')
        estimation += nb_ddown_r * MyPuissance4Env.ESTIMATE_3

        # diag down with a free fourth slot on their upper left
        ddownthrees_with_empty_up_left = (ddownthrees >>
                                          (3 * (SHIFT - 1))) & empty
        nb_ddown_l = BB(initial=ddownthrees_with_empty_up_left).count()
        #debug(f' Found {nb_ddown_l} down left diags with a free space on the upper left')
        estimation += nb_ddown_l * MyPuissance4Env.ESTIMATE_3

        return estimation
