from collections.abc import Iterable
from functools import lru_cache
import numpy as np
from bb import BB

import tensorflow as tf
import tf_agents
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import TimeStep

from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager

DISCOUNT = 0.95

EMPTY = np.float32(0.0)
PLAYER1 = np.float32(0.5)
PLAYER2 = np.float32(1.0)

# Colors
BLUE = 'blue'
BLACK = 'black'
WHITE = 'white'
GREEN = 'green'
YELLOW = '#CCFF00'
RED = 'red'

# Display params
CELL_SIZE = 6 * 16
RADIUS = 32
SPACE = (CELL_SIZE - 2 * RADIUS) // 2
FOOTER_HEIGHT = 6 * 24

# Params
COLOR_PLAYER1 = RED
COLOR_PLAYER2 = YELLOW
COLOR_EMPTY = BLACK
COLOR_WIN = GREEN
COLORS = [COLOR_PLAYER1, COLOR_PLAYER2, COLOR_EMPTY]


class MyPuissance4Env(py_environment.PyEnvironment):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    #### rewarding_mode can be 'estimation' : the reward will be base on the estimation of the board
    #### or 'simple' : reward will be 1 if last player won, 0.5 if it's a draw, 0 otherwise
    def __init__(self, rewarding_mode='simple'):
        super().__init__(handle_auto_reset=True)

        self.rewarding_mode = rewarding_mode

        if self.rewarding_mode == 'estimation':
            self.REWARD_DRAW = 0.5
            self.REWARD_WIN = 100000
            self.REWARD_LOST = -100000
            self.REWARD_BAD_MOVE = -100000
            self.REWARD_OTHER_FAILED = 99000
        else:
            self.REWARD_DRAW = 0.5
            self.REWARD_WIN = 1
            self.REWARD_LOST = -1
            self.REWARD_BAD_MOVE = -1
            self.REWARD_OTHER_FAILED = 0.99

        self.BOARD_WIDTH = BB.NB_COLS * CELL_SIZE
        self.BOARD_HEIGHT = BB.NB_ROWS * CELL_SIZE

        self._action_spec = array_spec.BoundedArraySpec(shape=(),
                                                        dtype=np.int32,
                                                        minimum=0,
                                                        maximum=BB.NB_COLS - 1,
                                                        name='action')
        self._observation_spec = {
            'observation': array_spec.BoundedArraySpec(
                shape=(BB.NB_ROWS, BB.NB_COLS, 4),
                dtype=np.float32,
                minimum=0.0,
                maximum=1.0,
                name='observation'),
            'next_player': array_spec.BoundedArraySpec(
                name="next_player",
                shape=(),
                dtype=np.int64,
                minimum=0,
                maximum=1
            ),
            'valid_actions': array_spec.BoundedArraySpec(
                name="valid_actions",
                shape=(BB.NB_COLS,),
                dtype=np.int32,
                minimum=0,
                maximum=1
            ),
            'winner': array_spec.BoundedArraySpec(
                name="winner",
                shape=(2,),
                dtype=np.int32,
                minimum=0.0,
                maximum=1.0
            )
        }
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
            minimum=self.REWARD_BAD_MOVE,
            maximum=self.REWARD_WIN)
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
        #        self.reward_range = (self.REWARD_LOST, self.REWARD_WIN)
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
        return self._compute_current_BB().get(BB.NB_ROWS - 1, col)

    def _step(self, action: int):
        if self._episode_ended:
            # print("AUTO RESET !", self.current_step)
            self._reset()
        # print("STEP", self.current_step)
        whoseTurn = self._state['next_player']
        tstep = self._inner_step(action)
        self.cumulated_rewards[whoseTurn] += tstep.reward
        self.last_reward = tstep.reward
        return tstep

    def _compute_current_BB(self):
        bb_current = 0
        for bb_player in self.bb_players:
            bb_current |= bb_player.bb
        return BB(initial=bb_current)

    def print_bb(self):
        current = self._compute_current_BB()
        for row in reversed(range(current.NB_ROWS)):
            line = ""
            for col in range(current.NB_COLS):
                if current.get(row, col) == 0:
                    line += '_ '
                else:
                    for i, bb_player in enumerate(self.bb_players):
                        if bb_player.get(row, col) == 1:
                            line += str(i + 1) + ' '
            print(line)
        print("--")

    def _inner_step(self, action: int):

        self.current_step += 1
        whoseTurn = (self.current_step) % 2

        ok = not self._isColumnAlreadyFull(action)

        if not ok:
            # print("BAD MOVE !", self.current_step)
            self._episode_ended = True
            self.winner = (whoseTurn + 1) % 2
            self._state['next_player'] = 2
            self._state['winner'] = self._compute_winner()
            # why not recompute whole state ?
            # print("\n\n\n@@@@@@@@@@ BAD MOVE !!!!! @@@@@@@@@@\n\n\n")
            return ts.termination(self._state, self.REWARD_BAD_MOVE)

        # take action
        bb_current = self._compute_current_BB()
        for targetRow in range(BB.NB_ROWS):
            if bb_current.get(targetRow, action) == 0:
                self.bb_players[whoseTurn].set(targetRow, action)
                # actualize the current view
                bb_current = self._compute_current_BB()
                break

        # done status
        reward = 0
        if BB.HasFour(self.bb_players[whoseTurn].bb):
            self.winner = whoseTurn
            self._episode_ended = True
            reward = self.REWARD_WIN
        elif bb_current.isFull():
            self.winner = 2
            self._episode_ended = True
            reward = self.REWARD_DRAW

        # observation
        self._state = self._compute_state()

        if self.rewarding_mode == 'estimation':
            reward = MyPuissance4Env.estimate(self.bb_players[0].bb, self.bb_players[1].bb)

        if self._episode_ended:
            # print("====> FIN ", reward)
            return ts.termination(self._state, reward)

        return ts.transition(self._state, reward=reward, discount=DISCOUNT)

    def _computeColor(self, cell):
        index = 0 if cell[0] == 1 else 1 if cell[1] == 1 else 2
        return COLORS[index]

    def _compute_state(self):
        # _compute_valid_actions_mask() uses _state.observation so affect it in a second time:
        observation = self._compute_observation()
        valid_actions_mask = self._compute_valid_actions_mask(observation)
        next_player = self._compute_next_player()
        winner = self._compute_winner()
        self._state = {'observation': observation, 'valid_actions': valid_actions_mask, 'next_player': next_player,
                       'winner': winner}
        return self._state

    def _compute_observation(self):
        # 4 LAYERS  R/G/B/A corresponding to P1/P2/empty/non-empty
        # _state = np.zeros((BB.NB_ROWS, BB.NB_COLS, 4), dtype=np.float32)
        # modifying a np.ndarray is pretty slow - manipulate lists then convert eventually.
        _observations = []
        for row in range(BB.NB_ROWS):
            line = []
            _observations.append(line)
            for col in range(BB.NB_COLS):
                p1 = self.bb_players[0].get(row, col)
                p2 = self.bb_players[1].get(row, col)
                not_empty = p1 | p2
                line.append([float(p1), float(p2), float(not not_empty), float(not_empty)])
        _observations = np.array(_observations, dtype=np.float32)
        return _observations

    def _compute_valid_actions_mask(self, obs):
        # available cols have a 1 in the upper cell's third layer/element
        return obs[-1, :, 2].astype(np.int32)

    def _compute_next_player(self):
        return np.array((self.current_step + 1) % 2)

    def _compute_winner(self):
        return np.array([self.winner == 0 or self.winner == 2, self.winner == 1 or self.winner == 2], dtype=np.int32)

    def get_current_winning_fours(self):
        return BB.getWinningFours(self.bb_players[0].bb), BB.getWinningFours(self.bb_players[1].bb)

    def _reset(self):
        # print("RESET")
        self.viewer = None
        self.current_step = -1
        self._episode_ended = False
        self.cumulated_rewards = [0.0, 0.0, 0.0]
        self.last_reward = 0.0
        # self.bb_player1 = BB()
        # self.bb_player2 = BB()
        self.winner = -1
        self.bb_players = [BB(), BB()]
        self._state = self._compute_state()
        return ts.restart(self._state)

    def _colorize_winning_fours(self, draw: ImageDraw):
        for bb in self.bb_players:
            self._colorize_winning_fours_player(draw, bb)

    def _colorize_winning_fours_player(self, draw: ImageDraw, bb: BB):

        horizons, verticals, diagups, diagdowns = BB.getWinningFours(bb.bb)

        # compute
        # Check \
        if diagdowns:
            bb2 = BB(initial=diagdowns)
            for row in range(BB.NB_ROWS):
                for col in range(BB.NB_COLS):
                    if bb2.get(row, col):
                        # print("Found diag down", row, col)
                        for i in range(4):
                            self._draw_cell(draw, row - i, col + i, color=COLOR_WIN)
        # Check -
        if horizons:
            bb2 = BB(initial=horizons)
            for row in range(BB.NB_ROWS):
                for col in range(BB.NB_COLS):
                    if bb2.get(row, col):
                        # print("Found horizontal ", row, col)
                        for i in range(4):
                            self._draw_cell(draw, row, col + i, color=COLOR_WIN)
        # Check /
        if diagups:
            bb2 = BB(initial=diagups)
            for row in range(BB.NB_ROWS):
                for col in range(BB.NB_COLS):
                    if bb2.get(row, col):
                        # print("Found diag up ", row, col)
                        for i in range(4):
                            self._draw_cell(draw, row + i, col + i, color=COLOR_WIN)
        # Check |
        if verticals:
            bb2 = BB(initial=verticals)
            for row in range(BB.NB_ROWS):
                for col in range(BB.NB_COLS):
                    if bb2.get(row, col):
                        # print("Found vertical ", row, col)
                        for i in range(4):
                            self._draw_cell(draw, row + i, col, color=COLOR_WIN)

    def render(self, mode="rgb_array", close=False):
        screen_width = self.BOARD_WIDTH
        screen_height = self.BOARD_HEIGHT + FOOTER_HEIGHT
        FOOTER_TOP = self.BOARD_HEIGHT

        obs = self._state['observation']

        background_color = (BLUE if not self._episode_ended else
                            COLORS[self.winner])

        img = Image.new('RGB', (screen_width, screen_height))
        draw = ImageDraw.Draw(img)

        # draw background color
        draw.rectangle([(0, 0), (self.BOARD_WIDTH, self.BOARD_HEIGHT)], fill=background_color)

        for row in reversed(range(BB.NB_ROWS)):
            for col in range(0, BB.NB_COLS):
                self._draw_cell(draw, row, col, color=self._computeColor(obs[row, col]))

        reward_label_text = f'Last reward : {self._current_time_step.reward:.2f}'

        # prepare fonts
        from matplotlib import font_manager
        fonts = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
        base_fonts = [font for font in fonts if
                      font.__contains__('LiberationSans - Regular.ttf') or font.__contains__('verdana.ttf')]
        if len(base_fonts) < 1:
            base_font = fonts[0]
        else:
            base_font = base_fonts[0]

        font_label = ImageFont.truetype(base_font, 16)
        font_big = ImageFont.truetype(base_font, 32)

        draw.line((0, FOOTER_TOP, self.BOARD_WIDTH, FOOTER_TOP), fill='white')

        draw.text((SPACE, FOOTER_TOP + 10), "Cumulated rewards :", font=font_label, fill=WHITE)
        draw.text((SPACE, FOOTER_TOP + 42), "Last reward :", font=font_label, fill=WHITE)
        draw.text((SPACE + 200, FOOTER_TOP + 42), f"{self.last_reward}", font=font_label,
                  fill=COLORS[self._state['next_player']])

        whoseTurn = (self._state['next_player'] - 1) % 2
        if not self._current_time_step.is_last():
            draw.text((SPACE, FOOTER_TOP + 74), "Next player :", font=font_label, fill=WHITE)
            self._draw_circle(draw, SPACE + 200 + RADIUS // 3, FOOTER_TOP + 74 + 10, radius=RADIUS // 3,
                              fill=COLORS[self._state['next_player']], outline=COLORS[self._state['next_player']])

        draw.text((SPACE + 200, FOOTER_TOP + 10), f'{self.cumulated_rewards[0]:.2f}', font=font_label, fill=COLORS[0])
        draw.text((SPACE + 320, FOOTER_TOP + 10), f'{self.cumulated_rewards[1]:.2f}', font=font_label, fill=COLORS[1])

        winner_label_text = ""
        step_type_label_text = ""
        winner_label_color = BLUE
        step_type_label_color = BLUE
        if self._current_time_step.is_last():
            if self.winner == 2:
                winner_label_text = "IT'S A DRAW"
                winner_label_color = WHITE
            else:
                winner_label_color = COLORS[self._state['next_player']]
                winner_label_text = f'PLAYER {self.winner + 1} WINS'
                if self._current_time_step.reward == self.REWARD_BAD_MOVE:
                    step_type_label_color = 'orange'
                    step_type_label_text = "BAD MOVE !"

            draw.text((SPACE + 390, FOOTER_TOP + 16), step_type_label_text, font=font_big, fill=step_type_label_color)

            draw.text((SPACE + 390, FOOTER_TOP + 58), winner_label_text, font=font_big, fill=winner_label_color)

            self._colorize_winning_fours(draw)
        return img

    def _draw_circle(self, draw: ImageDraw, x: int, y: int, radius: int = RADIUS, fill='white', outline='#666666'):
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline)

    def _draw_cell(self, draw: ImageDraw, row: int, col: int, color: int):
        self._draw_circle(draw, SPACE + RADIUS + CELL_SIZE * col,
                          self.BOARD_HEIGHT - (SPACE + RADIUS + CELL_SIZE * row), fill=color)

    def close(self):
        pass

    def _inverse(self, timestep: TimeStep):
        if hasattr(timestep.observation['observation'].shape, 'rank'):
            outer_rank = tf_agents.utils.nest_utils.get_outer_rank(
                timestep.observation['observation'], self._observation_spec)
            batch_squash = tf_agents.networks.utils.BatchSquash(outer_rank)
            observations = tf.nest.map_structure(batch_squash.flatten,
                                                 timestep.observation)
            new_obs = np.copy(observations.numpy())
            for obs in new_obs:
                MyPuissance4Env._inplace_inverse(new_obs)
        else:
            new_obs = np.copy(timestep.observation['observation'])
            MyPuissance4Env._inplace_inverse(new_obs)

        inversed_observation = {k: v for k, v in timestep.observation.items()}
        inversed_observation['observation'] = new_obs
        new_ts = TimeStep(step_type=timestep.step_type,
                          reward=timestep.reward,
                          discount=timestep.discount,
                          observation=inversed_observation)
        return new_ts

    # inverse P1 and P2
    def _inplace_inverse(obs: np.array):
        for row in obs:
            for cell in row:
                c0 = cell[0]
                c1 = cell[1]
                cell[0] = c1
                cell[1] = c0

    ESTIMATE_4 = 100000
    ESTIMATE_3 = 1000
    ESTIMATE_2 = 10
    ESTIMATE_1 = 1

    @lru_cache(maxsize=64 * 1024 * 1024)
    def estimate(bb1: int, bb2: int) -> int:

        bb_current = bb1 | bb2
        # draw
        if BB.IsFull(bb_current):
            # debug("ESTIMATION: DRAW !")
            # bb_current.printBB()
            return 0

        # debug(f'Player 1 ({bb1}):')
        estimation_p1 = MyPuissance4Env.estimate_player(bb1, bb2)
        # debug(f'Player 2 ({bb2}):')
        estimation_p2 = -MyPuissance4Env.estimate_player(bb2, bb1)
        estimation = estimation_p1 + estimation_p2
        # debug(f'ESTIMATION: P1:{estimation_p1}\tP2:{estimation_p2}\tGLOBAL:{estimation}\t')
        return estimation

    def estimate_player(bb_estimated: int, bb_other: int) -> int:
        # 4
        if BB.HasFour(bb_estimated):
            # debug(f' Found 4 !')
            return MyPuissance4Env.ESTIMATE_4

        estimation = 0

        empty = ~(bb_estimated | bb_other)

        # three vertically, with an empty cell above
        # player 1
        vthrees = ((bb_estimated & bb_estimated << 1) &
                   ((bb_estimated & bb_estimated << 1) << 1))
        vthrees_with_empty_above = (vthrees << 1) & empty
        nb_vthrees = BB.Count(vthrees_with_empty_above)
        # debug(f' Found {nb_vthrees} vertical 3s')
        estimation += nb_vthrees * MyPuissance4Env.ESTIMATE_3

        # two vertically, with an empty cell above
        # player 1
        vtwos = (bb_estimated & bb_estimated << 1) & ~vthrees
        vtwos_with_empty_above = (vtwos << 1) & empty
        nb_vtwos = BB.Count(vtwos_with_empty_above)
        # debug(f' Found {nb_vtwos} vertical 2s')
        estimation += nb_vtwos * MyPuissance4Env.ESTIMATE_2

        # three horizontally
        hthrees = ((bb_estimated & bb_estimated << BB.SIZE) &
                   ((bb_estimated & bb_estimated << BB.SIZE) << BB.SIZE))
        # three with a free slot on their right
        hthrees_with_empty_right = (hthrees << BB.SIZE) & empty
        nb_hthrees_r = BB.Count(hthrees_with_empty_right)
        # debug(f' Found {nb_hthrees_r} horizontal 3s with right free space')
        estimation += nb_hthrees_r * MyPuissance4Env.ESTIMATE_3
        # three with a free slot on their left
        hthrees_with_empty_left = (hthrees >> (2 * BB.SIZE)) & empty
        nb_hthrees_l = BB.Count(hthrees_with_empty_left)
        # debug(f' Found {nb_hthrees_l} horizontal 3s with left free space')
        estimation += nb_hthrees_l * MyPuissance4Env.ESTIMATE_3

        # two horizontally
        htwos = bb_estimated & (bb_estimated << BB.SIZE) & ~hthrees
        # two with a free slot on their right
        htwos_with_empty_right = ((htwos) << BB.SIZE) & empty
        nb_htwos_r = BB.Count(htwos_with_empty_right)
        # debug(f' Found {nb_htwos_r} horizontal 2s with right free space')
        estimation += nb_htwos_r * MyPuissance4Env.ESTIMATE_2
        # two with a free slot on their left
        htwos_with_empty_left = (htwos >> (2 * BB.SIZE)) & empty
        nb_htwos_l = BB.Count(htwos_with_empty_left)
        # debug(f' Found {nb_htwos_l} horizontal 2s with left free space')
        estimation += nb_htwos_l * MyPuissance4Env.ESTIMATE_2

        # one
        ones = bb_estimated & (
                bb_estimated << BB.SIZE) & ~htwos & ~vtwos & ~hthrees & ~vthrees
        # one with a free slot above
        ones_with_empty_above = (ones << 1) & empty
        nb_ones_a = BB.Count(ones_with_empty_above)
        # debug(f' Found {nb_ones_a} slots with a free space above')
        estimation += nb_ones_a * MyPuissance4Env.ESTIMATE_1

        # one with a free slot on their right
        ones_with_empty_right = (ones << BB.SIZE) & empty
        nb_ones_r = BB.Count(ones_with_empty_right)
        # debug(f' Found {nb_ones_r} slots with a free space on the right')
        estimation += nb_ones_r * MyPuissance4Env.ESTIMATE_1

        # one with a free slot on their left
        ones_with_empty_left = (ones >> BB.SIZE) & empty
        nb_ones_l = BB.Count(ones_with_empty_left)

        estimation += nb_ones_l * MyPuissance4Env.ESTIMATE_1
        # debug(f' Found {nb_ones_l} slots with a free space on the left')

        # diag down,left>up,right
        dupthrees = (bb_estimated & (bb_estimated <<
                                     (BB.SIZE + 1))) & ((bb_estimated &
                                                         (bb_estimated <<
                                                          (BB.SIZE + 1))) <<
                                                        (BB.SIZE + 1))
        # diag up with a free fourth slot on their upper right
        dupthrees_with_empty_up_right = (dupthrees << (BB.SIZE + 1)) & empty
        nb_dup_r = BB.Count(dupthrees_with_empty_up_right)
        # debug(f' Found {nb_dup_r} up right diags with a free space on the top right')
        estimation += nb_dup_r * MyPuissance4Env.ESTIMATE_3

        # diag up with a free fourth slot on their bottom left
        dupthrees_with_empty_bottom_left = (dupthrees >>
                                            (3 * (BB.SIZE + 1))) & empty
        nb_dup_l = BB.Count(dupthrees_with_empty_bottom_left)
        # debug(f' Found {nb_dup_l} up right diags with a free space on the bottom left')
        estimation += nb_dup_l * MyPuissance4Env.ESTIMATE_3

        # diag up,left>bottom,right
        ddownthrees = (bb_estimated & (bb_estimated <<
                                       (BB.SIZE - 1))) & ((bb_estimated &
                                                           (bb_estimated <<
                                                            (BB.SIZE - 1))) <<
                                                          (BB.SIZE - 1))
        # diag down with a free fourth slot on their bottom right
        ddownthrees_with_empty_bottom_right = (ddownthrees <<
                                               (BB.SIZE - 1)) & empty
        nb_ddown_r = BB.Count(ddownthrees_with_empty_bottom_right)
        # debug(f' Found {nb_ddown_r} down left diags with a free space on the bottom right')
        estimation += nb_ddown_r * MyPuissance4Env.ESTIMATE_3

        # diag down with a free fourth slot on their upper left
        ddownthrees_with_empty_up_left = (ddownthrees >>
                                          (3 * (BB.SIZE - 1))) & empty
        nb_ddown_l = BB.Count(ddownthrees_with_empty_up_left)
        # debug(f' Found {nb_ddown_l} down left diags with a free space on the upper left')
        estimation += nb_ddown_l * MyPuissance4Env.ESTIMATE_3

        return estimation
