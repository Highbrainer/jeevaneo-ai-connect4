import numpy as np
from board import Board
from pyglet_utils import Label

#import tensorflow as tf
import tf_agents
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import TimeStep

DISCOUNT=0.95

EMPTY=np.float32(0.0)
PLAYER1=np.float32(0.5)
PLAYER2=np.float32(1.0)

# Colors
BLUE = (0, 0.1, 0.8)
BLACK = (0, 0, 0)
WHITE = (1, 1, 1)
GREEN = (0, 1, 0)
YELLOW = (0.8, 1, 0)
RED = (1, 0, 0)

# Display params
CELL_SIZE = 6*16
RADIUS = 32
SPACE = (CELL_SIZE - 2 * RADIUS) // 2
FOOTER_HEIGHT = 6*16


BOARD_WIDTH = Board.WIDTH * CELL_SIZE
BOARD_HEIGHT = Board.HEIGHT * CELL_SIZE

# Params
COLOR_PLAYER1 = RED
COLOR_PLAYER2 = YELLOW
COLOR_EMPTY = BLACK
COLORS = [COLOR_EMPTY, COLOR_PLAYER1, COLOR_PLAYER2]

class REWARD:    
    LOST = np.float32(-1.0*1000)
    DRAW = np.float32(0.0)
    WIN = np.float32(1.0*1000)
    BAD_MOVE = np.float32(-0.95*1000)
    OTHER_FAILED = np.float32(0.1*1000)
    GOOD_MOVE = np.float32(0.001*1000)

class MyPuissance4Env(py_environment.PyEnvironment):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()

        self.whoseTurn = 1

        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=6, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(shape=(Board.HEIGHT, Board.WIDTH, 1), dtype=np.float32, minimum=0, maximum=1, name='observation')
        self._time_step_observation = self._observation_spec
        self._time_step_step_type_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, name='step_type', minimum=0)
        self._time_step_discount_spec  = array_spec.BoundedArraySpec(shape=(), dtype=np.float32, name='discount', minimum=0.0, maximum=1.0)
        self._time_step_reward_spec    = array_spec.BoundedArraySpec(shape=(), dtype=np.float32, name='reward', minimum=-1.0, maximum=1.0)
        self._time_step_spec = TimeStep(
            self._time_step_step_type_spec, 
            self._time_step_reward_spec, 
            self._time_step_discount_spec, 
            self._time_step_observation
        )

        # to comply with PyEnvironment from tensorflow's tf-agents
        self.handle_auto_reset: bool = True
        self._parallel_execution: bool = False

#        # Define action and observation space
#        # They must be gym.spaces objects
#        # Example when using discrete actions:
#        self.action_space = spaces.Discrete(Board.WIDTH)
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
      return len(self.board.board[col])>=Board.HEIGHT

    def _step(self, action):
      whoseTurn = self.whoseTurn
      tstep = self._inner_step(action)
      self.cumulated_rewards[whoseTurn] += tstep.reward
      return tstep

    def _inner_step(self, action):
      self.current_step += 1

      if self._episode_ended :
        #print("AUTO RESET !", self.current_step)
        self._reset() 

      ok = not self._isColumnAlreadyFull(action)

      if not ok:
        #print("BAD MOVE !", self.current_step)
        self._episode_ended = True
        self.winner = (self.whoseTurn%2)+1
        self.whoseTurn = 0
        #print("\n\n\n@@@@@@@@@@ BAD MOVE !!!!! @@@@@@@@@@\n\n\n")
        return ts.termination(self._state, REWARD.BAD_MOVE)

      self.board.makeMove(action)
      # done status      
      self.winner = self.board.isTerminal()
      self._episode_ended = self.winner > -1


      # observation
      self._state = self._compute_state()
      self.whoseTurn = (self.board.numMoves % 2)+1

      if self._episode_ended:        
        if self.winner == 0:
            reward = REWARD.DRAW
            #print("\n\n\n@@@@@@@@@@ DRAW !!!!! @@@@@@@@@@\n\n\n")
        elif self.winner == 1:
            reward = REWARD.WIN
            #print("\n\n\n@@@@@@@@@@ WIN !!!!! @@@@@@@@@@\n\n\n")
        else:
            reward = REWARD.LOST
            #print("\n\n\n@@@@@@@@@@ LOST !!!!! @@@@@@@@@@\n\n\n")
        #print("THE END", self.current_step, reward)
        self.whoseTurn = 0
        return ts.termination(self._state, reward)      

      reward = REWARD.GOOD_MOVE * ((self.current_step+1)//2)
      return ts.transition(self._state, reward=reward, discount=DISCOUNT)
      
    def _computeColor(self, cell:float):
      index = 0 if cell == EMPTY else 1 if cell == PLAYER1 else 2
      return COLORS[index]

    def _compute_state(self):
      obs = np.zeros((Board.HEIGHT, Board.WIDTH, 1), dtype=np.float32)
      for col in range(0, Board.WIDTH):
        for row in range(len(self.board.board[col])):
          #print("col", col, "row", row, "=>", row*Board.WIDTH+col, "\t| obs shape", obs.shape, len(obs), "\t| columns :", len(self.board.board), "x", len(self.board.board[col]))
          obs[row][col]=[(self.board.board[col][row]+1)/2]
      #return np.reshape(obs, (Board.HEIGHT*Board.WIDTH))
      return obs

    def _reset(self):
        self.viewer = None
        self.board = Board()
        self.current_step = 0
        self._episode_ended = False
        self._state = self._compute_state()
        self.cumulated_rewards = [0.0, 0.0, 0.0]
        return ts.restart(self._state)

    def render(self, mode="rgb_array", close=False):
        screen_width = BOARD_WIDTH
        screen_height = BOARD_HEIGHT + FOOTER_HEIGHT

        obs = self._state        

        background_color = (BLUE if not self._episode_ended else COLORS[self.winner])
        if self.viewer is None:            

            self.viewer = rendering.Viewer(screen_width, screen_height)                       
            pyglet.gl.glEnable(pyglet.gl.GL_LINE_SMOOTH);
            pyglet.gl.glHint(pyglet.gl.GL_LINE_SMOOTH_HINT, pyglet.gl.GL_NICEST);       

            #print("Background :", background_color)
            
            self.bg = rendering.FilledPolygon(
                [
                    (0, 0),
                    (screen_width, 0),
                    (screen_width, screen_height),
                    (0, screen_height),
                ],
                #color=background_color,
              )
            self.viewer.add_geom(self.bg)

            self.cumul_reward_label = Label("Cumulated reward :", x=5, y=FOOTER_HEIGHT-10, font_size=16)
            self.viewer.add_geom(self.cumul_reward_label)

            self.cumul_reward_1_label = Label("0.00", x=250, y=FOOTER_HEIGHT-10, font_size=16)
            r, g, b = COLORS[1]
            self.cumul_reward_1_label.set_color(r, g, b)
            self.viewer.add_geom(self.cumul_reward_1_label)

            self.cumul_reward_2_label = Label("0.00", x=320, y=FOOTER_HEIGHT-10, font_size=16)
            r, g, b = COLORS[2]
            self.cumul_reward_2_label.set_color(r, g, b)
            self.viewer.add_geom(self.cumul_reward_2_label)

            self.reward_label = Label("Reward", x=5, y=FOOTER_HEIGHT-42, font_size=16)
            self.viewer.add_geom(self.reward_label)

            self.nextplayer_label = Label("Next player :", x=5, y=FOOTER_HEIGHT-74, font_size=16)
            self.viewer.add_geom(self.nextplayer_label)
            
            self.nextplayer_indicator = rendering.make_circle(RADIUS/3)
            r, g, b = COLORS[self.whoseTurn]
            self.nextplayer_indicator.set_color(r, g, b)
            self.nextplayer_indicator.add_attr(
                    rendering.Transform(translation=(200,FOOTER_HEIGHT-74))
                  )
            self.viewer.add_geom(self.nextplayer_indicator)

            self.step_type_label = Label("", x=300, y=FOOTER_HEIGHT-16, font_size=32, color=(255,128,0, 255))
            self.viewer.add_geom(self.step_type_label)

            self.winner_label = Label("", x=300, y=FOOTER_HEIGHT-58, font_size=32)
            self.viewer.add_geom(self.winner_label)
            
            self.cells = []
            for col in range(0, Board.WIDTH):
                column = []
                self.cells.append(column)
                for row in range(Board.HEIGHT-1, -1, -1):
                  #print("Go", col, row, "=>", row+col*Board.HEIGHT)
                  #print(obs)
                  cell = rendering.make_circle(RADIUS)
                  r, g, b = self._computeColor(obs[row,col,0])
                  cell.set_color(r, g, b)
                  #self.cells[row][col] = cell
                  column.append(cell)
                  cell.add_attr(
                    rendering.Transform(translation=(SPACE + RADIUS + CELL_SIZE * col, FOOTER_HEIGHT + BOARD_HEIGHT -
                        (SPACE + RADIUS + CELL_SIZE * row)))
                  )
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
          if self.winner == 0: 
            self.winner_label.text= "IT's A DRAW"
          else:
            if self._current_time_step.reward == REWARD.BAD_MOVE:
              r, g, b = COLORS[self.whoseTurn]
              self.step_type_label.set_color(r, g, b)
              self.step_type_label.text = "BAD MOVE !"
            r, g, b = COLORS[self.whoseTurn]
            self.winner_label.set_color(r,g,b)
            self.winner_label.text= f'PLAYER {self.winner} WINS'


        r, g, b = background_color
        self.bg.set_color(r, g, b)
        for col in range(0, Board.WIDTH):
          for row in range(Board.HEIGHT-1, -1, -1):
            #print("Editing ", row, col, len(self.cells), len(self.cells[col]))
            cell = self.cells[col][row]
            r, g, b = self._computeColor(obs[row,col,0])
            cell.set_color(r, g, b)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def _inverse(self, timestep : TimeStep):
      if hasattr(timestep.observation.shape, 'rank'):
        outer_rank = tf_agents.utils.nest_utils.get_outer_rank(timestep.observation, self._observation_spec)
        batch_squash = tf_agents.networks.utils.BatchSquash(outer_rank)
        observations = tf.nest.map_structure(batch_squash.flatten, timestep.observation)
        new_obs = np.copy(observations.numpy())
        for obs in new_obs:
          self._inplace_inverse(new_obs)
      else:
        new_obs = timestep.observation
        self._inplace_inverse(new_obs)
      new_ts = TimeStep(step_type=timestep.step_type,
              reward=timestep.reward,
              discount=timestep.discount,
              observation=new_obs
              )
      return new_ts

    def _inplace_inverse(self, obs: np.array):
      for col in obs:
          for cell in col:
            cell[0]=1 if cell[0]==0.5 else 0.5 if cell[0]==1 else 0.0
