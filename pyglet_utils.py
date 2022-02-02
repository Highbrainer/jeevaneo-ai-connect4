from gym.envs.classic_control import rendering
import pyglet.gl

class Label(rendering.Geom):
  
    def __init__(self, text, x=20, y=20, font_size=36, color=(255, 255, 255, 255)):
        rendering.Geom.__init__(self)
        self.text=text
        self.pyglabel = pyglet.text.Label(
                text,
                font_size=font_size,
                x=x,
                y=y,
                anchor_x="left",
                anchor_y="center",
                color=color,
            )
        
    def set_color(self, r:float, g:float, b:float):
      color = (int(r*255), int(g*255), int(b*255), 255)
      self.pyglabel.color= color

    def render1(self):
      self.pyglabel.text = self.text
      self.pyglabel.draw()