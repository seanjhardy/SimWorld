from abc import ABC

import pyglet
from pyglet.gl import *
from gym.envs.classic_control import rendering


# Texture class for vision drawing
class Texture(rendering.Geom):
    def __init__(self, array, x, y, width, height):
        rendering.Geom.__init__(self)
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.texture = self.create_texture_from_ndarray(array)
        self.flip = False

    def create_texture_from_ndarray(self, array):
        """Create a 2D texture from a numpy ndarray."""
        height, width = array.shape[:2]

        texture = pyglet.image.Texture.create_for_size(GL_TEXTURE_2D, width, height,
                                                       GL_RGB32F)
        glBindTexture(texture.target, texture.id)
        glTexImage2D(texture.target, 0, GL_RGB, width, height, 0,
                     GL_RGB, GL_FLOAT, array.ctypes.data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glBindTexture(texture.target, 0)
        return texture

    def render1(self):
        glPushMatrix()
        glTranslatef(self.x, self.y, 0)
        glScalef(self.width / self.texture.width, self.height / self.texture.height, 1.0)
        glColor3f(1.0, 1.0, 1.0)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.texture.id)

        # Draw a textured quad
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0)
        glVertex3f(0, 0, 0)
        glTexCoord2f(0, 1)
        glVertex3f(0, self.texture.height, 0)
        glTexCoord2f(1, 1)
        glVertex3f(self.texture.width, self.texture.height, 0)
        glTexCoord2f(1, 0)
        glVertex3f(self.texture.width, 0, 0)
        glEnd()

        glDisable(GL_TEXTURE_2D)
        glPopMatrix()


# pyglet label
class DrawText(rendering.Geom, ABC):
    def __init__(self, label: pyglet.text.Label):
        super().__init__()
        self.label = label

    def render(self):
        self.label.draw()
