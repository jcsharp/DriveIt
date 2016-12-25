import math
import numpy as np
from numpy import cos, sin, pi
from gym.envs.classic_control import rendering


class Part():

    def __init__(self):
        self.parent = None
        self.parts = []
        self._transform = None
        self._render_transforms = None
        self.set_position(0., 0., 0.)


    def set_position(self, x, y, theta):
        self._position = np.array([x, y, theta, 1])
        if len(self.parts) > 0:
            self.set_transform()


    def set_rotation(self, theta):
        x, y, _, _ = self._position
        self.set_position(x, y, theta)


    def set_transform(self):
        self._transform = Part.get_transform(*self._position)


    def get_position(self):
        if self.parent is None:
            return self._position[:-1]
        else:
            return (self.parent.transform * self._position)[:-1]

        
    def add_part(self, part, x, y, theta):
        if self._transform is None:
            self.set_transform()
        part.set_position(x, y, theta)
        part.parent = self
        self.parts.append(part)


    def get_rotation(a):
        ca = cos(a)
        sa = sin(a)
        return np.matrix([\
            [ ca, sa, 0., 0.], \
            [-sa, ca, 0., 0.], \
            [ 0., 0., 1., a ], \
            [ 0., 0., 0., 1.]])


    def get_translation(x, y):
        return np.matrix([\
            [1., 0., 0., x ], \
            [0., 1., 0., y ], \
            [0., 0., 1., 0.], \
            [0., 0., 0., 1.]])


    def get_transform(x, y, theta, unit=1.):
        translation = Part.get_translation(x, y)
        rotation = Part.get_rotation(theta)
        transform = translation * rotation
        return transform


    def get_geometry(self):
        return []


    def render(self, viewer):

        if self._render_transforms is None:
            self._render_transform = rendering.Transform()
            self._render_transforms = [self._render_transform]
            if not self.parent is None:
                self._render_transforms.extend(self.parent._render_transforms)
            for g in self.get_geometry():
                viewer.add_geom(g)
                for trans in self._render_transforms:
                    g.add_attr(trans)

        x, y, theta, _ = self._position
        self._render_transform.set_translation(x, y)
        self._render_transform.set_rotation(theta)

        for part in self.parts:
            part.render(viewer)

