import math
import numpy as np
from numpy import cos, sin, pi
from gym.envs.classic_control import rendering
from utils import *


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


    def translate(self, dist, angle):
        x, y, theta, _ = self._position
        self.set_position(x + dist * cos(angle), y + dist * sin(angle), theta)


    def set_transform(self):
        x, y, theta = self.get_position() 
        self._transform = Part.get_transform(x, y, theta)


    def get_position(self):
        if self.parent is None:
            return self._position[:-1]
        else:
            pos = np.reshape(self._position, (4,1))
            return (self.parent._transform * pos)[:-1]

        
    def part_distances(self, parts):
        '''
        Returns the distances and angles to the specified list of parts.
        '''
        distances = []
        for part in parts:
            if part != self:
                d, alpha = self.part_distance(part)
                distances.append((d, alpha, part))
        return distances
        

    def part_distance(self, part):
        '''
        Calculates the distance and relative angle to the specified part.
        '''
        x1, y1, th1 = self.get_position()
        x2, y2, th2 = part.get_position()
        d, alpha = self.distance(x2, y2)
        alpha2 = th2 - th1 + alpha 
        bd2 = part.boundary_distance(alpha2)
        return d - bd2, alpha
        

    def distance(self, x, y):
        '''
        Calculates the distance and relative angle to the specified location.
        '''
        x1, y1, th1 = self.get_position()
        dx = x - x1
        dy = y - y1
        dc = math.sqrt(dx ** 2 + dy ** 2)
        alpha = canonical_angle(math.atan2(dy, dx) - th1)
        bd = self.boundary_distance(alpha)
        return dc - bd, alpha
        

    def boundary_distance(self, alpha):
        '''
        Calculates the part's boundary distance in the specified direction.
        '''
        return 0.0


    def is_collided(self, part):
        dist, angle = self.part_distance(part)
        return dist <= 0.


    def add_part(self, part, x, y, theta):
        if self._transform is None:
            self.set_transform()
        part.set_position(x, y, theta)
        part.parent = self
        self.parts.append(part)
        return part


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



class RectangularPart(Part):

    def __init__(self, length, width):
        Part.__init__(self)
        self.length = length
        self.width = width
        self._diag_angle = math.atan2(width, length)

        self.front_left = self.add_part(Part(), self.length / 2., self.width / 2., 0.)
        self.front_right = self.add_part(Part(), self.length / 2., -self.width / 2., 0.)
        self.back_left = self.add_part(Part(), -self.length / 2., self.width / 2., 0.)
        self.back_right = self.add_part(Part(), -self.length / 2., -self.width / 2., 0.)


    def boundary_distance(self, alpha):
        '''
        Calculates the rectangle's boundary distance in the specified direction.
        '''
        l = self.length / 2.0
        w = self.width / 2.0
        alpha = abs(wrap(alpha, -right_angle, right_angle))
        if alpha <= self._diag_angle:
            return l / cos(alpha)
        else:
            return w / sin(alpha)

