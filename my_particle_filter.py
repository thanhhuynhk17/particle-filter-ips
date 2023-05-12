# source: https://salzi.blog/2015/05/25/particle-filters-with-python/
from __future__ import annotations

import math
import os
import random
import sys
from copy import copy
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

X_SIZE = 10
Y_SIZE = 23

@dataclass(frozen=True)
class Point:
    x: float = 0.
    y: float = 0.

    def __post_init__(self) -> None:
        if not 0 <= self.x < X_SIZE:
            raise ValueError(f'x = {self.x} is out of bounds')
        if not 0 <= self.y < Y_SIZE:
            raise ValueError(f'y = {self.y} is out of bounds')


@dataclass(frozen=True)
class Noise:
    forward: float = 0.
    turn: float = 0.
    sense: float = 0.


LANDMARKS = (Point(9.9,22.), Point(9.9,13.), Point(9.9, 4.),
            Point(0., 22.), Point(0., 13.), Point(0., 4.))

class RobotState:
    def __init__(self, point: Point = None, angle: float = None,
                noise: Noise = None) -> None:
        self.point = point if point else Point(random.random() * X_SIZE,
                                               random.random() * Y_SIZE)
        self._noise = noise if noise else Noise(0., 0., 0.)

        if angle:
            if not 0 <= angle <= 2 * math.pi:
                raise ValueError(f'Angle must be within [{0.}, {2 * math.pi}, '
                                f'the given value is {angle}]')
        self.angle = angle if angle!=None else random.random() * 2.0 * math.pi
        # self.angle = self.angle - math.pi/2 # heading to the north
    @property
    def point(self) -> Point:
        return self._point

    @point.setter
    def point(self, point: Point) -> None:
        self._point = point

    @property
    def angle(self) -> float:
        return self._angle

    @angle.setter
    def angle(self, value: float) -> None:
        self._angle = float(value)

    def __str__(self) -> str:
        x, y = self.point.x, self.point.y
        return f'x = {x:.3f} y = {y:.3f} angle = {self.angle:.3f}'

    def __copy__(self) -> 'RobotState':
        return type(self)(self.point, self.angle, self._noise)
    
    def _distance(self, landmark: Point) -> float:
        x, y = self.point.x, self.point.y
        dist = (x - landmark.x) ** 2 + (y - landmark.y) ** 2
        return math.sqrt(dist)
    
    def sense(self) -> list[float]:
        return [self._distance(x) + random.gauss(.0, self._noise.sense)
                for x in LANDMARKS]
    
    def move(self, turn: float, forward: float) -> None:
        if forward < 0.:
            raise ValueError('RobotState cannot move backwards')
    
        # turn, and add randomness to the turning command
        # angle = self._angle + turn + random.gauss(0., self._noise.turn)
        angle = turn + random.gauss(0., self._noise.turn)
        angle %= 2 * math.pi
    
        # move, and add randomness to the motion command
        gain = forward + random.gauss(0., self._noise.forward)
        x = self.point.x - math.cos(angle) * gain
        y = self.point.y + math.sin(angle) * gain
    
        self.point = Point(x % X_SIZE, y % Y_SIZE)
        self.angle = angle
    
    @staticmethod
    def gaussian(mu: float, sigma: float, x: float) -> float:
        var = sigma ** 2
        numerator = math.exp(-((x - mu) ** 2) / (2 * var))
        denominator = math.sqrt(2 * math.pi * var)
        return numerator / (denominator + sys.float_info.epsilon)

    def meas_probability(self, measurement: list[float]) -> float:
        prob = 1.
        for ind, landmark in enumerate(LANDMARKS):
            dist = self._distance(landmark)
            prob *= self.gaussian(dist, self._noise.sense, measurement[ind])
        return prob
    


def visualization(ax: any, robot: RobotState, step: int, particles: list[RobotState],
                    particles_resampled: list[RobotState]) -> any:
    ax.set_title('Particle filter, step ' + str(step))

    # draw coordinate grid for plotting
    ax.grid(visible=True, color='0.75', linestyle='--')

    def draw_circle(ax, x_: float, y_: float, face: str, edge: str,
                    alpha: float = 1., size: float = 0.2) -> None:
        circle = plt.Circle(
            (x_, y_), size, facecolor=face, edgecolor=edge, alpha=alpha)
        ax.add_patch(circle)

    def draw_arrow(ax, x_: float, y_: float, angle: float, face: str, edge: str,
                    alpha: float = 1.) -> None:
        ax.arrow(x_, y_, 0.2*math.cos(angle),
                            0.2*math.sin(angle), facecolor=face,
                            edgecolor=edge, alpha=alpha, head_width=0.1)

    # draw particles
    for particle in particles:
        x, y = particle.point.x, particle.point.y
        draw_circle(ax, x, y, '#d9d9d9', '#d9d9d9', 0.4, size=0.1)
        draw_arrow(ax, x, y, particle.angle, '#d9d9d9', '#1a1a1a')

    if particles_resampled != None:
        # draw resampled particles
        for particle in particles_resampled:
            x, y = particle.point.x, particle.point.y
            draw_circle(ax, x, y, '#66ff66', '#009900', 0.4, size=0.1)
            draw_arrow(ax, x, y, particle.angle, '#d9d9d9', '#006600')

    # draw landmarks
    for landmark in LANDMARKS:
        draw_circle(ax, landmark.x, landmark.y, '#cc0000', '#330000', size=0.5)

    # robot's location and angle
    draw_circle(ax, robot.point.x, robot.point.y, '#6666ff', '#0000cc')
    draw_arrow(ax, robot.point.x, robot.point.y, robot.angle, '#000000', '#000000', 1)

    return ax
    # plt.savefig(os.path.join('output', 'figure_' + str(step) + '.png'))
    # plt.close()

def systematic_resample(weights):
    N = len(weights)

    # make N subdivisions, choose positions 
    # with a consistent random offset
    positions = (np.arange(N) + np.random.random()) / N

    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes

def resample_from_index(particles, weights, indexes):
    particles_np = np.array([np.array([p.point.x, p.point.y]) for p in particles])

    particles_np[:] = particles_np[indexes]
    weights[:] = weights[indexes]
    weights /= np.sum(weights)

    return particles_np, weights

def estimate(particles_np, weights):
    """returns mean and variance of the weighted particles"""

    pos = particles_np[:, 0:2]
    mean = np.average(pos, weights=weights, axis=0)
    var  = np.average((pos - mean)**2, weights=weights, axis=0)
    return mean, var