import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import math

# These are used for random track generator
from shapely.geometry import LineString
import random


class RaceTrack:

    def __init__(self, shape='circuit'):

        if shape == 'square':
            self.track = self.create_square()
        if shape == 'circuit':
            self.track = self.create_circuit()
        if shape == 'random':
            self.track = None

    def create_square(self):

        x  = 0
        y  = 100
        th = 0
        k = 0
        X  = [[x,y,th,k]]

        steps_list = [60,    63, 120,    63, 120,    63, 120,    63, 58]
        curv_list  = [ 0, -1/20,   0, -1/20,   0, -1/20,   0, -1/20,  0]

        for steps, k in zip(steps_list, curv_list):
            X += self.create_segment(x, y, th, k, steps)
            x, y, th, _ = X[-1]

        return np.asarray(X)

    def create_circuit(self):

        x  = 0
        y  = 100
        th = 0
        k = 0
        X  = [[x,y,th,k]]

        steps_list = [60,    63,   63,    63, 40,    63, 60,    63, 30,   95, 30,   126, 120,    63, 60]
        curv_list  = [ 0, -1/20, 1/20, -1/20,  0, -1/20,  0, -1/20,  0, 1/15,  0, -1/20,   0, -1/20,  0]

        for steps, k in zip(steps_list, curv_list):
            X += self.create_segment(x, y, th, k, steps)
            x, y, th, _ = X[-1]

        return np.asarray(X)

    def create_random(self, num_segments):
        """
        Creates a random track with the defined number of segments.
        Maximum curvature is 1/15
        Minimum length is 30 steps, maximum is 95 (U-turn with max curvature) to avoid self-intersection as much as possible
        """

        intersects = True
        while intersects:
            x  = 0
            y  = 0
            th = 0
            k = 0
            X  = [[x,y,th,k]]
            for seg in range(num_segments):
                seg_len = random.randint(30,95)
                seg_type = random.randint(0,2) # 0-straight, 1-curve right, 2-curve left
                if seg_type == 0:
                    X += self.create_segment(x, y, th, 0, seg_len)
                if seg_type == 1:
                    k = random.uniform(1/100, 1/15)
                    X += self.create_segment(x, y, th, k, seg_len)
                if seg_type == 2:
                    k = random.uniform(-1/15, -1/100)
                    X += self.create_segment(x, y, th, k, seg_len)
                x, y, th, _ = X[-1]

            X = np.asarray(X)
            x, y = X[:,0], X[:,1]
            coords = list(zip(x, y))
            line = LineString(coords)

            intersects = not line.is_simple

        return X

    def create_segment(self, x, y, th, k, steps, step_size=0.5):
        X = []
        for i in range(steps):
            th += step_size*k
            x += step_size*np.cos(th)
            y += step_size*np.sin(th)
            X  += [[x,y,th,k]]

        return X


# track = RaceTrack("circuit")

# plt.plot(track.track[:,0], track.track[:,1])
# plt.axis('equal')
# plt.show()




# track = RaceTrack("random")

# track1 = track.create_random(10)
# plt.plot(track1[:,0], track1[:,1])
# plt.axis('equal')
# plt.show()
