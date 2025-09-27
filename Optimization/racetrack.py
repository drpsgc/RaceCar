import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import math


class RaceTrack:

    def __init__(self, shape='circuit'):

        if shape == 'square':
            self.track = self.create_square()
        if shape == 'circuit':
            self.track = self.create_circuit()

    def create_square(self):

        x  = 0
        y  = 100
        th = 0
        k = 0
        X  = [[x,y,th,k]]

        # straight
        for i in range(60):
            x += 0.5*np.cos(th)
            y += 0.5*np.sin(th)
            th += 0
            k = 0
            X  += [[x,y,th, k]]

        k = -1/20 # curvature
        for i in range(63):
            th += 0.5*k
            x += 0.5*np.cos(th)
            y += 0.5*np.sin(th)
            X  += [[x,y,th,k]]

        # straight
        for i in range(120):
            x += 0.5*np.cos(th)
            y += 0.5*np.sin(th)
            th += 0
            k = 0
            X  += [[x,y,th,k]]

        k = -1/20 # curvature
        for i in range(63):
            th += 0.5*k
            x += 0.5*np.cos(th)
            y += 0.5*np.sin(th)
            X  += [[x,y,th,k]]

        # straight
        for i in range(120):
            x += 0.5*np.cos(th)
            y += 0.5*np.sin(th)
            th += 0
            k = 0
            X  += [[x,y,th, k]]

        k = -1/20 # curvature
        for i in range(63):
            th += 0.5*k
            x += 0.5*np.cos(th)
            y += 0.5*np.sin(th)
            X  += [[x,y,th, k]]

        # straight
        for i in range(120):
            x += 0.5*np.cos(th)
            y += 0.5*np.sin(th)
            th += 0
            k = 0
            X  += [[x,y,th,k]]

        k = -1/20 # curvature
        for i in range(63):
            th += 0.5*k
            x += 0.5*np.cos(th)
            y += 0.5*np.sin(th)
            X  += [[x,y,th, k]]

        # straight
        for i in range(58):
            x += 0.5*np.cos(th)
            y += 0.5*np.sin(th)
            th += 0
            k = 0
            X  += [[x,y,th,k]]

        return np.asarray(X)

    def create_circuit(self):

        x  = 0
        y  = 100
        th = 0
        k = 0
        X  = [[x,y,th,k]]

        # straight -->
        for i in range(60):
            x += 0.5*np.cos(th)
            y += 0.5*np.sin(th)
            th += 0
            k = 0
            X  += [[x,y,th, k]]

        k = -1/20 # curvature --|
        for i in range(63):
            th += 0.5*k
            x += 0.5*np.cos(th)
            y += 0.5*np.sin(th)
            X  += [[x,y,th,k]]

        k = 1/20 # curvature |--
        for i in range(63):
            th += 0.5*k
            x += 0.5*np.cos(th)
            y += 0.5*np.sin(th)
            X  += [[x,y,th,k]]

        k = -1/20 # curvature -|
        for i in range(63):
            th += 0.5*k
            x += 0.5*np.cos(th)
            y += 0.5*np.sin(th)
            X  += [[x,y,th,k]]

        # straight |
        for i in range(40):
            x += 0.5*np.cos(th)
            y += 0.5*np.sin(th)
            th += 0
            k = 0
            X  += [[x,y,th,k]]

        k = -1/20 # curvature -|
        for i in range(63):
            th += 0.5*k
            x += 0.5*np.cos(th)
            y += 0.5*np.sin(th)
            X  += [[x,y,th,k]]

        # straight <--
        for i in range(60):
            x += 0.5*np.cos(th)
            y += 0.5*np.sin(th)
            th += 0
            k = 0
            X  += [[x,y,th, k]]

        k = -1/20 # curvature
        for i in range(63):
            th += 0.5*k
            x += 0.5*np.cos(th)
            y += 0.5*np.sin(th)
            X  += [[x,y,th,k]]

        # straight |
        for i in range(30):
            x += 0.5*np.cos(th)
            y += 0.5*np.sin(th)
            th += 0
            k = 0
            X  += [[x,y,th, k]]

        k = 1/15 # curvature U turn
        for i in range(95):
            th += 0.5*k
            x += 0.5*np.cos(th)
            y += 0.5*np.sin(th)
            X  += [[x,y,th,k]]

        # straight |
        for i in range(30):
            x += 0.5*np.cos(th)
            y += 0.5*np.sin(th)
            th += 0
            k = 0
            X  += [[x,y,th, k]]

        k = -1/20 # curvature
        for i in range(126):
            th += 0.5*k
            x += 0.5*np.cos(th)
            y += 0.5*np.sin(th)
            X  += [[x,y,th, k]]

        # straight
        for i in range(120):
            x += 0.5*np.cos(th)
            y += 0.5*np.sin(th)
            th += 0
            k = 0
            X  += [[x,y,th,k]]

        k = -1/20 # curvature
        for i in range(63):
            th += 0.5*k
            x += 0.5*np.cos(th)
            y += 0.5*np.sin(th)
            X  += [[x,y,th, k]]

        # straight
        for i in range(60):
            x += 0.5*np.cos(th)
            y += 0.5*np.sin(th)
            th += 0
            k = 0
            X  += [[x,y,th,k]]

        return np.asarray(X)

# track = RaceTrack()

# plt.plot(track.track[:,0], track.track[:,1])
# plt.axis('equal')
# plt.show()