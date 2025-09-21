import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import math


class RaceTrack:

    def __init__(self):

        self.track = self.create_track()

    def create_track(self):

        x  = 0
        y  = 100
        th = 0
        X  = [[x,y,th]]

        # straight
        for i in range(60):
            x += 0.5*np.cos(th)
            y += 0.5*np.sin(th)
            th += 0
            X  += [[x,y,th]]
        
        k = -1/20 # curvature
        for i in range(63):
            th += 0.5*k
            x += 0.5*np.cos(th)
            y += 0.5*np.sin(th)
            X  += [[x,y,th]]
        
        # straight
        for i in range(120):
            x += 0.5*np.cos(th)
            y += 0.5*np.sin(th)
            th += 0
            X  += [[x,y,th]]
        
        k = -1/20 # curvature
        for i in range(63):
            th += 0.5*k
            x += 0.5*np.cos(th)
            y += 0.5*np.sin(th)
            X  += [[x,y,th]]
        
        # straight
        for i in range(120):
            x += 0.5*np.cos(th)
            y += 0.5*np.sin(th)
            th += 0
            X  += [[x,y,th]]

        k = -1/20 # curvature
        for i in range(63):
            th += 0.5*k
            x += 0.5*np.cos(th)
            y += 0.5*np.sin(th)
            X  += [[x,y,th]]

        # straight
        for i in range(120):
            x += 0.5*np.cos(th)
            y += 0.5*np.sin(th)
            th += 0
            X  += [[x,y,th]]
        
        return np.asarray(X)
    
# track = RaceTrack()

# plt.plot(track.track[:,0], track.track[:,1])
# plt.axis('equal')
# plt.show()