#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 27 10:53:35 2025

@author: psgc
"""

# from casadi import *
# from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import time
from MPPI import MPPI
from numba import jit

# Hacky way to import utilities
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

import utils
from racetrack import RaceTrack


def dyn(x, u):
    L = 3
    xdot = np.array([x[3]*np.cos(x[2]), x[3]*np.sin(x[2]), x[3]/L*np.tan(u[1]), u[0]])
    return xdot

@jit(nopython=True)
def f_continuous_filter(x, u, curv):
    L=3
    v   = x[3][0]
    ph  = u[1][0]
    d   = x[1][0]
    th  = x[2][0]
    a   = u[0][0]
    xdot = np.array([[v*np.cos(th)/(1-d*curv)],
                     [v*np.sin(th)],
                     [v/L*np.tan(ph) - (v*curv*np.cos(th))/(1-d*curv)],
                     [a]])

    return xdot

@jit(nopython=True)
def f_continuous(x, u, curv):
    L=3
    v   = x[3][0]
    ph  = x[4][0]
    d   = x[1][0]
    th  = x[2][0]
    dph = u[1][0]
    a   = u[0][0]
    xdot = np.array([[v*np.cos(th)/(1-d*curv)],
                     [v*np.sin(th)],
                     [v/L*np.tan(ph) - (v*curv*np.cos(th))/(1-d*curv)],
                     [a],
                     [dph]])

    return xdot

@jit(nopython=True)
def cost(x, u, r):
    return 0.0*(x[1] - r[1])**2 - 50*x[0] + 0*u.T @ u

@jit(nopython=True)
def constraints_filter(x, u, curv):

    L = 3

    # max lateral deviation
    q1 = 100
    q2 = 10
    f1 = np.abs(x[1]) - 3
    c1 = q1*np.exp(q2*f1)

    # # max lateral acceleration
    # q1 = 100
    # q2 = 10
    # # f2 = abs((tan(u[1])/L)*x[3]**2) - 3
    # f2 = abs(curv*x[3]**2) - 3
    # c2 = q1*exp(q2*f2) if f2 > 0 else 0*f2

    # f1 = abs(x[1]) - 3
    # c1 = 100.*np.maximum(f1,0*f1)

    f2 = np.abs((np.tan(u[1])/L)*x[3]**2) - 3
    # f2 = abs(curv*x[3]**2) - 3 # more conservative, but more stable
    c2 = 1000.*np.maximum(f2,0*f2)

    return c1 + c2

@jit(nopython=True)
def constraints(x, u, curv):

    L = 3

    # max lateral deviation
    q1 = 100
    q2 = 10
    f1 = np.abs(x[1]) - 3
    c1 = q1*np.exp(q2*f1)

    # # max lateral acceleration
    # q1 = 100
    # q2 = 10
    # # f2 = abs((tan(u[1])/L)*x[3]**2) - 3
    # f2 = abs(curv*x[3]**2) - 3
    # c2 = q1*exp(q2*f2) if f2 > 0 else 0*f2

    # f1 = abs(x[1]) - 3
    # c1 = 100.*np.maximum(f1,0*f1)

    f2 = np.abs((np.tan(x[4])/L)*x[3]**2) - 3
    # f2 = abs(curv*x[3]**2) - 3 # more conservative, but more stable
    c2 = 1000.*np.maximum(f2,0*f2)

    # constrain max steering
    q1 = 100
    q2 = 10
    f3 = np.abs(x[4]) - 0.5
    c3 = q1*np.exp(q2*f3)

    return c1 + c2 + c3

def sim(x, u, dt=0.05, M=1):
    # RK4 with M steps
    DT = dt/M
    xf = x.copy()
    for j in range(M):
        k1 = dyn(x,             u)
        k2 = dyn(x + DT/2 * k1, u)
        k3 = dyn(x + DT/2 * k2, u)
        k4 = dyn(x + DT   * k3, u)
        xf += DT/6*(k1   + 2*k2   + 2*k3   + k4)
        # xf[2] = (xf[2] + np.pi) % (2*np.pi) - np.pi # wrap to [-pi pi]
    return xf

def rollout(x0, U, dt):
    x = x0.copy()
    X = [x]
    N = U.shape[1]
    for i in range(N):
        x = sim(x, U[:,i:i+1], dt)
        X.append(x)
    return np.asarray(X).squeeze().T


params = {
    "N": 40,
    "T": 4,
    "Q": [0., 0.0, 0.00, 0.],
    "q": [0., 0., 0., -1.0],
    "R": [1, 200],
    "max_iter": 1
}

# Choose wheter to use a,phi as input with filter or a, dph without filter
use_filter_version = False

nvar = 6
nv = 2*100 + 4*(101)
v0 = np.zeros(nv)
v0[0::6] = 0.5*(np.arange(101))

# CREATE SOLVER
if use_filter_version:
    mppi = MPPI(f_continuous_filter, cost, constraints_filter, use_filter=True)
else:
    mppi = MPPI(f_continuous, cost, constraints, use_filter=False)

start_time = time.perf_counter()

track = RaceTrack()

################### SIMULATION #####################
dt = params["T"]/params["N"]#0.05
L = 3

x = np.array([[0.],[100.],[0],[10.]])
# x = np.array([[-50.],[35],[-3*pi/2],[10.]])
# propagate state with assumed 0 input to obtain v0
x0 = x.copy()
u0 = np.zeros((2,1))
v0[0:4] = x0.flatten()
for k in range(1,params["N"]+1):
    x0 = sim(x0,u0)
    v0[6*k:6*k+4] = x0.flatten()
# v0[2::nvar] = x0[2]
# v0[3::nvar] = x0[3]
v_opt = v0
x1p = v_opt[0::nvar].copy()
x2p = v_opt[1::nvar].copy()
x3p = v_opt[2::nvar].copy()
v_opt[1::nvar] = 0
v_opt[2::nvar] = 0
X = [x]
Xr = []
X1p = []
X2p =[]
U1p = []
U2p = []
Xr1 = []
Xr2 = []
Xr3 = []
Xr4 = []
xf = []
SOLVED = []
U = []
Alat = []
idx0 = 0

fig, ax = plt.subplots()
ax.set_aspect('equal')
vehicle_artist = None
trajectory_artist = None
utils.draw_track(track.track, 6)
phi = 0
# for step in range(475):
for step in range(525):
    t = step*dt
    print(t)

    xx = np.vstack((x1p, x2p, x3p))
    t1 = time.perf_counter()

    xref = utils.get_ref_race_frenet(xx, params["N"]+1, track.track, idx0)
    t2 = time.perf_counter()
    # print(xref[:,1:5])
    # t1 = time.perf_counter()

    # Calculate state and ref in frenet frame
    if use_filter_version:
        x_frenet = np.zeros((4,1))#x.copy()
        dth = x[2] - xref[2,0]
        x_frenet[0] = 0
        x_frenet[1] = -(x[0] - xref[0,0])*np.sin(xref[2,0]) + (x[1] - xref[1,0])*np.cos(xref[2,0])
        x_frenet[2] = dth
        x_frenet[3] = x[3]
        x_ref = xref.copy()
        x_ref[0:2,:] = 0

        u, up, Xpf = mppi.solve(x_frenet, x_ref)

        u1p = up[0,:]
        u2p = up[1,:]
    else:
        x_frenet = np.zeros((5,1))#x.copy()
        dth = x[2] - xref[2,0]
        x_frenet[0] = 0
        x_frenet[1] = -(x[0] - xref[0,0])*np.sin(xref[2,0]) + (x[1] - xref[1,0])*np.cos(xref[2,0])
        x_frenet[2] = dth
        x_frenet[3] = x[3]
        x_frenet[4] = phi
        x_ref = xref.copy()
        x_ref[0:2,:] = 0

        u, up, Xpf = mppi.solve(x_frenet, x_ref)

        u1p = up[0,:]
        u2p = Xpf[4,1:] #up[1,:] # input to "real" system is steering angle
        up = np.vstack((u1p,u2p))
        phi = u2p[0]
        u[1] = phi

    # ROLLOUT dynamics for prediction in cartesian space
    Xp = rollout(x, up, dt)
    
    x1p = Xp[0,:] #+ x[0]
    x2p = Xp[1,:] #+ x[1]
    x3p = Xp[2,:] #+ x[2]
    

    # t2 = time.perf_counter()

    elapsed_time = t2 - t1
    print("solve time: ", elapsed_time)

    xf += [Xpf[0:4,0]]

    a_lat = x[3]**2*( 1/L*np.tan(u[1]))
    Alat += [a_lat]
    
    # SIMULATION
    x = sim(x, u, dt)
    
    X += [x]
    U += [u]
    Xr += [xref[:,0]]

    X1p += [x1p]
    X2p += [x2p]
    U1p += [u1p]
    U2p += [u2p]

    Xr1 += [xref[0,:]]
    Xr2 += [xref[1,:]]
    # Xr3 += [x_ref[2,:]]
    # Xr4 += [x_ref[3,:]]

    # SOLVED += [solved]

    vehicle_artist, trajectory_artist = utils.draw_vehicle_and_trajectory(ax, x[0], x[1], x[2], Xp, vehicle_artist=vehicle_artist, trajectory_artist=trajectory_artist)

    plt.pause(0.01)  # brief pause for animation


end_time = time.perf_counter()

elapsed_time = end_time - start_time


print("Time: ", elapsed_time)

# Retrieve the solution
X = np.asarray(X)
U = np.asarray(U)
Xr = np.asarray(Xr)
X1p = np.asarray(X1p)
X2p = np.asarray(X2p)
U2p = np.asarray(U2p)
# Xr1 = np.asarray(Xr1)
# Xr2 = np.asarray(Xr2)
# Xr3 = np.asarray(Xr3)
# Xr4 = np.asarray(Xr4)
xf = np.asarray(xf)
Alat = np.asarray(Alat)


x1_opt = X[:, 0]
x2_opt = X[:, 1]
x3_opt = X[:, 2]
x4_opt = X[:, 3]
xr1 = Xr[:, 0]
xr2 = Xr[:, 1]
xr3 = Xr[:, 2]
u1_opt = U[:, 0]
u2_opt = U[:, 1]

# print(x1_opt)
# print(x2_opt)
#%%
plt.figure(4)
plt.plot(x4_opt)
plt.title("speed")

# Show prediction
# plt.figure(2)
# at_sample = 5
# # plt.plot(X1p[at_sample,:,0], X2p[at_sample,:,0])
# # plt.plot(Xr1[at_sample,:], Xr2[at_sample,:])
# plt.plot(X2p[at_sample,:])
# plt.plot(Xr2[at_sample,:])
# plt.grid()
# plt.title("Prediction at given sample")

plt.figure(3)
ax = plt.subplot(211)
# ax.plot(Xr3[at_sample,:], label='th_ref_atsamp')
# ax.plot(SOLVED, label='solved')
ax.plot(xr3, label='th_ref')
# ax.plot(xr2, label='y_ref')
# ax.plot(xr1, label='x_ref')
ax.plot(xf[:,1], label='d')#, marker='o')
ax.plot(Alat, label='a_lat')
ax.legend()
plt.subplot(212)
# plt.plot(U2p[at_sample,:,0])
plt.plot()

# Plot the results
plt.figure(1)
plt.clf()
# plt.subplot(211)
plt.plot(x1_opt, x2_opt)#, marker='o')
# plt.plot(xr1, xr2)#, marker='o')
utils.draw_track(track.track, 6)
plt.title("Solution: Gauss-Newton SQP")
plt.xlabel('time')
plt.legend(['x0 trajectory','u trajectory'])
plt.axis('equal')
plt.xlim(-80, 100)  # Set x-axis limits from 0 to 5
plt.ylim(-5, 105) # Set y-axis limits from -1.5 to 1.5
plt.grid()

plt.figure(2)
plt.plot(u1_opt)
plt.plot(u2_opt)
plt.title('Inputs')

# ax = plt.gca()

# T = 10
# N = 100
# plt.subplot(212)
# plt.title("SQP solver output")
# plt.plot(u1_opt)
# plt.plot(u2_opt)
# plt.xlabel('iteration')
# plt.grid()

plt.show()
