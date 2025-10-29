#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 27 10:53:35 2025

@author: psgc
"""

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
    return 0.0*(x[1] - r[1])**2 - 75*x[0] + 0*u.T @ u

@jit(nopython=True)
def constraints_filter(x, u, curv):

    L = 3

    # max lateral deviation
    q1 = 100
    q2 = 10
    f1 = np.abs(x[1]) - 3
    c1 = q1*np.exp(q2*f1)

    # max lateral acceleration
    f2 = np.abs((np.tan(u[1])/L)*x[3]**2) - 3
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

    # max lateral acceleration

    f2 = np.abs((np.tan(x[4])/L)*x[3]**2) - 3
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
    "K": 2000,
    "Sigma": [0.5, 0.025],
}

# Choose wheter to use a,phi as input with filter or a, dph without filter
use_filter_version = False

nx = 4
nu = 2
nvar = nx + nu
nv = nu*params["N"] + 4*(params["N"]+1)
v0 = np.zeros(nv)
v0[0::nvar] = 0.5*(np.arange(params["N"]+1))

# CREATE SOLVER
if use_filter_version:
    mppi = MPPI(f_continuous_filter, cost, constraints_filter, params, use_filter=True)
else:
    mppi = MPPI(f_continuous, cost, constraints, params, use_filter=False)

start_time = time.perf_counter()

track = RaceTrack()

################### SIMULATION #####################
dt = params["T"]/params["N"]#0.05
L = 3

x = np.array([[0.],[100.],[0],[10.]])
# propagate state with assumed 0 input to obtain v0
x0 = x.copy()
u0 = np.zeros((2,1))
v0[0:4] = x0.flatten()
for k in range(1,params["N"]+1):
    x0 = sim(x0,u0)
    v0[nvar*k:nvar*k+nx] = x0.flatten()

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
T = []
idx0 = 0

fig, ax = plt.subplots()
ax.set_aspect('equal')
vehicle_artist = None
trajectory_artist = None
utils.draw_track(track.track, 6)
phi = 0
# for step in range(475):
for step in range(625):
    t = step*dt

    xx = np.vstack((x1p, x2p, x3p))

    xref = utils.get_ref_race_frenet(xx, params["N"]+1, track.track, idx0)

    t1 = time.perf_counter()
    # Calculate state and ref in frenet frame
    if use_filter_version:
        x_frenet = np.zeros((4,1))
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
        x_frenet = np.zeros((5,1))
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
        u2p = Xpf[4,1:] # input to "real" system is steering angle
        up = np.vstack((u1p,u2p))
        phi = u2p[0]
        u[1] = phi

    t2 = time.perf_counter()
    elapsed_time = t2 - t1
    T += [elapsed_time]
    print("solve time: ", elapsed_time)

    # ROLLOUT dynamics for prediction in cartesian space
    Xp = rollout(x, up, dt)
    
    x1p = Xp[0,:] #+ x[0]
    x2p = Xp[1,:] #+ x[1]
    x3p = Xp[2,:] #+ x[2]

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
    Xr3 += [xref[2,:]]
    Xr4 += [xref[3,:]]

    vehicle_artist, trajectory_artist = utils.draw_vehicle_and_trajectory(ax, x[0], x[1], x[2], Xp, vehicle_artist=vehicle_artist, trajectory_artist=trajectory_artist)

    plt.pause(0.01)  # brief pause for animation

    if xref[5,0] >= track.track[-5,4]:
        print("FINISHED! Lap time: ", dt*step)
        break

print("max time: ", max(T))
print("mean time: ", np.mean(T))


#%% Retrieve the solution
X = np.asarray(X)
U = np.asarray(U)
Xr = np.asarray(Xr)
X1p = np.asarray(X1p)
X2p = np.asarray(X2p)
U2p = np.asarray(U2p)
Xr1 = np.asarray(Xr1)
Xr2 = np.asarray(Xr2)
Xr3 = np.asarray(Xr3)
Xr4 = np.asarray(Xr4)
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

#%% Plots

legend_loc = 'upper right'
t1 = dt*np.arange(step+1)
t2 = dt*np.arange(step+2)

# Track and trajectory plot
plt.figure(1)
plt.clf()
ax1 = plt.gca()
utils.draw_track(track.track, 6)
utils.plot_colored_line(x1_opt, x2_opt, x4_opt.squeeze(), cmap='viridis', ax=ax1)
plt.title("Gauss-Newton SQP")
plt.axis('equal')
plt.xlim(-80, 100)
plt.ylim(-5, 105)
plt.grid()

# Plot relevant variables
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, num=2)
ax1.plot(t2, x4_opt, label='speed')
ax2.plot(t1, xf[:,1], label='d')
ax2.plot([t1[0],t1[-1]],[-3, -3],'r--',linewidth=1)
ax2.plot([t1[0],t1[-1]],[ 3,  3],'r--',linewidth=1)
ax3.plot(t1, Alat, label='a_lat')
ax3.plot([t1[0],t1[-1]],[-3, -3],'r--',linewidth=1)
ax3.plot([t1[0],t1[-1]],[ 3,  3],'r--',linewidth=1)
for ax in (ax1, ax2, ax3):
    ax.legend(loc=legend_loc)
    ax.grid(True)


# Control input plots
fig, (ax1, ax2) = plt.subplots(2, 1, num=3)
fig.suptitle("control inputs")
ax1.plot(t1, u1_opt, label='acceleration')
ax1.grid(True)
ax1.legend(loc=legend_loc)
ax2.plot(t1, u2_opt, label='steering angle')
ax2.grid(True)
ax2.legend(loc=legend_loc)


# Show prediction
plt.figure(4)
at_sample = 470
plt.plot(Xr1[at_sample,:], label='x_ref')
plt.plot(Xr2[at_sample,:], label='y_ref')
plt.plot(Xr3[at_sample,:], label='th_ref')
plt.plot(Xr4[at_sample,:], label='v_ref')
plt.legend(loc=legend_loc)
plt.grid()
plt.title("Trajectory at given sample")



plt.show()

