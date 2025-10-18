import numpy as np
import matplotlib.pyplot as plt
import time
from SQP import SQP

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
    "N": 50,
    "T": 5,
    "Q": [0., 0., 0, 0.],
    "q": [0., 0., 0., -2.0],
    "R": [1, 200],
    "max_iter": 1
}

nx = 4
nu = 2
ns = 2
nvar = nx + nu + ns
nv = nu*params["N"] + nx*(params["N"]+1) + ns*(params["N"]+1) # nu, nx, ns
v0 = np.zeros(nv)
v0[0::nvar] = 0.5*(np.arange(params["N"]+1))
sqp = SQP(params)

start_time = time.perf_counter()

track = RaceTrack()

################### SIMULATION #####################
dt = params["T"]/params["N"]#0.05
L = 3

x = np.array([[0.],[100.],[0],[10.]])
# propagate state with assumed 0 input to obtain v0
x0 = x.copy()
u0 = np.zeros((2,1))
v0[0:nx] = x0.flatten()
for k in range(1,params["N"]+1):
    x0 = sim(x0,u0)
    v0[nvar*k : nvar*k + nx] = x0.flatten()
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
T = []

# Prepare plots
fig, ax = plt.subplots()
ax.set_aspect('equal')
vehicle_artist = None
trajectory_artist = None
utils.draw_track(track.track, 6)

for step in range(675):
    t = step*dt

    # x,y,theta trajectory in global coordinates
    xx = np.vstack((x1p, x2p, x3p))

    xref = utils.get_ref_race_frenet(xx, params["N"]+1, track.track, idx0)

    t1 = time.perf_counter()
    v_opt, solved, x_ref = sqp.solve(x, v_opt, xref)
    t2 = time.perf_counter()
    
    # Prediction from SQP is in Frenet frame
    # Calculate x,y,theta by rolling out input trajectory
    # (alternatively could be done geometrically)
    u1p = v_opt[nx + ns::nvar]
    u2p = v_opt[nx + ns + 1::nvar]
    up = np.concat((u1p, u2p),axis=1).T
    Xp = rollout(x, up, dt)
    x1p = Xp[0,:]
    x2p = Xp[1,:]
    x3p = Xp[2,:]

    elapsed_time = t2 - t1
    T += [elapsed_time]
    print("solve time: ", elapsed_time)

    # Save trajectories for visualization
    u = np.array(v_opt[nx+ns : nx+ns+nu])

    # Simulate race car
    x = sim(x, u, dt)

    # Save trajectories for visualization
    xf += [v_opt[0:4]]
    a_lat = x[3]**2*( 1/L*np.tan(u[1]))
    Alat += [a_lat]
    
    X += [x]
    U += [u]
    Xr += [xref[:,0]]

    X1p += [x1p]
    X2p += [x2p]
    U1p += [u1p]
    U2p += [u2p]

    Xr1 += [xref[0,:]]
    Xr2 += [xref[1,:]]
    Xr3 += [x_ref[2,:]]
    Xr4 += [x_ref[3,:]]

    SOLVED += [solved]

    vehicle_artist, trajectory_artist = utils.draw_vehicle_and_trajectory(ax, x[0], x[1], x[2], Xp, vehicle_artist=vehicle_artist, trajectory_artist=trajectory_artist)

    plt.pause(0.01)  # brief pause for animation
    if xref[5,0] >= track.track[-2,4]:
        print("FINISHED! Lap time: ", dt*step)
        break


print("max time: ", max(T))
print("mean time: ", np.mean(T))

# Convert trajectories to numpy
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
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, num=2)
ax1.plot(t2, x4_opt, label='speed')
ax2.plot(t1, xf[:,1], label='d')
ax2.plot([t1[0],t1[-1]],[-3, -3],'r--',linewidth=1)
ax2.plot([t1[0],t1[-1]],[ 3,  3],'r--',linewidth=1)
ax3.plot(t1, Alat, label='a_lat')
ax3.plot([t1[0],t1[-1]],[-3, -3],'r--',linewidth=1)
ax3.plot([t1[0],t1[-1]],[ 3,  3],'r--',linewidth=1)
ax4.plot(t1, SOLVED, label='solved')
for ax in (ax1, ax2, ax3, ax4):
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
