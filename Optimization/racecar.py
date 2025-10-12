from casadi import *
from numpy import *
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
    xdot = np.array([x[3]*cos(x[2]), x[3]*sin(x[2]), x[3]/L*tan(u[1]), u[0]])
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
nv = 2*params["N"] + 4*(params["N"]+1) + 2*(params["N"]+1) # nu, nx, ns
v0 = zeros(nv)
v0[0::nvar] = 0.5*(np.arange(params["N"]+1))
sqp = SQP(params)

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
    v0[nvar*k : nvar*k + 4] = x0.flatten()
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
T = []

fig, ax = plt.subplots()
ax.set_aspect('equal')
vehicle_artist = None
trajectory_artist = None
utils.draw_track(track.track, 6)
# for step in range(950):
for step in range(475):
    t = step*dt
    # print(t)

    xx = horzcat(x1p, x2p, x3p).T

    xref = utils.get_ref_race_frenet(xx.full(), params["N"]+1, track.track, idx0)

    t1 = time.perf_counter()
    v_opt, solved, x_ref = sqp.solve(x, v_opt, xref)
    t2 = time.perf_counter()
    
    # Have to modify this by converting from prediction to x,y or rolling out input
    # x1p = v_opt[0::nvar] #+ x[0]
    # x2p = v_opt[1::nvar] #+ x[1]
    # x3p = v_opt[2::nvar] #+ x[2]
    u1p = v_opt[nx + ns::nvar]
    u2p = v_opt[nx + ns + 1::nvar]
    up = np.array(horzcat(u1p, u2p).T)
    Xp = rollout(x, up, dt)
    x1p = Xp[0,:] #+ x[0]
    x2p = Xp[1,:] #+ x[1]
    x3p = Xp[2,:] #+ x[2]

    elapsed_time = t2 - t1
    T += [elapsed_time]
    print("solve time: ", elapsed_time)

    xf += [v_opt[0:4]]
    u = np.array(v_opt[nx+ns:nx+ns+nu])
    a_lat = x[3]**2*( 1/L*tan(u[1]))
    Alat += [a_lat]
    
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
    Xr3 += [x_ref[2,:]]
    Xr4 += [x_ref[3,:]]

    SOLVED += [solved]

    vehicle_artist, trajectory_artist = utils.draw_vehicle_and_trajectory(ax, x[0], x[1], x[2], Xp, vehicle_artist=vehicle_artist, trajectory_artist=trajectory_artist)

    plt.pause(0.01)  # brief pause for animation


print("max time: ", max(T))
print("mean time: ", mean(T))
# Retrieve the solution
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

# print(x1_opt)
# print(x2_opt)
#%%
plt.figure(4)
plt.plot(x4_opt)
plt.title("speed")

# Show prediction
plt.figure(2)
at_sample = 5
# plt.plot(X1p[at_sample,:,0], X2p[at_sample,:,0])
# plt.plot(Xr1[at_sample,:], Xr2[at_sample,:])
plt.plot(X2p[at_sample,:])
plt.plot(Xr2[at_sample,:])
plt.grid()
plt.title("Prediction at given sample")

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
plt.plot(U2p[at_sample,:,0])
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


# ax = plt.gca()

plt.figure(5)
plt.title("control inputs")
plt.plot(u1_opt)
plt.plot(u2_opt)
plt.grid()

plt.show()
