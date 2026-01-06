import numpy as np
import matplotlib.pyplot as plt
import time
import imageio.v2 as imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
from SQP import SQP

# Hacky way to import utilities
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'utils')))

import utils
from racetrack import RaceTrack
from VehicleModel import DynVehicleModel, LatTireForce


## SETTINGS
save_animation = False

def dyn(x, u):
    L = 3
    xdot = np.array([x[3]*np.cos(x[2]), x[3]*np.sin(x[2]), x[3]/L*np.tan(u[1]), u[0]])
    return xdot

def sim(x, u, dt=0.05, M=1):
    # RK4 with M steps
    DT = dt/M
    xf = x.copy()
    for j in range(M):
        k1 = DynVehicleModel(x,             u)
        k2 = DynVehicleModel(x + DT/2 * k1, u)
        k3 = DynVehicleModel(x + DT/2 * k2, u)
        k4 = DynVehicleModel(x + DT   * k3, u)
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
    "N": 150,
    "T": 7.5,
    "Q": [0., 0.0, 0, 0.0, 0, 0.0, 25, 5],
    "P": [0., 0.0, 0., 0.0, 0, 0.0, 25, 5],
    "q": [-200., 0., 0., -0., 0., 0., 0., 0.],
    "R": [0.0025, 0.05],
    "max_iter": 1,
    "Hess_approx": "Newton",
    "disc_method": "trapezoidal",
    "M": 1
}
nxs = 6
nx = 8
nu = 2
ns = 1
ne = 0
nvar = nx + nu + ns + ne
nv = nu*params["N"] + nx*(params["N"]+1) + ns*(params["N"]+1) + ne*(params["N"]+1) # nu, nx, ns
v0 = np.zeros(nv)
v0[0::nvar] = 0.5*(np.arange(params["N"]+1))
sqp = SQP(params)

start_time = time.perf_counter()

track = RaceTrack("circuit")
# track.track = track.create_random(20)

################### SIMULATION #####################
dt = params["T"]/params["N"]#0.05
L = 3

x = np.array([[0.],[103.],[0],[10.],[0.],[0.]])
# propagate state with assumed 0 input to obtain v0
x0 = x.copy()
u0 = np.zeros((2,1))
v0[0:nxs] = x0.flatten()
for k in range(1,params["N"]+1):
    x0 = sim(x0,u0)
    v0[nvar*k : nvar*k + nxs] = x0.flatten()
v_opt = v0
x1p = v_opt[0::nvar].copy()
x2p = v_opt[1::nvar].copy()
x3p = v_opt[2::nvar].copy()
v_opt[0::nvar] /= 100.
v_opt[1::nvar] = 3
v_opt[2::nvar] = 0
v_opt[3::nvar] /= 20.
mu = np.zeros_like(sqp.gmin)
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
Af = []
Ar = []
idx0 = 0
T = []

# Prepare plots
fig = plt.figure(figsize=(10, 5))
gs = plt.GridSpec(4, 6, figure=fig)
ax_main = fig.add_subplot(gs[:, :4])
ax_bicycle = fig.add_subplot(gs[:, 4])
ax_front_circle = fig.add_subplot(gs[0:2, 5])  # top right
ax_rear_circle = fig.add_subplot(gs[2:4, 5])   # bottom right
agg_canvas = FigureCanvasAgg(fig)
# ax.set_aspect('equal')
vehicle_artist = None
trajectory_artist = None
tire_artists = None
arrow_artists = None
utils.draw_track(track.track, 6, ax=ax_main)
frames_filenames = []
for step in range(450*2):
    t = step*dt

    # x,y,theta trajectory in global coordinates
    xx = np.vstack((x1p, x2p, x3p))

    xref = utils.get_ref_race_frenet(xx, params["N"]+1, track.track, idx0)

    t1 = time.perf_counter()
    u, v_opt, solved, mu, x_ref = sqp.solve(x, v_opt, mu, xref)
    t2 = time.perf_counter()
    
    # Save trajectories for visualization
    u1p = v_opt[nx - 2::nvar]
    u2p = v_opt[nx - 1::nvar]
    # u = np.array(v_opt[nvar+6:nvar+8])

    # Prediction from SQP is in Frenet frame
    # Convert to xy
    sp = v_opt[0::nvar]*100.
    dp = v_opt[1::nvar]
    thp = v_opt[2::nvar]
    Xp = utils.get_traj_xy(sp, dp, thp, track.track, x)

    xx  = x[0][0]
    yx  = x[1][0]
    thx = x[2][0]
    vxx = x[3][0]
    vyx = x[4][0]
    rx  = x[5][0]

    # Simulate race car
    x = sim(x, u, dt, M=4)

    x1p = Xp[0,:]
    x2p = Xp[1,:]
    x3p = Xp[2,:]

    elapsed_time = t2 - t1
    T += [elapsed_time]
    print("solve time: ", elapsed_time)

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


    ### ANIMATION
    # Calculate tire forces for animation
    Fx = u[0][0]
    delta = u[1][0]
    Fzf = sqp.m*sqp.lr*sqp.g/sqp.L
    Fzr = sqp.m*sqp.lf*sqp.g/sqp.L

    # Slip angles
    af = np.arctan2(vyx + sqp.lf*rx, vxx) - delta
    ar = np.arctan2(vyx - sqp.lr*rx, vxx)

    # Forces
    Fxfmax = np.cos(af)*sqp.muf*Fzf
    Fxrmax = np.cos(ar)*sqp.mur*Fzr
    Fxf = min(Fx, Fxfmax) if Fx > 0 else min(0.75*Fx, Fxfmax)
    Fxr = 0.              if Fx > 0 else min(0.25*Fx, Fxrmax)
    Caf = sqp.Caf*Fzf
    Car = sqp.Car*Fzr
    Fyfmax = np.sqrt((sqp.muf*Fzf)**2 - Fxf**2)
    Fyrmax = np.sqrt((sqp.mur*Fzr)**2 - Fxr**2)
    Fyf = LatTireForce(Caf, af, Fyfmax,True)
    Fyr = LatTireForce(Car, ar, Fyrmax,True)

    Af += [af]
    Ar += [ar]
    vehicle_artist, trajectory_artist, tire_artists, arrow_artists = utils.draw_vehicle_grid(fig, ax_main, ax_bicycle, states=(xx, yx, thx), future_states=Xp, steering=delta, velocity=vxx, slipf=-af, slipr=-ar,
                          vehicle_artist=vehicle_artist, trajectory_artist=trajectory_artist,
                          tire_artists=tire_artists, arrow_artists=arrow_artists,
                          L=4.5, W=2.0, tire_len=0.8, tire_wid=0.3,
                          Fx_front=Fxf, Fy_front=Fyf, Fx_rear=Fxr, Fy_rear=Fyr, F_maxf=sqp.muf*Fzf, F_maxr=sqp.mur*Fzr,
                          ax_front_circle=ax_front_circle, ax_rear_circle=ax_rear_circle)

    if save_animation:
        # --- save frame as PNG ---
        filename = f"frames/frame_{step:03d}.png"
        fig.savefig(filename, dpi=150)
        frames_filenames.append(filename)

        plt.pause(0.0001)  # brief pause for animation

    ## LAP END CONDITION
    if xref[5,2] >= track.track[-2,4]:
        print("FINISHED! Lap time: ", dt*step)
        break

# --- combine PNGs into GIF ---
if save_animation:
    images = []
    target_size = (1000, 500)  # width, height in pixels, fixed figure size

    for filename in frames_filenames:
        img = Image.open(filename)
        img = img.resize(target_size)  # force same dimensions
        images.append(np.array(img))

    imageio.mimsave("trajectory2.gif", images, fps=int(1/dt), loop=0)

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
Af = np.asarray(Af)
Ar = np.asarray(Ar)

x1_opt = X[:, 0]
x2_opt = X[:, 1]
x3_opt = X[:, 2]
x4_opt = X[:, 3]
xr1 = Xr[:, 0]
xr2 = Xr[:, 1]
xr3 = Xr[:, 2]
u1_opt = U[:, 0]
u2_opt = U[:, 1]


#%% PLOTS
legend_loc = 'upper right'
t1 = dt*np.arange(step+1)
t2 = dt*np.arange(step+2)

# Track and trajectory plot
plt.figure(1)
plt.clf()
ax1 = plt.gca()
utils.draw_track(track.track, 6)
utils.plot_colored_line(x1_opt, x2_opt, x4_opt.squeeze(), cmap='viridis', ax=ax1)
plt.title("Newton SQP")
plt.axis('equal')
plt.xlim(-80, 100)
plt.ylim(-5, 105)
plt.grid()

# Plot relevant variables
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, num=2, sharex=True)
ax1.plot(t2, x4_opt, label='speed')
ax2.plot(t1, xf[:,1], label='d')
ax2.plot([t1[0],t1[-1]],[-3, -3],'r--',linewidth=1)
ax2.plot([t1[0],t1[-1]],[ 3,  3],'r--',linewidth=1)
ax3.plot(t1, Alat, label='a_lat')
ax4.plot(t1, SOLVED, label='solved')
for ax in (ax1, ax2, ax3, ax4):
    ax.legend(loc=legend_loc)
    ax.grid(True)


# Control input plots
fig, (ax1, ax2) = plt.subplots(2, 1, num=3, sharex=True)
fig.suptitle("control inputs")
ax1.plot(t1, u1_opt, label='acceleration')
ax1.plot([t1[0],t1[-1]],[-7500, -7500],'r--',linewidth=1)
ax1.plot([t1[0],t1[-1]],[ 5000,  5000],'r--',linewidth=1)
ax1.grid(True)
ax1.legend(loc=legend_loc)
ax2.plot(t1, u2_opt, label='steering angle')
ax2.plot([t1[0],t1[-1]],[-np.pi/5, -np.pi/5],'r--',linewidth=1)
ax2.plot([t1[0],t1[-1]],[ np.pi/5,  np.pi/5],'r--',linewidth=1)
ax2.grid(True)
ax2.legend(loc=legend_loc)


# Slip angles
fig, (ax1, ax2) = plt.subplots(2, 1, num=4, sharex=True)
fig.suptitle("slip angles")
ax1.plot(t1, 57.3*Af, label='front')
# ax1.plot([t1[0],t1[-1]],[-5000, -5000],'r--',linewidth=1)
# ax1.plot([t1[0],t1[-1]],[ 5000,  5000],'r--',linewidth=1)
ax1.grid(True)
ax1.legend(loc=legend_loc)
ax2.plot(t1, 57.3*Ar, label='rear')
# ax2.plot([t1[0],t1[-1]],[-np.pi/5, -np.pi/5],'r--',linewidth=1)
# ax2.plot([t1[0],t1[-1]],[ np.pi/5,  np.pi/5],'r--',linewidth=1)
ax2.grid(True)
ax2.legend(loc=legend_loc)


# Show prediction
# plt.figure(5)
# at_sample = step - 5
# plt.plot(Xr1[at_sample,:], label='x_ref')
# plt.plot(Xr2[at_sample,:], label='y_ref')
# plt.plot(Xr3[at_sample,:], label='th_ref')
# plt.plot(Xr4[at_sample,:], label='v_ref')
# plt.legend(loc=legend_loc)
# plt.grid()
# plt.title("Trajectory at given sample")



plt.show()
