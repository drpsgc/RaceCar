import numpy as np
import matplotlib.pyplot as plt
import math

from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrow


def signed_distance_to_line(p, p1, p2):
    px = p[0]
    py = p[1]
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    dx = x2 - x1
    dy = y2 - y1
    segment_length_sq = dx**2 + dy**2

    if segment_length_sq == 0:
        # The segment is a point
        dist = math.hypot(px - x1, py - y1)
        return dist

    # Vector from A to P
    apx = px - x1
    apy = py - y1

    # Project AP onto AB, normalized by length^2 to get scalar projection
    t = (apx * dx + apy * dy) / segment_length_sq

    if 0 <= t <= 1:
        # Projection falls on the segment
        # Compute perpendicular signed distance to the infinite line
        signed_area = dx * (y1 - py) - dy * (x1 - px)
        perp_dist = signed_area / math.sqrt(segment_length_sq)
        return abs(perp_dist)
    else:
        # Outside the segment, use Euclidean distance to closest endpoint
        dist1 = math.hypot(px - x1, py - y1)
        dist2 = math.hypot(px - x2, py - y2)
        closest_x, closest_y = (x1, y1) if dist1 < dist2 else (x2, y2)
        euclidean_dist = min(dist1, dist2)

        # Use the same signed_area method to determine sign (based on the line)
        signed_area = dx * (y1 - py) - dy * (x1 - px)
        sign = 1 if signed_area >= 0 else 1
        return sign * euclidean_dist

def proj_to_path(x, path, idx0):
    """ 
    Find the closest path sample to x
    """
    N = path.shape[0]

    # Find closest index
    deltas = (path[idx0:, 0:2].T - x[0:2]).T  # shape: (N, 2)
    dist_squared = np.einsum('ij,ij->i', deltas, deltas)  # Fast dot product row-wise
    idx = np.argmin(dist_squared) + idx0
    
    # Find second closest
    d_pre = 1e20 if idx < 1 else signed_distance_to_line(x[0:2], path[idx-1, 0:2], path[idx, 0:2])
    d_post = 1e20 if idx >= N-1 else signed_distance_to_line(x[0:2], path[idx, 0:2], path[idx+1, 0:2])
    
    if d_pre < d_post:
        p1 = path[idx-1, 0:2]
        p2 = path[idx, 0:2]
        s1 = path[idx-1, 4]
        s2 = path[idx, 4]
    else:
        p1 = path[idx, 0:2]
        p2 = path[idx+1, 0:2]
        s1 = path[idx, 4]
        s2 = path[idx+1, 4]
        idx = idx+1

    v1 = p2 - p1
    v2 = (x[0:2,0] - p1)

    norm_v1 = np.linalg.norm(v1)
    proj = (v1.T @ v2) / norm_v1
    proj = max(min(proj, norm_v1),0)
    xy = p1 + proj*v1/norm_v1
    s  = s1 + proj
    
    return xy, s, idx


def get_ref_race(X, N, TRACK):

    N = X.shape[1]
    ref = np.zeros((5,N))
    for i in range(N):
        # Propagate state
        x = X[:,i]
        ref[0:2, i],idx = proj_to_path(x, TRACK)
        ref[2, i] = TRACK[idx,2]
        ref[4, i] = TRACK[idx,3]
    ref[3,:] = 10

    return ref

def get_ref_race_frenet(X, N, TRACK, idx0):
    # Track's state is [x,y,th,k,s]
    # ref is [x,y,th(idx),v,k(idx),s] x,y,s are interpolated
    N = X.shape[1]
    ref = np.zeros((6,N))
    for i in range(N):
        # Propagate state
        x = X[:,i:i+1]
        ref[0:2, i], s, idx = proj_to_path(x, TRACK, 0)
        ref[2, i] = TRACK[idx,2]
        ref[4, i] = TRACK[idx,3]
        ref[5, i] = s
        idx0 = idx # start search from current
    ref[3,:] = 10

    return ref

def get_traj_xy1(d, th, ref):
    # Converts trajectory from d,th to x,y,th given a reference
    # ref is [x,y,th(idx),v,k(idx),s]

    traj = ref[0:3].copy()
    for i in range(1,ref.shape[1]):
        traj[0,i] = ref[0,i] - d[i]*np.sin(ref[2,i])
        traj[1,i] = ref[1,i] + d[i]*np.cos(ref[2,i])
    traj[2,:] = th.squeeze()
    return traj

def get_traj_xy(s, d, th, track, x):
    _, s0, _ = proj_to_path(x, track, 0)
    smax = track[-1,4]
    s = [(si + s0)%smax  for si in s]
    th = th.squeeze()
    indices = np.searchsorted(track[:,4], s, side='left') # return first "greater than" index
    N = indices.shape[0]
    Nt = track.shape[0]

    x = np.zeros(N)
    y = np.zeros(N)
    for i in range(N):
        idx = min(indices[i],Nt-1)
        frac = (s[i]-track[idx-1,4])/(track[idx,4]-track[idx-1,4])

        thp  = (1-frac)*track[idx-1,2] + frac*track[idx,2]
        x[i] = (1-frac)*track[idx-1,0] + frac*track[idx,0] - d[i]*np.sin(thp)
        y[i] = (1-frac)*track[idx-1,1] + frac*track[idx,1] + d[i]*np.cos(thp)

    return np.vstack((x,y,th))

def draw_track(track, w, ax=None):
    """
    Draw track centerline and boundaries.
    track: array of shape (N, 3) [x, y, heading]
    w: track width
    ax: optional matplotlib axis
    """
    if ax is None:
        ax = plt.gca()

    left = np.zeros((track.shape[0], 2))
    right = np.zeros((track.shape[0], 2))
    for i in range(track.shape[0]):
        normal = np.array([
            np.cos(track[i, 2] + np.pi / 2),
            np.sin(track[i, 2] + np.pi / 2)
        ])
        left[i, :] = track[i, 0:2] + w / 2 * normal
        right[i, :] = track[i, 0:2] - w / 2 * normal

    ax.plot(track[:, 0], track[:, 1], 'k', linewidth=1)
    ax.plot(left[:, 0], left[:, 1], 'r--', linewidth=0.8)
    ax.plot(right[:, 0], right[:, 1], 'r--', linewidth=0.8)
    ax.set_aspect('equal')


def draw_vehicle_and_trajectory(ax, x, y, heading, future_states,
                                vehicle_artist=None, trajectory_artist=None,
                                L=4.5, W=2.0):
    """
    Draw or update a vehicle rectangle and trajectory line.
    """
    rear_to_center = L / 2
    cx = x + rear_to_center * np.cos(heading)
    cy = y + rear_to_center * np.sin(heading)

    transform = Affine2D().rotate_around(cx, cy, heading)
    bl_x = cx - L / 2
    bl_y = cy - W / 2

    # vehicle
    if vehicle_artist is None:
        vehicle_artist = Rectangle((bl_x, bl_y), L, W,
                                   facecolor='red', edgecolor='black', alpha=0.8,
                                   transform=transform + ax.transData, zorder=10)
        ax.add_patch(vehicle_artist)
    else:
        vehicle_artist.set_xy((bl_x, bl_y))
        vehicle_artist.set_transform(transform + ax.transData)

    # trajectory
    if future_states is not None:
        traj_x = future_states[0, :]
        traj_y = future_states[1, :]
        if trajectory_artist is None:
            trajectory_artist, = ax.plot(traj_x, traj_y, 'b-', linewidth=1.5, zorder=5)
        else:
            trajectory_artist.set_data(traj_x, traj_y)

    return vehicle_artist, trajectory_artist


def draw_vehicle_grid(fig, ax_main, ax_bicycle,
                      states, future_states,
                      steering, velocity, slipf, slipr,
                      vehicle_artist=None, trajectory_artist=None,
                      tire_artists=None, arrow_artists=None,
                      L=4.5, W=2.0, tire_len=0.8, tire_wid=0.3,
                      Fx_front=0, Fy_front=0, Fx_rear=0, Fy_rear=0, F_maxf=1.0, F_maxr=1.0,
                      ax_front_circle=None, ax_rear_circle=None):
    """
    Draws:
      - main trajectory view on ax_main
      - vertical bicycle model on ax_bicycle
    """
    x, y, psi = states

    # --- Main plot update ---
    vehicle_artist, trajectory_artist = draw_vehicle_and_trajectory(
        ax_main, x, y, psi, future_states,
        vehicle_artist=vehicle_artist,
        trajectory_artist=trajectory_artist,
        L=L, W=W
    )

    # --- Bicycle model plot ---
    lf = 1.5#L / 2
    lr = -1.5#-L / 2
    if tire_artists is None:
        tire_artists = []
        arrow_artists = []

        tire_positions = [
            (0, lr, 'rear'),  # rear tire
            (0, lf, 'front')  # front tire
        ]

        # Connecting chassis line
        ax_bicycle.plot([0, 0], [lr, lf],
                                          'k-', linewidth=3, zorder=0)

        for i, (tx, ty, name) in enumerate(tire_positions):
            tire = Rectangle((tx - tire_wid / 2, ty - tire_len / 2),
                             tire_wid, tire_len,
                             facecolor='black', alpha=0.7)
            ax_bicycle.add_patch(tire)

            # arrows for each tire
            arrow = FancyArrow(tx, ty + 0.05, 0, velocity * 0.2,
                               width=0.1, color='red')
            ax_bicycle.add_patch(arrow)

            tire_artists.append(tire)
            arrow_artists.append(arrow)

        ax_bicycle.set_xlim(-1, 1)
        ax_bicycle.set_ylim(-L / 1.2, L / 1.2)
        ax_bicycle.set_aspect('equal')
        ax_bicycle.set_title("Bicycle Model")
        ax_bicycle.axis('off')

    else:
        # Update tire angles and arrow directions
        positions = [lr, lf]
        slip_angles = [slipr, slipf]

        for i, (tire, arrow) in enumerate(zip(tire_artists, arrow_artists)):
            ty = positions[i]

            # rotate front tire with steering angle
            if i == 1:
                transform = Affine2D().rotate_deg_around(0, ty, np.rad2deg(steering))
            else:
                transform = Affine2D().rotate_deg_around(0, ty, 0)

            tire.set_transform(transform + ax_bicycle.transData)

            # update arrows
            slip_angle = slip_angles[i]
            arrow.remove()
            arrow = FancyArrow(0, ty, velocity * 0.0,
                               velocity * 0.06,
                               width=0.05, color='red')
            transform_arrow = Affine2D().rotate_deg_around(0, ty, np.rad2deg(slip_angle))
            arrow.set_transform(transform_arrow + ax_bicycle.transData)
            ax_bicycle.add_patch(arrow)
            arrow_artists[i] = arrow

    # --- Front friction circle ---
    if ax_front_circle is not None:
        draw_friction_circle(ax_front_circle, Fx_front, Fy_front, F_maxf, title=True)

    # --- Rear friction circle ---
    if ax_rear_circle is not None:
        draw_friction_circle(ax_rear_circle, Fx_rear, Fy_rear, F_maxr)

    return vehicle_artist, trajectory_artist, tire_artists, arrow_artists


def draw_friction_circle(ax, Fx, Fy, F_max, title=False):
    """
    Draw friction circle and force arrow.
    Fx, Fy: longitudinal/lateral force
    F_max: maximum tire force
    """
    ax.clear()
    ax.set_aspect('equal')
    ax.set_xlim(-F_max*1.1, F_max*1.1)
    ax.set_ylim(-F_max*1.1, F_max*1.1)
    ax.set_xlabel("Fy")
    ax.set_ylabel("Fx")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title("Force Circle")
    # ax.axis('off')

    # Circle for max force
    circle = plt.Circle((0,0), radius=F_max, color='k', fill=False, linewidth=1.5)
    ax.add_patch(circle)

    # Resulting force vector
    ax.plot([Fy], [Fx],'o')
    # ax.arrow(0, 0, Fx, Fy)#, head_width=0.05*F_max, head_length=0.1*F_max, fc='r', ec='r', zorder=10)

def plot_colored_line(x, y, c, cmap='RdYlGn_r', linewidth=2, colorbar_label='Value', ax=None):
    """
    Plot a 2D line (x, y) whose color varies according to a third variable c.

    Parameters
    ----------
    x, y : array-like
        Coordinates of the line.
    c : array-like
        Values used to color the line (same length as x and y).
    cmap : str, optional
        Matplotlib colormap name (default 'RdYlGn_r' for red=high, green=low).
    linewidth : float, optional
        Width of the plotted line (default 2).
    colorbar_label : str, optional
        Label for the colorbar.
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, a new figure and axis are created.
    """

    x = np.asarray(x)
    y = np.asarray(y)
    c = np.asarray(c)

    # Create line segments
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Normalize color values
    norm = Normalize(vmin=np.nanmin(c), vmax=np.nanmax(c))

    # Create LineCollection
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(c)
    lc.set_linewidth(linewidth)

    # Create figure/axis if needed
    if ax is None:
        fig, ax = plt.subplots()

    # Add the colored line
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())

    # Add colorbar
    cb = plt.colorbar(lc, ax=ax)
    cb.set_label(colorbar_label)

    return ax

# Savitzky-Golay filter
class SGolay:
    def __init__(self, m, w):

        z = np.arange(-(w-1)/2, (w-1)/2+1)

        JT = np.ones((m+1, w))
        for i in range(m+1):
            JT[i,:] = z**i
        a = np.linalg.solve(JT @ JT.T, JT)

        # final filter coesfficients
        self.c = a[0,:]

    def smooth(self, x):
        return np.array([np.convolve(xi, self.c, mode='same') for xi in x])
