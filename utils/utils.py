import numpy as np
import matplotlib.pyplot as plt
import math

from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


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

def draw_track(track, w):
    left = np.zeros((track.shape[0],2))
    right = np.zeros((track.shape[0],2))
    for i in range(track.shape[0]):
        normal = np.array([np.cos(track[i,2]+np.pi/2) , np.sin(track[i,2]+np.pi/2)])
        left[i,:] = track[i,0:2] + w/2*normal
        right[i,:] = track[i,0:2] - w/2*normal
    
    plt.plot(track[:,0], track[:,1],'k', linewidth=1)
    plt.plot(left[:,0], left[:,1], 'r--')
    plt.plot(right[:,0], right[:,1],'r--')
    

from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

def draw_vehicle_and_trajectory(ax, x, y, heading, future_states,
                                vehicle_artist=None, trajectory_artist=None,
                                L=4.5, W=2.0):
    """
    Draw or update a vehicle rectangle and trajectory line.

    Parameters:
    - ax: matplotlib Axes
    - x, y, heading: rear axle position and heading in radians
    - future_states: np.array shape (3, N) of future x, y, heading
    - vehicle_artist: existing rectangle (for reuse)
    - trajectory_artist: existing line (for reuse)
    - L, W: length and width of vehicle

    Returns:
    - Updated vehicle_artist and trajectory_artist
    """

    # Compute center of rectangle (vehicle center is forward from rear axle)
    rear_to_center = L / 2
    cx = x + rear_to_center * np.cos(heading)
    cy = y + rear_to_center * np.sin(heading)

    # Define transform for rotation + translation around center
    transform = Affine2D().rotate_around(cx, cy, heading)

    # Bottom-left corner before rotation (centered rectangle)
    bl_x = cx - L / 2
    bl_y = cy - W / 2

    if vehicle_artist is None:
        # Create the rectangle at (bl_x, bl_y) with identity transform
        vehicle_artist = Rectangle((bl_x, bl_y), L, W,
                                   facecolor='blue', edgecolor='black', alpha=0.8,
                                   transform=transform + ax.transData,
                                   zorder=10)
        ax.add_patch(vehicle_artist)
    else:
        # Update position and transform
        vehicle_artist.set_xy((bl_x, bl_y))
        vehicle_artist.set_transform(transform + ax.transData)

    # Update trajectory
    if future_states is not None:
        traj_x = future_states[0, :]
        traj_y = future_states[1, :]

        if trajectory_artist is None:
            trajectory_artist, = ax.plot(traj_x, traj_y, linewidth=1.5, zorder=5)
        else:
            trajectory_artist.set_data(traj_x, traj_y)

    return vehicle_artist, trajectory_artist


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
