import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import math

# def signed_distance_to_line(p, p1, p2):
#     px = p[0]
#     py = p[1]
#     x1 = p1[0]
#     y1 = p1[1]
#     x2 = p2[0]
#     y2 = p2[1]

#     # Direction vector of the line
#     dx = x2 - x1
#     dy = y2 - y1
    
#     # Normalized perpendicular vector (for the denominator)
#     length = math.hypot(dx, dy)
#     if length == 0:
#         raise ValueError("The two endpoints of the line must be distinct.")
    
#     # Signed distance formula
#     distance = ((dx)*(y1 - py) - (x1 - px)*(dy)) / length
#     return abs(distance)
import math

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

def proj_to_path(x, path):
    """ 
    Find the closest path sample to x
    """
    N = path.shape[0]

    # Find closest index
    dmin = 1e10
    idx = 0
    for i in range(N):
        d = np.linalg.norm(path[i, 0:2] - x[0:2]) 
        if d < dmin:
            dmin = d
            idx = i
    
    # Find second closest
    # d_pre = 1e10 if idx < 1 else np.linalg.norm(path[idx-1, 0:2] - x[0:2])
    # d_post = 1e10 if idx > N-1 else np.linalg.norm(path[idx-1, 0:2] - x[0:2])
    d_pre = 1e10 if idx < 1 else signed_distance_to_line(x[0:2], path[idx-1, 0:2], path[idx, 0:2])
    d_post = 1e10 if idx >= N-1 else signed_distance_to_line(x[0:2], path[idx, 0:2], path[idx+1, 0:2])

    if d_pre < d_post:
        p1 = path[idx-1, 0:2]
        p2 = path[idx, 0:2]
    else:
        p1 = path[idx, 0:2]
        p2 = path[idx+1, 0:2]

    v1 = p2 - p1
    v2 = (x[0:2] - p1).full()

    norm_v1 = np.linalg.norm(v1)
    proj = (v1.T @ v2) / norm_v1
    proj = max(min(proj, norm_v1),0)
    
    return p1 + proj*v1/norm_v1

def next_path_point(x, path):
    """ 
    Find the closest path sample to x
    """
    N = path.shape[0]

    # Find closest index
    dmin = 1e10
    idx = 0
    for i in range(N):
        d = np.linalg.norm(path[i, 0:2] - x[0:2]) 
        if d < dmin:
            dmin = d
            idx = i
    
    # Find second closest
    # d_pre = 1e10 if idx < 1 else np.linalg.norm(path[idx-1, 0:2] - x[0:2])
    # d_post = 1e10 if idx > N-1 else np.linalg.norm(path[idx-1, 0:2] - x[0:2])
    d_pre = 1e10 if idx < 1 else signed_distance_to_line(x[0:2], path[idx-1, 0:2], path[idx, 0:2])
    d_post = 1e10 if idx >= N-1 else signed_distance_to_line(x[0:2], path[idx, 0:2], path[idx+1, 0:2])

    if d_pre < d_post:
        p1 = path[idx-1, 0:2]
        p2 = path[idx, 0:2]
    else:
        p1 = path[idx, 0:2]
        p2 = path[idx+1, 0:2]

    print("x: ", x, "p2: ", p2)
    
    return p2

def get_ref_race(X, N):

    # Heading is a placeholder as it will be recomputed
    # TRACK = [[0, 100, 0, 10], [50, 100, 0, 10], [50, 5, 0, 10], [49.5, 4, 0,10], [47, 2, 0, 10], [46, 0, 0, 10],[45, 0, 0, 10],  [-50, 0, 0, 10], [-50, 100, 0, 10], [0, 100, 0, 10]]
    TRACK = [[0, 100, 0, 10], [50, 100, 0, 10],[50, 0, 0, 10],  [-50, 0, 0, 10], [-50, 100, 0, 10], [0, 100, 0, 10]]
    # TRACK = [[0, 100, 0, 10], [20, 90, 0, 10],[60, 110, 0, 10],  [20, 90, 0, 10], [60, 110, 0, 10], [0, 100, 0, 10]]
    TRACK = np.asarray(TRACK)

    N = X.shape[1]
    ref = np.zeros((4,N))
    for i in range(N):
        # Propagate state
        x = X[:,i]# + ca.vertcat(0.1*np.cos(X[2,i]), 0.1*np.sin(X[2,i]),0)
        ref[0:2, i] = proj_to_path(x, TRACK)
        # ref[0:2, i] = next_path_point(X[:, i], TRACK)
    ref[3,:] = 10

    return ref



# path = np.asarray([[0,0],[1,0],[2,0],[3,0], [3,1],[3,2],[3,3]])
# # x=np.asarray([1.51,1])
# # x = ca.DM([1.51,1])
# x = ca.DM([3.61,-3.9])
# xp = proj_to_path(x,path)


# print(xp)

# plt.plot(path[:,0], path[:,1],marker='o')
# plt.plot(x[0], x[1],marker='o')
# plt.plot(xp[0], xp[1], marker='*')

# plt.show()