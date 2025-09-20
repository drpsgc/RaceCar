from casadi import *
from numpy import *
import matplotlib.pyplot as plt
import time

import utils


class SQP:

    def __init__(self, param):
        
        self.N = param.get("N", 100) # Horizon samples
        self.T = param.get("T", 10) # Time horizon
        self.M = param.get("M", 1) # Number of discretization steps per timestep
        self.Q = param.get("Q")
        self.q = param.get("q")
        self.R = param.get("R")
        self.N_iter = params.get("max_iter", 25) # SQP iterations per time step

        self.g_fcn = None
        self.Jac_g_fcn = None 
        self.Jac_J_fcn = None
        self.Hess_J_fcn = None
        self.solver = None

        self.vmin = None
        self.vmax = None
        self.gmin = None
        self.gmax = None

        # sizes
        self.nvar = 6
        self.nv = 2*self.N + 4*(self.N+1)

        # Build solver
        self.build_solver()

        # Merit
        self.m_old = 1e20
        self.alpha = 0.2
    
    def build_solver(self):
        # Declare variables (use scalar graph)
        u  = SX.sym("u",2)    # control
        x  = SX.sym("x",4)  # states

        # System dynamics
        L = 3
        xdot = vertcat(x[3]*cos(x[2]), x[3]*sin(x[2]), x[3]/L*tan(u[1]), u[0])
        f = Function('f',[x,u],[xdot])

        # RK4 with M steps
        U = SX.sym("U",2)
        X = SX.sym("X",4)
        DT = self.T/(self.N*self.M)
        XF = X
        QF = 0
        for j in range(self.M):
            k1 = f(XF,             U)
            k2 = f(XF + DT/2 * k1, U)
            k3 = f(XF + DT/2 * k2, U)
            k4 = f(XF + DT   * k3, U)
            XF += DT/6*(k1   + 2*k2   + 2*k3   + k4)
        F = Function('F',[X,U],[XF])

        # Formulate NLP (use matrix graph)
        v = SX.sym("v", self.nv)
        xref = SX.sym("x0", 4, self.N + 1)

        # Get the state for each shooting interval
        xk = [v[self.nvar*k : self.nvar*k + 4] for k in range(self.N+1)]

        # Get the control for each shooting interval
        uk = [v[self.nvar*k + 4 : self.nvar*k + 6] for k in range(self.N)]


        # Variable bounds
        self.vmin = -inf*ones(self.nv)
        self.vmax =  inf*ones(self.nv)

        # Initial solution guess
        v0 = zeros(self.nv)

        # Constraint function with bounds
        g = []; self.gmin = []; self.gmax = []

        # Define weighting matrices for calculations
        Q = np.asarray(self.Q)
        q = np.asarray(self.q).reshape(-1, 1)
        R = np.asarray(self.R)
        J = 0

        # Build up a graph of integrator calls
        for k in range(self.N):
            # Call the integrator
            xf = F(xk[k], uk[k])

            # Append continuity constraints
            g.append(xf - xk[k+1])
            self.gmin += [0, 0, 0, 0]
            self.gmax += [0, 0, 0, 0]

            # Input constraints
            self.vmin[self.nvar*k + 4 : self.nvar*k + 6] = [-3, -np.pi/5]
            self.vmax[self.nvar*k + 4 : self.nvar*k + 6] = [ 3,  np.pi/5]

            # Nonlinear constraints
            th = np.deg2rad(-45)
            Rot = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
            W = np.array([[2/5**2, 0],[0, 2/2**2]])
            W1 = Rot.T @ W @ Rot

            cx = np.array([[-0.5],[0]])
            xx = xk[k][0:2] - cx
            g  += [1 - xx.T @ W1 @ xx]

            self.gmin += [-inf]
            self.gmax += [inf]#[0]

            th = np.deg2rad(30)
            Rot = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
            W = np.array([[2/5**2, 0],[0, 2/2**2]])
            W1 = Rot.T @ W @ Rot

            cx = np.array([[10],[0.5]])
            xx = xk[k][0:2] - cx
            g  += [1 - xx.T @ W1 @ xx]

            self.gmin += [-inf]
            self.gmax += [inf]#[0]

            J += (xk[k] - xref[:, k]).T @ (Q * (xk[k] - xref[:, k])) + uk[k].T @ (R * uk[k]) + q.T @ xk[k]

        # Concatenate constraints
        g = vertcat(*g)
        self.gmin = vertcat(*self.gmin)
        self.gmax = vertcat(*self.gmax)

        # Form function for calculating the constraints
        self.g_fcn = Function('g_fcn',[v],[g])

        # Generate functions for the Jacobians
        self.Jac_g_fcn = Function('J_g_fcn', [v], [jacobian(g, v), g])

        # Dummy xref
        x_ref = np.zeros((4, self.N + 1))

        # Form quadratic approximation of constraints
        Jac_g = self.Jac_g_fcn(DM(v0))
        J_g_k = Jac_g[0]

        # Hessian/Jacobian
        self.Jac_J_fcn = Function('J_J_fcn', [v, xref], [jacobian(J, v)])
        self.Hess_J_fcn = Function('H_J_fcn', [v, xref], [hessian(J, v)[0]])

        H_k = self.Hess_J_fcn(DM(v0), x_ref)

        qp = {
            'h': H_k.sparsity(),
            'a': J_g_k.sparsity(),
        }

        opts = {'osqp':{'verbose':False, 'max_iter':5000}, 'error_on_fail': False}
        self.solver = conic('solver', 'osqp', qp, opts)

    def merit(self, v, r, g):
        rhog = 1
        rhov = 1
        return 0.5*(r.T @ r) + rhov*sum((-fmin(0, v - self.vmin) + fmax(0, v - self.vmax))) + rhog*sum((-fmin(0, g - self.gmin) + fmax(0, g - self.gmax)))
    
    def transformReference(self, x, xref):
        """
            Transforms a cartesian trajectory to ego frame
        """

        x_ref = xref.copy()

        # ego Rotation matrix
        R = np.array([[np.cos(-x[2,0]), -np.sin(-x[2,0])], 
                      [np.sin(-x[2,0]),  np.cos(-x[2,0])]])
        
        x_ref[0:2,:] = R @ (x_ref[0:2,:] - x[0:2])
        # x_ref[2,:] = x_ref[2,:] - x[2]

        # Recompute heading
        # for i in range(x_ref.shape[1]-1):
        # x_ref[2] = (x_ref[2] + np.pi) % (2*np.pi) - np.pi # wrap to [-pi pi]
            # x_ref[2,i] = (x_ref[2,i] + np.pi) % (2*np.pi) - np.pi # wrap to [-pi pi]
        for i in range(x_ref.shape[1]-1):
            x_ref[2,i] = np.atan2(x_ref[1,i+1] - x_ref[1,i], x_ref[0,i+1] - x_ref[0,i])
        x_ref[2,-1] = x_ref[2,-2]

        return x_ref



    def solve(self, x, v0, xref):
        v_opt = DM(v0)

        x_ref = self.transformReference(x, xref)
        
        # Initial value
        self.vmin[0] = self.vmax[0] = v0[0] = 0.#x[0]
        self.vmin[1] = self.vmax[1] = v0[1] = 0.#x[1]
        self.vmin[2] = self.vmax[2] = v0[2] = 0.#x[2]
        self.vmin[3] = self.vmax[3] = v0[3] = x[3]

        lam = DM(zeros_like(v0))
        mu = DM(zeros_like(self.gmin))
        start_time = time.perf_counter()
        for k in range(self.N_iter):
            # Form quadratic approximation of constraints
            Jac_g = self.Jac_g_fcn(v_opt)
            J_g_k = Jac_g[0]
            g_k = Jac_g[1]
            
            # Gauss-Newton Hessian
            H_k = self.Hess_J_fcn(v_opt, x_ref)

            # Gradient of the objective function
            Grad_obj_k = self.Jac_J_fcn(v_opt, x_ref)

            # Bounds on delta_v
            dv_min = self.vmin - v_opt
            dv_max = self.vmax - v_opt
            if isnan(dv_min).any() or (isnan(dv_max)).any() or (dv_max < dv_min).full().any():
                print("nans found")
            
            # Solve the QP
            sol = self.solver(h=H_k, g=Grad_obj_k, a=J_g_k, lbx=dv_min, ubx=dv_max, lba=self.gmin-g_k, uba=self.gmax-g_k)
            dv, d_lam, d_mu = sol['x'], sol['lam_x'], sol['lam_a']
            solved = self.solver.stats()['success']

            # if max(abs(dv)) < 1e-5:
            #     break

            # Take step with scheduled alpha
            v_opt += self.alpha*dv
            lam = d_lam
            mu = d_mu
            self.alpha = min([1.0,self.alpha+0.2])

            # Merit-function-based line search
            # alpha = 1 #if k > 1 else 0.2
            # self.m_old = 1e20
            # while alpha > 1e-3:
            #     v_dv = v_opt + alpha*dv
            #     m_new = self.merit(v_dv, self.r_fcn(v_dv, x_ref), self.g_fcn(v_dv))
            #     if m_new < self.m_old:
            #         break
            #     alpha*= 0.5
            # if alpha < 1e-3:
            #     break
            # v_opt += alpha*dv
            # self.m_old = m_new

        return v_opt, solved, x_ref

def plot_ellipse(cx, cy, W, ax=None, **kwargs):
    """
    Plots a 2D ellipse defined by the quadratic form:
        (x - c)^T W (x - c) = 1
    where W is a 2x2 positive definite matrix and c = [cx, cy]

    Parameters:
    - cx, cy: center coordinates of the ellipse
    - W: 2x2 positive definite matrix
    - ax: optional matplotlib axes object to plot on
    - kwargs: additional keyword arguments passed to plt.plot
    """
    # Check that W is 2x2
    W = np.asarray(W)
    if W.shape != (2, 2):
        raise ValueError("W must be a 2x2 matrix.")

    # Eigen-decomposition of W
    eigvals, eigvecs = np.linalg.eigh(W)

    if np.any(eigvals <= 0):
        raise ValueError("Matrix W must be positive definite.")

    # Get the axes lengths
    axes_lengths = 1.0 / np.sqrt(eigvals)

    # Parametrize the unit circle
    theta = np.linspace(0, 2 * np.pi, 300)
    circle = np.stack((np.cos(theta), np.sin(theta)))

    # Transform the circle to the ellipse
    ellipse = eigvecs @ np.diag(axes_lengths) @ circle
    ellipse[0, :] += cx
    ellipse[1, :] += cy

    # Plot
    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(ellipse[0, :], ellipse[1, :], **kwargs)
    ax.set_aspect('equal')
    ax.set_title("2D Ellipse from Quadratic Form")

    return ax

def generate_straight_trajectory(x0, dt, T):
    x, y, yaw, v = x0
    N = int(T / dt) + 1
    trajectory = np.zeros((4, N))
    time = np.linspace(0, T, N)

    for i in range(N):
        trajectory[:,i:i+1] = np.array([[x], [y], [yaw], [v]])
        # Update state
        x += v * np.cos(yaw) * dt
        y += v * np.sin(yaw) * dt

    return trajectory, time


def get_ref(t, N, dt=0.05):
    T = t + dt * np.arange(N)
    # Heading is a placeholder as it will be recomputed
    TIME =  [              0,               5,             15,               25,                 35,             40]
    TRACK = [[0, 100, 0, 10], [50, 100, 0, 10], [50, 0, 0, 10],  [-50, 0, 0, 10], [-50, 100, 0, 10], [0, 100, 0, 10]]

    TIME = np.asarray(TIME)
    TRACK = np.asarray(TRACK)
    T = np.asarray(T)

    # Ensure shapes: V1 -> (N,), V2 -> (N, n), v1_vals -> (M,)
    assert TRACK.shape[0] == TIME.shape[0], "V2 must match V1 along first axis"

    idx = np.searchsorted(TIME, T, side='right') - 1
    idx = np.clip(idx, 0, len(TIME) - 2)

    x0 = TIME[idx]
    x1 = TIME[idx + 1]
    t = (T - x0) / (x1 - x0)  # shape: (M,)

    y0 = TRACK[idx]      # shape: (M, n)
    y1 = TRACK[idx + 1]  # shape: (M, n)

    # Broadcast t for vector interpolation
    t = t[:, np.newaxis]  # shape: (M, 1)

    ref = (1 - t) * y0 + t * y1  # shape: (M, n)

    return ref.T

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




params = {
    "N": 100,
    "T": 5,
    "Q": [0.001, 0.001, 10, 0.01],
    "q": [0., 0., 0., -0.001],
    "R": [0.01, 0.001]
}

nvar = 6
nv = 2*100 + 4*(101)
v0 = zeros(nv)
v0[0::6] = 0.5*(np.arange(101))
sqp = SQP(params)

start_time = time.perf_counter()

# x_ref, ti = generate_straight_trajectory([-15, 0, 0, 4], params["T"]/params["N"], params["T"])


################### SIMULATION #####################
dt = 0.05
v_opt = v0
x1p = v_opt[0::nvar]
x2p = v_opt[1::nvar]
x3p = v_opt[2::nvar]
x = np.array([[0.],[100.],[0.],[10.]])
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
SOLVED = []
U = []
for step in range(300):
    t = step*dt
    print(t)

    xxref = get_ref(t, params["N"]+1)
    xx = horzcat(x1p, x2p, x3p).T
    xref = utils.get_ref_race(xx, params["N"]+1)
    # print(xref[:,1:5])

    v_opt, solved, x_ref = sqp.solve(x, v_opt, xref)
    # print("x=", v_opt[0:6])
    # print("r=", xref[:,0])

    u = np.array(v_opt[4:6])
    x = sim(x, u)
    X += [x]
    U += [u]
    Xr += [xref[:,0]]

    x1p = v_opt[0::nvar] + x[0]
    x2p = v_opt[1::nvar] + x[1]
    x3p = v_opt[2::nvar] + x[2]
    u1p = v_opt[4::nvar]
    u2p = v_opt[5::nvar]

    X1p += [x1p]
    X2p += [x2p]
    U1p += [u1p]
    U2p += [u2p]

    Xr1 += [xref[0,:]]
    Xr2 += [xref[1,:]]
    Xr3 += [x_ref[2,:]]
    Xr4 += [x_ref[3,:]]

    SOLVED += [solved]


end_time = time.perf_counter()

elapsed_time = end_time - start_time

# Print result
# print("solution found: ", v_opt)
print("Time: ", elapsed_time)
# print("k", k)


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


x1_opt = X[:, 0]
x2_opt = X[:, 1]
x4_opt = X[:, 3]
xr1 = Xr[:, 0]
xr2 = Xr[:, 1]
xr3 = Xr[:, 2]
u1_opt = U[:, 0]
u2_opt = U[:, 1]

# print(x1_opt)
# print(x2_opt)

plt.figure(4)
plt.plot(x4_opt)
plt.title("speed")

# Show prediction
plt.figure(2)
at_sample = 90
plt.plot(X1p[at_sample,:,0], X2p[at_sample,:,0])
plt.plot(Xr1[at_sample,:], Xr2[at_sample,:])
plt.grid()
plt.title("Prediction at given sample")

plt.figure(3)
plt.subplot(211)
plt.plot(Xr3[at_sample,:])
plt.plot(SOLVED)
plt.plot(xr3)
plt.subplot(212)
plt.plot(U2p[at_sample,:,0])
plt.plot()

# Plot the results
plt.figure(1)
plt.clf()
plt.subplot(211)
plt.plot(x1_opt, x2_opt)#, marker='o')
plt.plot(xr1, xr2)
plt.title("Solution: Gauss-Newton SQP")
plt.xlabel('time')
plt.legend(['x0 trajectory','u trajectory'])
plt.grid()


ax = plt.gca()

th = np.deg2rad(-45)
R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
W = np.array([[2/5**2, 0],[0, 2/2**2]])
W1 = R.T @ W @ R
cx = np.array([[-0.5],[0]])

plot_ellipse(cx[0], cx[1], W1, ax)

th = np.deg2rad(30)
R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
W = np.array([[2/5**2, 0],[0, 2/2**2]])
W1 = R.T @ W @ R
cx = np.array([[10],[0.5]])

plot_ellipse(cx[0], cx[1], W1, ax)

T = 10
N = 100
plt.subplot(212)
plt.title("SQP solver output")
plt.step(u1_opt,'-.')
plt.step(u2_opt,'-.')
plt.xlabel('iteration')
plt.grid()

plt.show()
