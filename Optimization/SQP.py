#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 18:00:50 2025

@author: psgc
"""

import casadi as ca
import numpy as np

class SQP:

    def __init__(self, param):
        
        self.N = param.get("N", 100) # Horizon samples
        self.T = param.get("T", 10) # Time horizon
        self.M = param.get("M", 1) # Number of discretization steps per timestep
        self.Q = param.get("Q") # Cost (x-xref)'Q(x-xref)
        self.q = param.get("q") # Cost q'x
        self.R = param.get("R") # Cost u'Ru
        self.N_iter = param.get("max_iter", 5) # SQP iterations per time step

        self.J_fcn = None # Casadi cost function
        self.Jac_g_fcn = None # Jacobian/function of constraints
        self.Hess_J_fcn = None # Hessian/Jacobian of cost function
        self.solver = None # Conic QP solver

        self.vmin = None # Lower bounds on decision variables
        self.vmax = None # Upper bounds on decision variables
        self.gmin = None # Lower bounds for g
        self.gmax = None # Upper bounda for g

        # sizes
        self.nx = 4 # number of states
        self.nu = 2 # number of inputs
        self.ns = 2 # number of slacks
        self.nvar = self.nx + self.nu + self.ns # number of decision variables per stage
        self.nv = self.nu*self.N + self.nx*(self.N+1) + self.ns*(self.N+1) # Total decision variables

        # Build solver
        self.build_solver()

        # Linea search
        self.alpha = 0.2

    def build_solver(self):
        # Declare variables for dynamics
        u  = ca.SX.sym("u",self.nu)  # control
        x  = ca.SX.sym("x",self.nx)  # states
        curv = ca.SX.sym("k",1)      # curvature (exogenous)

        # System dynamics
        L = 3
        xdot = ca.vertcat(x[3]*ca.cos(x[2])/(1-x[1]*curv),
                       x[3]*ca.sin(x[2]),
                       x[3]/L*ca.tan(u[1]) - (x[3]*curv*ca.cos(x[2]))/(1-x[1]*curv),
                       u[0])
        f = ca.Function('f',[x,u,curv],[xdot])

        # RK4 with M steps
        U = ca.SX.sym("U",self.nu)
        X = ca.SX.sym("X",self.nx)
        K = ca.SX.sym("K",1)
        DT = self.T/(self.N*self.M)
        XF = X
        for j in range(self.M):
            k1 = f(XF,             U, K)
            k2 = f(XF + DT/2 * k1, U, K)
            k3 = f(XF + DT/2 * k2, U, K)
            k4 = f(XF + DT   * k3, U, K)
            XF += DT/6*(k1   + 2*k2   + 2*k3   + k4)
        F = ca.Function('F',[X,U,K],[XF])

        # Declare variables for optimization problem
        v = ca.SX.sym("v", self.nv)
        xref = ca.SX.sym("x0", 6, self.N + 1) # reference (exogenous)

        # Get the state for each shooting interval
        xk = [v[self.nvar*k : self.nvar*k + self.nx] for k in range(self.N+1)]

        # Get the slacks for each shooting interval
        sk = [v[self.nvar*k + self.nx : self.nvar*k + (self.nx + self.ns)] for k in range(self.N+1)]

        # Get the control for each shooting interval
        uk = [v[self.nvar*k + (self.nx + self.ns) : self.nvar*k + (self.nx + self.ns + self.nu)] for k in range(self.N)]

        # Variable bounds
        self.vmin = -ca.inf*np.ones(self.nv)
        self.vmax =  ca.inf*np.ones(self.nv)

        # Initial solution guess
        v0 = np.zeros(self.nv)

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
            xf = F(xk[k], uk[k], xref[4,k])

            # Add continuity constraints
            g.append(xf - xk[k+1])
            self.gmin += [0, 0, 0, 0]
            self.gmax += [0, 0, 0, 0]

            # Input constraints
            self.vmin[self.nvar*k + self.nx + self.ns : self.nvar*k + self.nx + self.ns + self.nu] = [-3, -np.pi/5]
            self.vmax[self.nvar*k + self.nx + self.ns : self.nvar*k + self.nx + self.ns + self.nu] = [ 3,  np.pi/5]

            # max lateral offset (soft)
            g.append( xk[k][1] - sk[k][0] )
            self.gmin += [-ca.inf]
            self.gmax += [3]
            g.append( xk[k][1] + sk[k][0] )
            self.gmin += [-3]
            self.gmax += [ca.inf]
            
            # max speed
            self.vmax[self.nvar*k + 3] =  20

            # maximum lateral acceleration (soft)
            g.append( (ca.tan(uk[k][1])/L)*xk[k][3]**2 - sk[k][1] )
            self.gmin += [-ca.inf]
            self.gmax += [3]
            g.append( (ca.tan(uk[k][1])/L)*xk[k][3]**2 + sk[k][1] )
            self.gmin += [-3]
            self.gmax += [ca.inf]

            J += (xk[k] - xref[0:4, k]).T @ (Q * (xk[k] - xref[0:4, k])) + uk[k].T @ (R * uk[k]) + 3e1*sk[k].T @ sk[k] + q[3] * xk[k][0]#xk[k][3]*cos(xk[k][2] - xref[2,k])

        # Cost function
        self.J_fcn = ca.Function('J_fcn', [v, xref], [J])

        # Concatenate constraints
        g = ca.vertcat(*g)
        self.gmin = ca.vertcat(*self.gmin)
        self.gmax = ca.vertcat(*self.gmax)

        # Create function for constraints and Jacobian
        self.Jac_g_fcn = ca.Function('J_g_fcn', [v, xref], [ca.jacobian(g, v), g])

        # Hessian/Jacobian function of cost
        self.Hess_J_fcn = ca.Function('H_J_fcn', [v, xref], ca.hessian(J, v))

        # Evalute Hessian of J and gradient of g to extract sparsity patterns
        x_ref = np.zeros((6, self.N + 1)) # Dummy xref
        H_k,_ = self.Hess_J_fcn(ca.DM(v0), x_ref)
        J_g_k = self.Jac_g_fcn(ca.DM(v0), x_ref)[0]

        qp = {
            'h': H_k.sparsity(),
            'a': J_g_k.sparsity(),
        }

        # Create solver
        opts = {'osqp':{'verbose':False, 'max_iter':2000}, 'error_on_fail': False}
        self.solver = ca.conic('solver', 'osqp', qp, opts)

    def solve(self, x_c, v0, xref):
        v_opt = ca.DM(v0)

        # calculate state in Frenet frame [s, d, th-th_path, v]
        x = x_c.copy()
        dth = np.atan2(np.sin(x_c[2] - xref[2,0]), np.cos(x_c[2] - xref[2,0])) # correct for angle wrap
        x[0] = 0
        x[1] = -(x_c[0] - xref[0,0])*np.sin(xref[2,0]) + (x_c[1] - xref[1,0])*np.cos(xref[2,0])
        x[2] = dth
        x[3] = x_c[3]

        x_ref = xref.copy()
        x_ref[0:2,:] = 0
        
        # Initial value
        self.vmin[0] = self.vmax[0] = v0[0] = x[0]
        self.vmin[1] = self.vmax[1] = v0[1] = x[1]
        self.vmin[2] = self.vmax[2] = v0[2] = x[2]
        self.vmin[3] = self.vmax[3] = v0[3] = x[3]

        lam = ca.DM(np.zeros_like(v0))
        mu = ca.DM(np.zeros_like(self.gmin))

        for k in range(self.N_iter):
            # Form linear approximation of constraints
            J_g_k, g_k = self.Jac_g_fcn(v_opt, x_ref)
            
            # Gauss-Newton Hessian and gradient of J
            H_k, Grad_obj_k = self.Hess_J_fcn(v_opt, x_ref)

            # Bounds on delta_v
            dv_min = self.vmin - v_opt
            dv_max = self.vmax - v_opt
            if np.isnan(dv_min).any() or (np.isnan(dv_max)).any() or (dv_max < dv_min).full().any():
                print("nans found")
            
            # Solve the QP
            sol = self.solver(h=H_k, g=Grad_obj_k, a=J_g_k, lbx=dv_min, ubx=dv_max, lba=self.gmin-g_k, uba=self.gmax-g_k)
            dv, lam, mu = sol['x'], sol['lam_x'], sol['lam_a']
            solved = self.solver.stats()['success']

            if max(np.abs(dv)) < 1e-5:
                break

            # Take step with scheduled alpha
            v_opt += self.alpha*dv
            self.alpha = min([1.0,self.alpha+0.2])

        return v_opt.full(), solved, x_ref
