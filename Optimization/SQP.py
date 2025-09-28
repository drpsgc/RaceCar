#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 18:00:50 2025

@author: psgc
"""

from casadi import *
from numpy import *

import utils


class SQP:

    def __init__(self, param):
        
        self.N = param.get("N", 100) # Horizon samples
        self.T = param.get("T", 10) # Time horizon
        self.M = param.get("M", 1) # Number of discretization steps per timestep
        self.Q = param.get("Q")
        self.q = param.get("q")
        self.R = param.get("R")
        self.N_iter = param.get("max_iter", 25) # SQP iterations per time step

        self.J_fcn = None
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
        
        # Regularization
        self.reg = 1e-2
    
    def build_solver(self):
        # Declare variables (use scalar graph)
        u  = SX.sym("u",2)    # control
        x  = SX.sym("x",4)  # states
        curv = SX.sym("k",1) # curvature
        reg = SX.sym("reg",1) # regularization factor

        # System dynamics
        L = 3
        # xdot = vertcat(x[3]*cos(x[2]), x[3]*sin(x[2]), x[3]/L*tan(u[1]), u[0])
        xdot = vertcat(x[3]*cos(x[2])/(1-x[1]*curv), 
                       x[3]*sin(x[2]), 
                       x[3]/L*tan(u[1]) - (x[3]*curv*cos(x[2]))/(1-x[1]*curv), 
                       u[0])
        f = Function('f',[x,u,curv],[xdot])

        # RK4 with M steps
        U = SX.sym("U",2)
        X = SX.sym("X",4)
        K = SX.sym("K",1)
        DT = self.T/(self.N*self.M)
        XF = X
        QF = 0
        for j in range(self.M):
            k1 = f(XF,             U, K)
            k2 = f(XF + DT/2 * k1, U, K)
            k3 = f(XF + DT/2 * k2, U, K)
            k4 = f(XF + DT   * k3, U, K)
            XF += DT/6*(k1   + 2*k2   + 2*k3   + k4)
        F = Function('F',[X,U,K],[XF])

        # Formulate NLP (use matrix graph)
        v = SX.sym("v", self.nv)
        xref = SX.sym("x0", 5, self.N + 1)

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
            xf = F(xk[k], uk[k], xref[4,k])

            # Append continuity constraints
            g.append(xf - xk[k+1])
            self.gmin += [0, 0, 0, 0]
            self.gmax += [0, 0, 0, 0]

            # Input constraints
            self.vmin[self.nvar*k + 4 : self.nvar*k + 6] = [-3, -np.pi/5]
            self.vmax[self.nvar*k + 4 : self.nvar*k + 6] = [ 3,  np.pi/5]

            # State constraints
            # max lateral distance
            self.vmax[self.nvar*k + 1] =  3
            self.vmin[self.nvar*k + 1] = -3
            
            # max speed
            self.vmax[self.nvar*k + 3] =  20

            # maximum lateral acceleration
            g.append( (tan(uk[k][1])/L)*xk[k][3]**2 )
            self.gmin += [-3]
            self.gmax += [3]

            J += (xk[k] - xref[0:4, k]).T @ (Q * (xk[k] - xref[0:4, k])) + uk[k].T @ (R * uk[k]) + q[3] * xk[k][0]#xk[k][3]*cos(xk[k][2] - xref[2,k])

        # Cost function
        self.J_fcn = Function('J_fcn', [v, xref], [J])

        # Concatenate constraints
        g = vertcat(*g)
        self.gmin = vertcat(*self.gmin)
        self.gmax = vertcat(*self.gmax)

        # Form function for calculating the constraints
        self.g_fcn = Function('g_fcn',[v, xref],[g])

        # Generate functions for the Jacobians
        self.Jac_g_fcn = Function('J_g_fcn', [v, xref], [jacobian(g, v), g])

        # Dummy xref
        x_ref = np.zeros((5, self.N + 1))

        # Form quadratic approximation of constraints
        Jac_g = self.Jac_g_fcn(DM(v0), x_ref)
        J_g_k = Jac_g[0]

        # Hessian/Jacobian
        self.Jac_J_fcn = Function('J_J_fcn', [v, xref], [jacobian(J, v)])
        self.Hess_J_fcn = Function('H_J_fcn', [v, xref], [hessian(J, v)[0]])

        H_k = self.Hess_J_fcn(DM(v0), x_ref)

        qp = {
            'h': H_k.sparsity(),
            'a': J_g_k.sparsity(),
        }

        opts = {'osqp':{'verbose':False, 'max_iter':2000}, 'error_on_fail': False}
        self.solver = conic('solver', 'osqp', qp, opts)

    def merit(self, v, J, g):
        rhog = 1
        rhov = 1
        return J + rhov*sum((-fmin(0, v - self.vmin) + fmax(0, v - self.vmax))) + rhog*sum((-fmin(0, g - self.gmin) + fmax(0, g - self.gmax)))

    def solve(self, x_c, v0, xref):
        v_opt = DM(v0)

        # calculate state in frenet frame [s, d, th-th_path, v]
        x = x_c.copy()
        dth = x_c[2] - xref[2,0]
        x[0] = 0
        x[1] = -(x_c[0] - xref[0,0])*sin(xref[2,0]) + (x_c[1] - xref[1,0])*cos(xref[2,0])
        x[2] = dth
        x[3] = x_c[3]

        x_ref = xref.copy()# self.transformReference(x, xref)
        x_ref[0:2,:] = 0
        
        # Initial value
        self.vmin[0] = self.vmax[0] = v0[0] = x[0]
        self.vmin[1] = self.vmax[1] = v0[1] = x[1]
        self.vmin[2] = self.vmax[2] = v0[2] = x[2]
        self.vmin[3] = self.vmax[3] = v0[3] = x[3]

        lam = DM(zeros_like(v0))
        mu = DM(zeros_like(self.gmin))

        for k in range(self.N_iter):
            # Form quadratic approximation of constraints
            Jac_g = self.Jac_g_fcn(v_opt, x_ref)
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
            # while alpha > 1e-7:
            #     v_dv = v_opt + alpha*dv
            #     m_new = self.merit(v_dv, self.J_fcn(v_dv, x_ref), self.g_fcn(v_dv, x_ref))
            #     if m_new < self.m_old:
            #         self.reg /= 10
            #         break
            #     alpha*= 0.5
            # if alpha < 1e-6:
            #     alpha = 0
            #     self.reg *= 10
            #     break
            # v_opt += alpha*dv
            # self.m_old = m_new

        return v_opt, solved, x_ref