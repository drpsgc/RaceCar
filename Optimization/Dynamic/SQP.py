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
        self.P = param.get("P", self.Q) # Terminal cost (x-xref)'P(x-xref)
        self.q = param.get("q") # Cost q'x
        self.R = param.get("R") # Cost u'Ru
        self.N_iter = param.get("max_iter", 5) # SQP iterations per time step
        self.Hess_approx = param.get("Hess_approx", "GN") # Hessian approximation: Newton, GN
        self.disc_method = param.get("disc_method", "rk4") # Hessian approximation: Newton, GN

        self.J_fcn = None # Casadi cost function
        self.Jac_g_fcn = None # Jacobian/function of constraints
        self.Hess_J_fcn = None # Hessian/Jacobian of cost function
        self.solver = None # Conic QP solver

        self.vmin = None # Lower bounds on decision variables
        self.vmax = None # Upper bounds on decision variables
        self.gmin = None # Lower bounds for g
        self.gmax = None # Upper bounda for g

        # sizes
        self.nx = 6 # number of states
        self.nu = 2 # number of inputs
        self.ns = 3 # number of slacks
        self.nvar = self.nx + self.nu + self.ns # number of decision variables per stage
        self.nv = self.nu*self.N + self.nx*(self.N+1) + self.ns*(self.N+1) # Total decision variables

        # Linea search
        self.alpha = 0.1
        self.alpha_max = 0.3 if self.Hess_approx == "GN" else 0.5

        # Symbols
        self.symX = ca.MX if self.Hess_approx == "Newton" else ca.SX

        # Vehicle parameters
        self.lf = 1.194
        self.lr = 1.436
        self.L = self.lf + self.lr
        self.m = 1868.
        self.Iz = 3049.
        self.Caf = 15.
        self.Car = 25.
        self.muf = 0.9
        self.mur = 1.03
        self.Cd0 = 218.
        self.Cd2 = 0.4243
        self.g = 9.81
        
        # Build solver
        self.build_solver()

    def build_solver(self):
        # Declare variables for dynamics
        u  = self.symX.sym("u",self.nu)  # control
        x  = self.symX.sym("x",self.nx)  # states
        curv = self.symX.sym("k",1)      # curvature (exogenous)

        ###### System dynamics
        Fx = u[0]
        delta = u[1]
        
        s  = x[0]
        d  = x[1]
        th = x[2]
        vx = x[3]
        vy = x[4]
        r  = x[5]

        # Assume no load transfer
        Fzf = self.m*self.lr*self.g/self.L
        Fzr = self.m*self.lf*self.g/self.L

        # Slip angles
        af = np.arctan2(vy + self.lf*r, vx) - delta
        ar = np.arctan2(vy - self.lr*r, vx)

        # Forces
        Fxd = self.Cd0 + self.Cd2*vx**2
        Fxfmax = ca.cos(af)*self.muf*Fzf
        Fxrmax = ca.cos(ar)*self.mur*Fzr
        Fxf = 1000.*Fx/2#ca.fmin(Fx/2, Fxfmax)
        Fxr = 1000.*Fx/2#ca.fmin(Fx/2, Fxrmax)
        Caf = self.Caf*Fzf
        Car = self.Car*Fzr
        Fyfmax = np.sqrt((self.muf*Fzf)**2 - Fxf**2)
        Fyrmax = np.sqrt((self.mur*Fzr)**2 - Fxr**2)

        # Lateral Tire forces
        Caj  = self.symX.sym("Caj",1)
        aj  = self.symX.sym("aj",1)
        Fyjmax  = self.symX.sym("Fyjmax",1)
        tan_aj_sl = 3*Fyjmax/Caj
        tan_aj = ca.tan(aj)
        Fyj = ca.if_else(ca.fabs(tan_aj) <= tan_aj_sl,
                         -Caj*tan_aj + Caj**2*tan_aj*ca.fabs(tan_aj)/(3*Fyjmax) - Caj**3*tan_aj**3/(27*Fyjmax**2),
                         -Fyjmax*ca.sign(aj))
        Fy_fcn = ca.Function("Fy_fcn",[Caj, aj, Fyjmax], [Fyj])
        Fyf = Fy_fcn(Caf, af, Fyfmax)
        Fyr = Fy_fcn(Car, ar, Fyrmax)

        # EoM
        sd = (vx*ca.cos(th) - vy*ca.sin(th))/(1 - curv*d)
        dd = vx*ca.sin(th) + vy*ca.cos(th)
        thd = r - curv*sd
        vxd = (Fxf*ca.cos(delta) - Fyf*ca.sin(delta) + Fxr - Fxd)/self.m + r*vy
        vyd = (Fyf*ca.cos(delta) + Fxf*ca.sin(delta) + Fyr)/self.m - r*vx
        rd = (self.lf*Fyf*ca.cos(delta) + self.lf*Fxf*ca.sin(delta) - self.lr*Fyr)/self.Iz
        
        xdot = ca.vertcat(*[sd, dd, thd, vxd, vyd, rd])

        f = ca.Function('f',[x,u,curv],[xdot])
        ######

        # RK4 with M steps
        U = self.symX.sym("U",self.nu)
        X = self.symX.sym("X",self.nx)
        K = self.symX.sym("K",1)
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
        v = self.symX.sym("v", self.nv)
        xref = self.symX.sym("x0", 7, self.N + 1) # reference (exogenous)

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
        v0 = np.ones(self.nv)

        # Constraint function with bounds
        g = []; self.gmin = []; self.gmax = []

        # Define weighting matrices for calculations
        Q = np.asarray(self.Q)
        P = np.asarray(self.P)
        q = np.asarray(self.q).reshape(-1, 1)
        R = np.asarray(self.R)
        J = 0

        # Build up a graph of integrator calls
        for k in range(self.N):

            # Add continuity constraints
            if self.disc_method == "rk4":
                xf = F(xk[k], uk[k], xref[6,k])
                g.append(xf - xk[k+1])
            elif self.disc_method == "trapezoidal":
                xi = f(xk[k], uk[k], xref[6,k])
                if k < self.N - 1:
                    xi1 = f(xk[k+1], uk[k+1], xref[6,k+1])
                else:
                    xi1 = f(xk[k+1], uk[k], xref[6,k+1])
                g.append(xk[k+1] - xk[k] - (DT/2)*(xi + xi1))

            self.gmin += [0, 0, 0, 0, 0, 0]
            self.gmax += [0, 0, 0, 0, 0, 0]

            # Input constraints
            self.vmin[self.nvar*k + self.nx + self.ns : self.nvar*k + self.nx + self.ns + self.nu] = [-5, -np.pi/5]
            self.vmax[self.nvar*k + self.nx + self.ns : self.nvar*k + self.nx + self.ns + self.nu] = [ 5,  np.pi/5]

            # max lateral offset (soft)
            g.append( xk[k][1] - sk[k][0] )
            self.gmin += [-ca.inf]
            self.gmax += [3]
            g.append( xk[k][1] + sk[k][0] )
            self.gmin += [-3]
            self.gmax += [ca.inf]
            
            # max speed
            # self.vmax[self.nvar*k + 3] =  20

            # maximum lateral force (soft)
            Fxks = 1000.*uk[k][0]
            afk = np.arctan2(xk[k][4] + self.lf*xk[k][5], xk[k][3]) - uk[k][1]
            ark = np.arctan2(xk[k][4] - self.lr*xk[k][5], xk[k][3])
            Fyfmaxk = np.sqrt((self.muf*Fzf)**2 - Fxks**2/4)
            Fyrmaxk = np.sqrt((self.mur*Fzr)**2 - Fxks**2/4)
            Fyfk = Fy_fcn(Caf, afk, Fyfmaxk)
            Fyrk = Fy_fcn(Car, ark, Fyrmaxk)
            g.append( (Fxks**2/4 + Fyfk**2)/(self.muf**2*Fzf**2) - sk[k][1] )
            self.gmin += [-ca.inf]
            self.gmax += [0.81]
            g.append( (Fxks**2/4 + Fyrk**2)/(self.mur**2*Fzr**2) - sk[k][2])
            self.gmin += [-ca.inf]
            self.gmax += [0.81]

            J += (xk[k] - xref[0:6, k]).T @ (Q * (xk[k] - xref[0:6, k])) + uk[k].T @ (R * uk[k]) + 5e1*sk[k].T @ sk[k] + q[3] * xk[k][0]#xk[k][3]*cos(xk[k][2] - xref[2,k])
        k = self.N
        J += (xk[k] - xref[0:6, k]).T @ (P * (xk[k] - xref[0:6, k]))# + 5e0*sk[k].T @ sk[k] + q[3] * xk[k][0]

        # Cost function
        self.J_fcn = ca.Function('J_fcn', [v, xref], [J])

        # Concatenate constraints
        g = ca.vertcat(*g)
        self.gmin = ca.vertcat(*self.gmin)
        self.gmax = ca.vertcat(*self.gmax)

        # Create function for constraints and Jacobian
        self.Jac_g_fcn = ca.Function('J_g_fcn', [v, xref], [ca.jacobian(g, v), g],{'jit': True})

        # Create Lagrangian function
        ng = g.shape[0]
        mu = self.symX.sym("mu",ng)

        # Hessian/Jacobian function of cost
        if self.Hess_approx == "Newton":
            L = J + mu.T @ g
            HL_, _ = ca.hessian(L, v)
            HL = ca.convexify(HL_, {'strategy':'eigen-reflect'})
            self.Hess_L_fcn = ca.Function('H_L_fcn', [v, mu, xref], [HL],{'jit': True})
        elif self.Hess_approx == "GN":
            L = J
            self.Hess_L_fcn = ca.Function('H_L_fcn', [v, mu, xref], [ca.hessian(L, v)[0]],{'jit': True})
        self.grad_J_fcn = ca.Function('grad_J_fcn', [v, xref], [ca.jacobian(J, v)],{'jit': True})

        # Evalute Hessian of L and gradient of g to extract sparsity patterns
        x_ref = np.zeros((7, self.N + 1)) # Dummy xref
        lam0 = ca.DM.ones((ng,1))
        H_k = self.Hess_L_fcn(ca.DM(v0), lam0, x_ref)
        J_g_k = self.Jac_g_fcn(ca.DM(v0), x_ref)[0]

        qp = {
            'h': H_k.sparsity(),
            'a': J_g_k.sparsity(),
        }

        # Create solver
        opts = {'osqp':{'verbose':False, 'max_iter':2000}, 'error_on_fail': False}
        self.solver = ca.conic('solver', 'osqp', qp, opts)

    def solve(self, x_c, v0, mu0, xref):
        v0[:-self.nx-self.nu-self.ns] = v0[self.nx+self.nu+self.ns:]
        mu0[:-self.nx-4] = mu0[self.nx+4:]
        v_opt = ca.DM(v0)

        # calculate state in Frenet frame [s, d, th-th_path, vx, vy, r_e]
        x = x_c.copy()
        dth = np.atan2(np.sin(x_c[2] - xref[2,0]), np.cos(x_c[2] - xref[2,0])) # correct for angle wrap
        curv = xref[4,0]
        x[0] = 0
        x[1] = -(x_c[0] - xref[0,0])*np.sin(xref[2,0]) + (x_c[1] - xref[1,0])*np.cos(xref[2,0])
        x[2] = dth
        x[3] = x_c[3]
        x[4] = x_c[4]
        sdot = (x_c[3]*np.cos(dth) - x_c[4]*np.sin(dth))/(1-curv*x[1])
        x[5] = x_c[5] - curv*sdot
        
        # [x,y,th(idx),v,k(idx),s] x,y,s are interpolated
        
        x_ref = np.zeros((self.nx+1,xref.shape[1]))
        x_ref[0:2,:] = 0 # s,d
        x_ref[3,:] = 18
        x_ref[4,:] = 0 # vy
        x_ref[5,:] = x_ref[3,:] * xref[4,:] # yawrate = v*k
        x_ref[6,:] = xref[4,:] # k
        
        # Initial value
        self.vmin[0] = self.vmax[0] = v0[0] = x[0]
        self.vmin[1] = self.vmax[1] = v0[1] = x[1]
        self.vmin[2] = self.vmax[2] = v0[2] = x[2]
        self.vmin[3] = self.vmax[3] = v0[3] = x[3]
        self.vmin[4] = self.vmax[4] = v0[4] = x[4]
        self.vmin[5] = self.vmax[5] = v0[5] = x[5]
        
        # v_opt = ca.DM(v0)
        lam = ca.DM(np.zeros_like(v0))
        mu = ca.DM(mu0)
        # mu = ca.DM(np.zeros_like(self.gmin))

        for k in range(self.N_iter):
            # Form linear approximation of constraints
            # print(v_opt)
            J_g_k, g_k = self.Jac_g_fcn(v_opt, x_ref)
            # maxg = np.max(J_g_k)
            # print(maxg)

            # Newton Hessian of L and gradient of J
            H_k = self.Hess_L_fcn(v_opt, mu, x_ref)

            # Convexify
            # H_k = 0.5*(H_k + H_k.T)
            # eigvals, eigvecs = np.linalg.eigh(H_k)
            # H_k = eigvecs @ np.diag(np.maximum(eigvals, 1e-6)) @ eigvecs.T

            Grad_obj_k = self.grad_J_fcn(v_opt, x_ref)
            # print(np.max(Grad_obj_k))

            # Bounds on delta_v
            dv_min = self.vmin - v_opt
            dv_max = self.vmax - v_opt
            if np.isnan(dv_min).any() or (np.isnan(dv_max)).any() or (dv_max < dv_min).full().any():
                print("nans found")
            if np.isnan(H_k).any() or np.isnan(Grad_obj_k).any() or np.isnan(J_g_k).any() or np.isnan(g_k).any():
                print("NaNs found")
            
            # Solve the QP
            sol = self.solver(h=H_k, g=Grad_obj_k, a=J_g_k, lbx=dv_min, ubx=dv_max, lba=self.gmin-g_k, uba=self.gmax-g_k)
            dv, lam, dmu = sol['x'], sol['lam_x'], sol['lam_a']
            solved = self.solver.stats()['success']
            # print("solved? ", solved)
            # if max(np.abs(dv)) < 1e-5:
            #     break
            if not np.isnan(dv).any() and solved:
                # Take step with scheduled alpha
                v_opt += self.alpha*dv
                mu = dmu
                self.alpha = min([self.alpha_max,self.alpha+0.2])

            # Scale input 0 longitudinal force
            v_opt[self.nvar*k + self.nx + self.ns] *= 1000.

        return v_opt.full(), solved, mu, x_ref
