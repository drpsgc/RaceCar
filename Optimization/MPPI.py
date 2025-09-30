#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 23:47:40 2025

@author: psgc
"""
import numpy as np
import utils
from numba import jit, njit, prange

class MPPI():
    
    def __init__(self, f_cont, cost, const_cost, costN = None, use_filter=True):
        
        self.N = 40
        self.K = 2000
        self.alpha = 0.01 # exploration ratio
        self.lambd = 100
        self.gamma = 0.02*self.lambd
        self.dt = 0.1
        self.Sigma = np.diag((0.5, 0.025))
        self.invSigma = np.linalg.inv(self.Sigma)
        
        self.umin = np.array([-3,-1.0])
        self.umax = np.array([ 3, 1.0])
        
        self.nu = 2
        
        self.u_nom = np.zeros((self.nu, self.N))
        
        # Function definitions
        self.f_cont = f_cont
        self.cost = cost
        self.const_cost = const_cost
        self.costN = costN
        
        # Filter
        if use_filter:
            self.filt = utils.SGolay(3, 11)
        else:
            self.filt = None
        
        # use numba
        self.use_numba = True
        
    def solve(self, x0, r):
        
        J = np.zeros((self.K,1))
        
        eps = np.random.multivariate_normal(np.zeros(self.nu), self.Sigma, size=(self.N, self.K))
        eps = eps.reshape(self.nu, self.N, self.K)
        
        # Elitism
        eps[:,:,0] = 0
        # Exploration (eps_new = eps - u_nom -> u = u_nom + esp_new = eps)
        n_explore = int(self.alpha*self.K)
        eps[:,:,1:n_explore+1] -= self.u_nom[:,:,np.newaxis]

        if not self.use_numba:
            J = self.evaluate_trajectories(x0, r, eps, J)
        else:
            J = evaluate_trajectories(x0, r, self.u_nom, eps, self.umin, self.umax, self.dt, J, self.K, self.N, 
                                  self.f_cont, self.cost, self.const_cost, self.costN, self.invSigma, self.gamma, self.alpha)
        
        # Calculate MPPI weights (w is [Kx1])
        Jmin = np.min(J)
        w = np.exp(-(J-Jmin)/self.lambd)
        w /= np.sum(w)
        
        # eps*w.T -> [nu x N x K]*[1 x K] =  [nu x N x K] then sum over 2nd axis [K] to obtain [nu x K]
        u_raw = self.u_nom + np.sum(eps*w.T, axis=2)
        u_sat = np.clip(u_raw, self.umin[:,None], self.umax[:,None])
        
        if self.filt is not None:
            self.u_nom = self.filt.smooth(u_sat)
        else:
            self.u_nom = u_sat

        u_out = np.clip(self.u_nom[:,0], self.umin, self.umax)
        u_out = u_out.reshape(-1,1)
        
        x_pred = self.rollout(x0, self.u_nom)
        
        # Shift input sequence for next iteration
        self.u_nom[:,0:-1] = self.u_nom[:,1:]
        
        return u_out, self.u_nom, x_pred
        
    def evaluate_trajectories(self, x0, r, eps, J):
        for k in range(self.K):
            x = x0.copy()
            J[k]  = 0
            for t in range(self.N):
                # Elitism, explloitation, exploration
                if k == 0:
                    u = self.u_nom[:,t]
                elif k < (1-self.alpha)*self.K:
                    u = self.u_nom[:,t] + eps[:,t,k]
                else:
                    u = eps[:,t,k]
                    
                v = np.clip(u, self.umin, self.umax)
                v = v.reshape(-1,1)
                
                x = x + self.dt*self.f_cont(x, v, r[4,t])
                
                cost_k = self.cost(x, v, r[:,t]) + self.const_cost(x, v, r[4, t]) + self.gamma*self.u_nom[:,t] @ self.invSigma @ u
                J[k] += cost_k.flatten()
                
            J[k] += self.costN if self.costN is not None else 0
        
        return J
    
    def rollout(self, x0, u):
        nx = x0.shape[0]
        x = np.zeros((nx, self.N+1))
        x[:,0:1] = x0
        for t in range(self.N):
            x[:,t+1:t+2] = x[:,t:t+1] + self.dt*self.f_cont(x[:,t:t+1], u[:,t:t+1], 0)
        
        return x

@jit(nopython=True)   
def evaluate_trajectories_old(x0, r, u_nom, eps, umin, umax, dt, J, K, N, f_cont, cost, const_cost, costN, invSigma, gamma, alpha):
    for k in range(K):
        x = x0.copy()
        J[k]  = 0
        for t in range(N):
            u = u_nom[:,t] + eps[:,t,k]
                
            v = np.clip(u, umin, umax)
            v = v.reshape(-1,1)
            
            x = x + dt*f_cont(x, v, r[4,t])
            
            cost_k = cost(x, v, r[:,t]) + const_cost(x, v, r[4, t]) + gamma*u_nom[:,t].T @ invSigma @ v
            J[k] += cost_k.flatten()
            
        J[k] += costN if costN is not None else 0
    
    return J


@njit(parallel=True)
def evaluate_trajectories(x0, r, u_nom, eps, umin, umax, dt, J, K, N, f_cont, cost, const_cost, costN, invSigma, gamma, alpha):
    final_costN = 0.0 if costN is None else costN

    for k in prange(K):
        x = x0.copy()
        J[k] = 0.0
        for t in range(N):
            u = u_nom[:, t] + eps[:, t, k]


            v = np.clip(u, umin, umax).reshape(-1, 1)
            x = x + dt * f_cont(x, v, r[4, t])

            c1 = cost(x, v, r[:, t])
            c2 = const_cost(x, v, r[4, t])
            c3 = gamma * u_nom[:, t].T @ invSigma @ v
            J[k] += (c1 + c2 + c3).flatten()[0]  # Ensure it's a scalar

        J[k] += final_costN

    return J

                