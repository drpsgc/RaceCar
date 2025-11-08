#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 19:35:08 2025

@author: psgc
"""
import numpy as np

def DynVehicleModel(state, u):
    
    Fx = u[0][0]
    delta = u[1][0]
    
    x  = state[0][0]
    y  = state[1][0]
    th = state[2][0]
    vx = state[3][0]
    vy = state[4][0]
    r  = state[5][0]
    
    # Vehicle parameters
    lf = 1.194
    lr = 1.436
    m = 1868.
    Iz = 3049.
    Caf_ = 15.
    Car_ = 25.
    muf = 0.9
    mur = 1.03
    Cd0 = 218.
    Cd2 = 0.4243
    g = 9.81
    
    # Assume no load transfer
    Fzf = m*lr*g/(lf+lr)
    Fzr = m*lf*g/(lf+lr)
    
    # Slip angles
    af = np.arctan2(vy + lf*r, vx) - delta
    ar = np.arctan2(vy - lr*r, vx)
    
    # Forces
    Fxd = Cd0 + Cd2*vx**2
    Fxfmax = np.cos(af)*muf*Fzf
    Fxrmax = np.cos(ar)*mur*Fzr
    Fxf = min(Fx, Fxfmax)
    Fxr = 0.#min(Fx/2, Fxrmax)
    Caf = Caf_*Fzf
    Car = Car_*Fzr
    Fyfmax = np.sqrt((muf*Fzf)**2 - Fxf**2)
    Fyrmax = np.sqrt((mur*Fzr)**2 - Fxr**2)
    Fyf = LatTireForce(Caf, af, Fyfmax)
    Fyr = LatTireForce(Car, ar, Fyrmax)
    
    # EoM
    xd = vx*np.cos(th) - vy*np.sin(th)
    yd = vx*np.sin(th) + vy*np.cos(th)
    thd = r
    vxd = (Fxf*np.cos(delta) - Fyf*np.sin(delta) + Fxr - Fxd)/m + r*vy
    vyd = (Fyf*np.cos(delta) + Fxf*np.sin(delta) + Fyr)/m - r*vx
    rd = (lf*Fyf*np.cos(delta) + lf*Fxf*np.sin(delta) -lr*Fyr)/Iz
    
    state_dot = np.array([[xd], [yd], [thd], [vxd], [vyd], [rd]])
    
    return state_dot
    
def LatTireForce(Caj, aj, Fyjmax, debug=False):
    
    tan_aj_sl = 3*Fyjmax/Caj
    tan_aj = np.tan(aj)
    if abs(tan_aj) < tan_aj_sl:
        Fyj = -Caj*tan_aj + Caj**2*tan_aj*abs(tan_aj)/(3*Fyjmax) - Caj**3*tan_aj**3/(27*Fyjmax**2)
    else:
        if debug:
            print("slipping")
        Fyj = -Fyjmax*np.sign(aj)
    
    return Fyj