#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 17:59:32 2025

@author: psgc
"""

import agent_torch
import numpy as np
import torch
import matplotlib.pyplot as plt

# Hacky way to import utilities
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

import utils
import racetrack

### Environment
class Racecar:
    def __init__(self, x0):
        self.L = 3
        self.x = x0
        self.max_d = 1.5 # max tolerated lateral error (normalized), if exceeded episode terminates
        self.steps = 0
        self.dt = 0.05;
        
        # Random track generator
        self.track_generator = racetrack.RaceTrack("random")
        self.episode = 0
        self.tracks = [self.create_track()]
        self.new_track_episode = [20000, 30000, 35000, 37500] # episodes at which to generate a new track
        
    def dyn(self, x, u):
        L = 3
        xdot = np.array([x[3]*np.cos(x[2]), x[3]*np.sin(x[2]), x[3]/self.L*np.tan(u[1]), u[0]])
        return xdot

    def sim(self, x, u, M=1):
        # RK4 with M steps
        DT = self.dt/M
        xf = x.copy()
        for j in range(M):
            k1 = self.dyn(x,             u)
            k2 = self.dyn(x + DT/2 * k1, u)
            k3 = self.dyn(x + DT/2 * k2, u)
            k4 = self.dyn(x + DT   * k3, u)
            xf += DT/6*(k1   + 2*k2   + 2*k3   + k4)
        return xf

    # def get_reward(self, x, u):
    #     r_dmax = max(abs(x[1]) - 3, 0)
    #     r_alatmax = max(abs(1./self.L*np.tan(u[1]))*x[3]**2 - 3, 0)
    #     r_v = x[0]
    #     ra = max(abs(u[0]) - 3, 0)
    #     rdelta = max(abs(u[1]) - 0.5, 0)
    #     return 0.5*r_v - 2*r_dmax - 2*r_alatmax - ra - rdelta

    def get_reward(self, obs, u):
        s = obs[0][0]
        d = obs[1][0]
        psi = obs[2][0]
        v = obs[3][0]
        delta = u[1][0]
    
        dmax_penalty = -2.*max(abs(d) - 1., 0.)
        d_penalty = -0.0*(abs(d)) # small penalty to encourage driving in the center
        lateral_accel = (v**2 / self.L) * np.tan(delta)
        a_lat_penalty = -0.2 * (lateral_accel / 3.0)**2
        neg_vel_penalty = 5.*min(v, 0.) # always < 0 -> penalty
        steer_penalty = -0.0*delta**2 # smooth steering

    
        # Progress reward
        progress = s + 0.1*v*np.cos(psi)
        
        # Being aligned with track
        heading = -2*abs(psi)
    
        # Combine
        return progress + dmax_penalty + d_penalty + 0*a_lat_penalty + heading + neg_vel_penalty + steer_penalty

        


    def step(self, u):
        """
        Parameters
        ----------
        u : TYPE
            DESCRIPTION.

        Returns
        -------
        observation (state in frenet + curvature at lookahead points)
        reward
        terminal flag
        state of the vehicle (for debug/plots)
        """
        self.x = self.sim(self.x, u)
        obs = self.get_observation()
        reward = self.get_reward(obs, u)
        terminal = False
        if abs(obs[1]) > self.max_d:
            reward += -5
            terminal = True
        if obs[0] > self.tracks[-1][-5,4]/100.0: # reach (a little bit before) end of track
            reward += 5
            terminal = True
        if self.dt*self.steps > self.tracks[-1][-1,4] / 5.0: # if end is not reached in time with 5 mps average speed
            reward += -2
            terminal = True
        
        self.steps = self.steps + 1;
        
        return obs, reward, terminal, self.x
    
    def reset(self):
        if self.episode in self.new_track_episode:
            self.tracks += [self.create_track()]
        self.x = np.array([[0.],[0.],[0.],[10.]])
        self.steps = 0
        return self.get_observation()
    
    def create_track(self):
        # track = np.asarray(self.track_generator.create_segment(0, 0, 0, 0, 0, steps=2000, ds=0.5))#
        track = np.asarray(self.track_generator.create_random(20))
        
        # Calculate s-coordinate
        # ds = np.linalg.norm(np.diff(track[:,0:2], axis=0), axis=1)
        # s = np.concatenate([[0], np.cumsum(ds)])
        
        # track = np.hstack((track, s[:,None]))
        return track
        
        
    def get_observation(self):
        
        xref = utils.get_ref_race_frenet(self.x, 1, self.tracks[-1], 0)
        
        xscale = np.array([[100],[3],[1],[15],[1],[1],[1],[1],[1],[1]])
        # calculate state in frenet frame [s, d, th-th_path, v]
        x = np.zeros((10,1))
        dth = self.x[2] - xref[2,0]
        x[0] = xref[5,0]
        x[1] = -(self.x[0] - xref[0,0])*np.sin(xref[2,0]) + (self.x[1] - xref[1,0])*np.cos(xref[2,0])
        x[2] = dth
        x[3] = self.x[3]
        x /= xscale
        
        # s-positions of curvature values to add to the observation
        s_lookahead = [5, 10, 20, 30, 40, 50]
        indices = np.searchsorted(self.tracks[-1][:,4], s_lookahead, side='left')
        indices = np.clip(indices, 0, self.tracks[-1].shape[0] - 1)
        k = self.tracks[-1][indices, 3]
        
        for i in range(4,10):
            x[i] = k[i-4]
        
        return x
            

###

TRAIN = True
LOAD_MODEL = False
WEIGHTS_FILE = 'SAC_race3.pt'
WEIGHTS_FILE_SAVE = 'SAC_race3.pt'

# RL agent
agent_parameters = {
    'network_config': {
        'state_dim': 10,
        'num_hidden_units': [256, 256],
        'num_actions': 2
    },
    'optimizer_config': {
        'step_size': 5e-4,#1e-3,
        'beta_m': 0.9, 
        'beta_v': 0.999,
        'epsilon': 1e-8 
    },
    'replay_buffer_size': 15000,
    'minibatch_sz': 32,
    'num_replay_updates_per_step': 8,
    'gamma': 0.8,
    'tau': 1,
    'polyak_tau': 0.005,
}

umin = torch.tensor([-3, -0.5])
umax = torch.tensor([ 3,  0.5])
agent = agent_torch.Agent(agent_parameters, umin, umax)

# Load checkpoint
if LOAD_MODEL:
    agent.load_checkpoint(WEIGHTS_FILE)

# Environment
# feature vector [s, d, th_error, v, k_5, k_10, k_20, k_30, k_40, k_50] where k_x is curvature at x meters ahead
x0 = np.array([[0.],[0.],[0.],[10.]])
env = Racecar(x0)

num_episodes = 150
if not TRAIN:
    num_episodes = 1
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    vehicle_artist = None
    trajectory_artist = None
    utils.draw_track(env.tracks[-1], 6)

X = [[] for _ in range(num_episodes)]
U = [[] for _ in range(num_episodes)]
R = [[] for _ in range(num_episodes)]
episode = 0
while episode < num_episodes:
    obs = env.reset()
    obsv = torch.tensor(obs.flatten(), dtype=torch.float32)
    agent.start(obsv)
    r = np.array(0)
    done = False
    total_reward = 0
    steps = 0
    while not done:
        rew = torch.tensor(r.flatten(), dtype=torch.float32)
        obsv = torch.tensor(obs.flatten(), dtype=torch.float32)
        if TRAIN:
            u = agent.step(rew, obsv)
        else:
            u = agent.policy(obsv, True)[0].detach()
        obs, r, done, x = env.step(u.numpy()[:,None])
        total_reward += float(r)
        steps += 1
        if done:
            rew = torch.tensor(r.flatten(), dtype=torch.float32)
            obsv = torch.tensor(obs.flatten(), dtype=torch.float32)
            agent.end(rew)
        X[episode] += [x]
        U[episode] += [u]
        R[episode] += [r]
        
        # Animation
        if not TRAIN:
            vehicle_artist, trajectory_artist = utils.draw_vehicle_and_trajectory(ax, x[0], x[1], x[2], future_states=None, vehicle_artist=vehicle_artist, trajectory_artist=trajectory_artist)
            plt.pause(0.025)

    episode += 1

    print(f"Ep {episode} | total reward {total_reward:.2f} | mean step {total_reward/steps:.3f} | steps {steps} | progress {obs[0]}/{env.tracks[-1][-1,4]/100}")


xx = np.asarray(X[-1])
uu = np.asarray(U[-1])
rr = np.asarray(R[-1])
print(u, " ", x, " ", r)
print("---")
print(obs)

print("")
if TRAIN:
    agent.save_checkpoint(WEIGHTS_FILE_SAVE)

#%% Plot
xx = np.asarray(X[episode-1])
plt.plot(env.tracks[0][:,0],env.tracks[0][:,1])
plt.plot(xx[:,0], xx[:,1])
plt.figure()
uu = np.asarray(U[episode-1][:-1])
plt.plot(uu[:,1])
plt.plot(uu[:,0])


# SAC. heading=2, d=0, steer =0, added progress s + 0.2vcos (a lot faster)
# SAC2 is initial working model. penalty=2 on heading so it doesnt cut corners a lot
    