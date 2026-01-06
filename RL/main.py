#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 17:59:32 2025

@author: psgc
"""

import agent_torch_ILRL
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Hacky way to import utilities
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

import utils
import racetrack
from VehicleModel import DynVehicleModel

import pickle

### Environment
class Racecar:
    def __init__(self, x0, train=False):
        self.L = 3
        self.x0 = x0
        self.x = x0
        self.max_d = 3.5 # max tolerated lateral error, if exceeded episode terminates
        self.steps = 0
        self.dt = 0.05;
        self.track_len = None
        self.max_time = None
        self.mean_speed_min = 7
        
        # Track generator
        self.tracks = []
        self.train = train
        if train:
            self.track_generator = racetrack.RaceTrack("random")
            #### CREATE TRACKS
            self.no_tracks = 18 # track no 17 with default seed is not good, near self-crossing
            tracklen = 20 # segments
            for i in range(self.no_tracks-1):
                self.tracks += [self.track_generator.create_random(tracklen)]
                # utils.draw_track(curr_track, 6)
            self.tracks += [self.track_generator.create_circuit()]
        else:
            self.track_generator = racetrack.RaceTrack("circuit")
            self.tracks += [self.track_generator.track]
        self.episode = -1
        self.curr_track = self.tracks[0]
        self.new_track_freq = 10
        
    def dyn(self, x, u):
        L = 3
        xdot = np.array([x[3]*np.cos(x[2]), x[3]*np.sin(x[2]), x[3]/self.L*np.tan(u[1]), u[0],[0],[0]])
        return xdot

    # def sim(self, x, u, M=1):
    #     # RK4 with M steps
    #     DT = self.dt/M
    #     xf = x.copy()
    #     us = u.copy()
    #     us[0] *= 3.
    #     for j in range(M):
    #         k1 = self.dyn(x,             us)
    #         k2 = self.dyn(x + DT/2 * k1, us)
    #         k3 = self.dyn(x + DT/2 * k2, us)
    #         k4 = self.dyn(x + DT   * k3, us)
    #         xf += DT/6*(k1   + 2*k2   + 2*k3   + k4)
    #     return xf
    def sim(self, x, u, M=1):
        # RK4 with M steps
        DT = self.dt/M
        xf = x.copy()
        us = u.copy()
        us[0] *= 5000.
        for j in range(M):
            k1 = DynVehicleModel(xf,             us)
            k2 = DynVehicleModel(xf + DT/2 * k1, us)
            k3 = DynVehicleModel(xf + DT/2 * k2, us)
            k4 = DynVehicleModel(xf + DT   * k3, us)
            xf += DT/6*(k1   + 2*k2   + 2*k3   + k4)
        return xf

    # def get_reward(self, x, u):
    #     r_dmax = max(abs(x[1]) - 3, 0)
    #     r_alatmax = max(abs(1./self.L*np.tan(u[1]))*x[3]**2 - 3, 0)
    #     r_v = x[0]
    #     ra = max(abs(u[0]) - 3, 0)
    #     rdelta = max(abs(u[1]) - 0.5, 0)
    #     return 0.5*r_v - 2*r_dmax - 2*r_alatmax - ra - rdelta

    # def get_reward(self, obs, u):
    #     s = obs[0][0]
    #     d = obs[1][0]
    #     psi = obs[2][0]
    #     v = obs[3][0]
    #     delta = u[1][0]
    
    #     dmax_penalty = -2.*max(abs(d) - 1., 0.)
    #     d_penalty = -0.0*(abs(d)) # small penalty to encourage driving in the center
    #     lateral_accel = (v**2 / self.L) * np.tan(delta)
    #     a_lat_penalty = -0.2 * (lateral_accel / 3.0)**2
    #     neg_vel_penalty = 5.*min(v, 0.) # always < 0 -> penalty
    #     steer_penalty = -0.0*delta**2 # smooth steering

    
    #     # Progress reward
    #     progress = s + 0.1*v*np.cos(psi)
        
    #     # Being aligned with track
    #     heading = -2*abs(psi)
    
    #     # Combine
    #     return progress + dmax_penalty + d_penalty + 0*a_lat_penalty + heading + neg_vel_penalty + steer_penalty

    def get_reward(self, obs, obs_pre, u):
        s = obs[0][0]
        d = obs[1][0]
        psi = obs[2][0]
        vx = obs[3][0]
        vy = obs[4][0]
        r = obs[5][0]
        a = u[0][0]
        delta = u[1][0]
        
        s_pre = obs_pre[0][0]
    
        progress = 0.0 * vx * np.cos(psi) + 150.*(s - s_pre)# note: s is scaled by 100
    
        # Centering: smooth attractive reward toward d=0 (gaussian-like)
        # at d=0 gives +0.1, at d=0.5 gives small positive, at d=1 gives small negative
        r_center = 0.15 * np.exp(-0.5 * (d / 0.4)**2)  # tune sigma=0.4
        # dmax_penalty = -1.*max(abs(d) - 1., 0.)
        dmax_penalty = -0.75 * F.softplus(torch.abs(torch.tensor(d)) - 1.0, beta=10)
    
        # Heading penalty (mild)
        r_heading = -0.1 * (abs(psi)) 
    
        # Vel penalty for going backwards
        r_neg_vel = -5.0 * max(-vx, 0.0)
    
        # Control costs (small, quadratic)
        # r_accel = -0.01 * (a**2)
        r_steer = -0.015 * (delta**2)
        
        lat_accel = ((15.0*vx)**2 / self.L) * np.tan(delta)
        # a_lat_penalty = -0.75 * max(abs(lat_accel) - 3., 0.)
        a_lat_penalty = -0.03 *  F.softplus(torch.abs(torch.tensor(lat_accel)) - 3.0, beta=3) 
    
        # Combine
        reward = progress + 0.*r_center + r_heading + 0*r_neg_vel + r_steer + 0*dmax_penalty + 0*a_lat_penalty
    
        return reward

        


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
        xx = self.x.copy()
        obs_pre, _ = self.get_observation(xx)
        self.x = self.sim(xx, u)
        obs, obs_unscaled = self.get_observation(self.x)
        reward = self.get_reward(obs, obs_pre, u)
        terminal = False
        if abs(obs_unscaled[1]) > self.max_d:
            reward += -2
            terminal = True
        if obs_unscaled[0] > self.track_len-2: # reach (a little bit before) end of track
            reward += 2. + 5.*(self.max_time - obs_unscaled[21])/self.max_time # higher reward if more time left
            terminal = True
        if self.dt*self.steps > self.max_time: # if end is not reached in time with min mean speed average speed
            reward += -1 - 2*(self.track_len - obs_unscaled[0])/self.track_len # more penalty if not driven far
            terminal = True
        if obs_unscaled[0] == 0 and obs_unscaled[3] < 0: # if reversing through start line
            reward += -5
            terminal = True
        
        self.steps = self.steps + 1;
        
        return obs, reward, terminal, self.x
    
    def reset(self):
        self.episode += 1
        if self.episode % self.new_track_freq == 0:
            self.set_track()
        self.x = self.x0
        self.steps = 0
        return self.get_observation(self.x)[0]
    
    def set_track(self):
        # track = np.asarray(self.track_generator.create_segment(0, 0, 0, 0, 0, steps=2000, ds=0.5))#
        if self.train:
            selection = np.random.randint(0, self.no_tracks)
            print("selecting track: ", selection)
            self.curr_track = self.tracks[selection]
            if selection == self.no_tracks-1: # last track starts at different position
                self.x0 = np.array([[0.],[100.],[0.],[10.],[0.],[0.]])
            else:
                self.x0 = np.array([[0.],[0.],[0.],[10.],[0.],[0.]])

        self.track_len = self.curr_track[-1,4]
        self.max_time = self.track_len / self.mean_speed_min # Maximum time assuming mean speed min


    def get_observation(self, xin):
        xcurr = xin.copy()
        xref = utils.get_ref_race_frenet(xcurr, 1, self.curr_track, 0)
        
        xscale = np.array([[100],[3],[1],[15],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[self.max_time]])
        # calculate state in frenet frame [s, d, th-th_path, v]
        x = np.zeros((22,1))
        dth = xcurr[2] - xref[2,0]
        x[0] = xref[5,0]
        x[1] = -(xcurr[0] - xref[0,0])*np.sin(xref[2,0]) + (xcurr[1] - xref[1,0])*np.cos(xref[2,0])
        x[2] = dth
        x[3] = xcurr[3]
        x[4] = xcurr[4]
        x[5] = xcurr[5]

        # s-positions of curvature values to add to the observation
        s_lookahead = [0, 2, 5, 10, 15, 20, 25,   30, 35, 40, 45, 50, 60, 70, 80, 90]
        s_lookahead = [s + x[0]  for s in s_lookahead]
        indices = np.searchsorted(self.curr_track[:,4], s_lookahead, side='left')
        indices = np.clip(indices, 0, self.curr_track.shape[0] - 1)
        k = self.curr_track[indices, 3]
        
        for i in range(6,21):
            x[i] = k[i-6]
        x[21] = self.dt * self.steps # time driven this episode
        
        obs = x/xscale # scale observation

        return obs, x
            

###

TRAIN = True
LOAD_MODEL = False#not TRAIN#True
WEIGHTS_FILE = 'Checkpoints/SAC_race_Dyn4_chk53.pt'
WEIGHTS_FILE_SAVE = 'SAC_race_Dyn5.pt'
CHKP_FILE_SAVE = 'SAC_race_Dyn5_chk'

# Load expert's demonstrations
with open("replay_buffer_expert_data18.pkl", "rb") as f:
    expert_buffer = pickle.load(f)

# RL agent
agent_parameters = {
    'network_config': {
        'state_dim': 22,
        'num_hidden_units': [256, 256],
        'num_actions': 2
    },
    'optimizer_config': {
        'step_size_c': 1e-4,#1e-3,
        'step_size_a': 1e-4,#1e-3,
        'step_size_alpha': 1e-4,#1e-3,
    },
    'replay_buffer_size': 150000,
    'minibatch_sz': 32,
    'num_replay_updates_per_step': 8,
    'gamma': 0.995,
    'alpha':0.005,
    'tau': 1,
    'polyak_tau': 0.005,
    'IL_weight': 0.0001,
    'learn_alpha': False
}

umin = torch.tensor([-1., -np.pi/5])
umax = torch.tensor([ 1,  np.pi/5])
agent = agent_torch_ILRL.Agent(agent_parameters, umin, umax, expert_buffer)

# Load checkpoint
if LOAD_MODEL:
    agent.load_checkpoint(WEIGHTS_FILE)

# Environment
# feature vector [s, d, th_error, v, k_5, k_10, k_20, k_30, k_40, k_50] where k_x is curvature at x meters ahead
if TRAIN:
    x0 = np.array([[0.],[0.],[0.],[10.],[0.],[0.]])
else:
    x0 = np.array([[0.],[100.],[0.],[10.],[0.],[0.]])
env = Racecar(x0, TRAIN)

num_episodes = 3000
if not TRAIN:
    num_episodes = 1
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    vehicle_artist = None
    trajectory_artist = None
    utils.draw_track(env.curr_track, 6)

X = [[] for _ in range(num_episodes)]
U = [[] for _ in range(num_episodes)]
R = [[] for _ in range(num_episodes)]
episode = 0
last_saved = 0
chk = 1
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
            u, _ = agent.policy(obsv, True)
            u = u.detach().cpu().numpy()
        action = u#.detach().numpy()
        obs, r, done, x = env.step(action[:,None])

        total_reward += float(r)
        steps += 1
        if done:
            rew = torch.tensor(r.flatten(), dtype=torch.float32)
            obsv = torch.tensor(obs.flatten(), dtype=torch.float32)
            agent.end(rew)
            print("FINISHED! Lap time: ", 0.1*steps)
            break
        X[episode] += [x]
        U[episode] += [u]
        R[episode] += [r]
        
        # Animation
        if not TRAIN:
            vehicle_artist, trajectory_artist = utils.draw_vehicle_and_trajectory(ax, x[0], x[1], x[2], future_states=None, vehicle_artist=vehicle_artist, trajectory_artist=trajectory_artist)
            plt.pause(0.025)

    if (episode > 1 and episode % 100 == 0) or (obs[0]/(env.curr_track[-1,4]/100) > 0.8 and total_reward > 0 and episode - last_saved > 10 ):
        agent.save_checkpoint("Checkpoints/" + CHKP_FILE_SAVE + str(chk) + ".pt")
        last_saved = episode
        chk += 1

    episode += 1

    # print(f"Ep {episode} | total reward {total_reward:.2f} | mean step {total_reward/steps:.3f} | steps {steps} | progress {obs[0]}/{env.curr_track[-1,4]/100}")
    print(f"Ep {episode} | total reward {total_reward:.2f} | alpha {agent.alpha:.4f} | steps {steps} | progress {obs[0]}/{env.curr_track[-1,4]/100}")


xx = np.asarray(X[-1])
# uu = np.asarray(U[-1])
# rr = np.asarray(R[-1])
#%%
print("")
if TRAIN:
    agent.save_checkpoint('Checkpoints/' + WEIGHTS_FILE_SAVE)

#%% Plot
xx = np.asarray(X[episode-1]).squeeze()

plt.plot(env.tracks[0][:,0],env.tracks[0][:,1])
plt.plot(xx[:,0], xx[:,1])

plt.figure()
uu = np.asarray(U[episode-1][:-1])
plt.plot(uu[:,1])
plt.plot(uu[:,0])

plt.figure()
L = 3
a_lat = xx[0:-1,3]**2*( 1/L*np.tan(uu[:,1]))
plt.plot(a_lat)
plt.title("Lateral accel")

plt.figure()
plt.plot(agent.crit_losses)
plt.plot(agent.actor_losses)
plt.title("Losses")



# SAC. heading=2, d=0, steer =0, added progress s + 0.2vcos (a lot faster)
# SAC2 is initial working model. penalty=2 on heading so it doesnt cut corners a lot
# SAC3 Fixed missing curvatures [0, 2, 5, 10, 15, 20, ] (last was missing). 
        # Rewards progress = 0.4 * v * np.cos(psi)  dmax_penalty = -0.75 * F.softplus(torch.abs(torch.tensor(d)) - 1.0, beta=10)
        # r_heading = -0.2 * (abs(psi)) 
        # r_neg_vel = -1.0 * max(-v, 0.0)
        # r_steer = -0.015 * (delta**2)
        # lat_accel = ((15.0*v)**2 / self.L) * np.tan(delta)
        # a_lat_penalty = -0.03 *  F.softplus(torch.abs(torch.tensor(lat_accel)) - 3.0, beta=3) 
        # Trained for 300 episodes.
# SAC4: Added additional rewards on episode end: if finished reward extra on time left. If timeout penalize more if not driven far
        # Added unscaled obs output to calculate rewards on physical quantities more easily
    