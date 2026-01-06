#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 20:27:35 2025

@author: psgc
"""
import numpy as np

class ReplayBuffer:
    def __init__(self, size, state_dim, action_dim, minibatch_size, seed):
        self.max_size = size
        self.ptr = 0
        self.size_ = 0
        self.minibatch_size = minibatch_size
        self.rand_generator = np.random.RandomState(seed)
        self.states = np.zeros((size, state_dim), dtype=np.float32)
        self.actions = np.zeros((size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((size), dtype=np.float32)
        self.terminals = np.zeros((size), dtype=np.float32)
        self.next_states = np.zeros((size, state_dim), dtype=np.float32)

    def append(self, state, action, reward, terminal, next_state):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.terminals[self.ptr] = terminal
        self.next_states[self.ptr] = next_state

        self.ptr = (self.ptr + 1) % self.max_size
        self.size_ = min(self.size_ + 1, self.max_size)

    def sample(self):
        idxs = self.rand_generator.randint(0, self.size_, size=self.minibatch_size)
        return (
            self.states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.terminals[idxs],
            self.next_states[idxs],
        )

    def size(self):
        return self.size_