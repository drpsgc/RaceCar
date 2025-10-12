import torch
import torch.nn as nn
import itertools
import torch.distributions as dist
import numpy as np
import random
from copy import deepcopy
from Models import Actor, Critic


class ReplayBuffer:
    def __init__(self, size, minibatch_size, seed):
        """
        Args:
            size (integer): The size of the replay buffer.              
            minibatch_size (integer): The sample size.
            seed (integer): The seed for the random number generator. 
        """
        self.buffer = []
        self.minibatch_size = minibatch_size
        self.rand_generator = np.random.RandomState(seed)
        self.max_size = size

    def append(self, state, action, reward, terminal, next_state):
        """
        Args:
            state (Numpy array): The state.              
            action (integer): The action.
            reward (float): The reward.
            terminal (integer): 1 if the next state is a terminal state and 0 otherwise.
            next_state (Numpy array): The next state.           
        """
        if len(self.buffer) == self.max_size:
            del self.buffer[0]
        self.buffer.append([state, action, reward, terminal, next_state])

    def sample(self):
        """
        Returns:
            A list of transition tuples including state, action, reward, terinal, and next_state
        """
        idxs = self.rand_generator.choice(np.arange(len(self.buffer)), size=self.minibatch_size)
        return [self.buffer[idx] for idx in idxs]

    def size(self):
        return len(self.buffer)




def softmax(action_values: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """
    Args:
        action_values (torch.Tensor): A 2D tensor of shape (batch_size, num_actions). 
                                      The action-values computed by an action-value network.              
        tau (float): The temperature parameter scalar.

    Returns:
        A 2D tensor of shape (batch_size, num_actions) with action probabilities.
    """
    preferences = action_values / tau

    # Subtract max for numerical stability
    max_preference = preferences.max(dim=1, keepdim=True).values
    exp_preferences = torch.exp(preferences - max_preference)

    # Compute softmax probabilities
    sum_exp_preferences = exp_preferences.sum(dim=1, keepdim=True)
    action_probs = exp_preferences / sum_exp_preferences

    # If used for action selection (batch size = 1), squeeze to make shape (num_actions,)
    return action_probs.squeeze()  # Only squeezes if dimension is 1

def polyak_update(params, target_params, tau):
    """
    Perform a Polyak average update on ``target_params`` using ``params``:
    target parameters are slowly updated towards the main parameters.
    ``tau``, the soft update coefficient controls the interpolation:
    ``tau=1`` corresponds to copying the parameters to the target ones whereas nothing happens when ``tau=0``.
    :param params: parameters to use to update the target params
    :param target_params: parameters to update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    """
    with torch.no_grad():
        for param, target_param in zip(params, target_params):
            target_param.data.mul_(1 - tau)
            torch.add(target_param.data, param.data, alpha=tau, out=target_param.data)


class Agent():
    def __init__(self, agent_config, amin, amax):
        """Setup for the agent called when the experiment first starts.

        Set parameters needed to setup the agent.

        Assume agent_config dict contains:
        {
            network_config: dictionary,
            optimizer_config: dictionary,
            replay_buffer_size: integer,
            minibatch_sz: integer, 
            num_replay_updates_per_step: float
            discount_factor: float,
        }
        """
        # Save config to a member, to save it to file
        self.agent_config = agent_config
        # Replay Buffer
        self.replay_buffer = ReplayBuffer(agent_config['replay_buffer_size'], 
                                          agent_config['minibatch_sz'], agent_config.get("seed"))
        
        # Networks
        self.critic1 = Critic(agent_config['network_config'])
        self.critic2 = Critic(agent_config['network_config'])
        self.target1 = Critic(agent_config['network_config'])
        self.target2 = Critic(agent_config['network_config'])
        self.actor = Actor(agent_config['network_config'])
        self.crit_params = itertools.chain(self.critic1.parameters(), self.critic2.parameters())
        # initialize target same as critic
        self.target1.load_state_dict(self.critic1.state_dict()) 
        self.target2.load_state_dict(self.critic2.state_dict())
        for p in self.target1.parameters():
            p.requires_grad = False
        for p in self.target2.parameters():
            p.requires_grad = False
        
        # Action limits
        self. a_scale = (amax - amin)/2
        self.a_bias = (amax + amin)/2

        # Optimizer
        self.critic_optimizer = torch.optim.Adam(self.crit_params, lr=agent_config["optimizer_config"]["step_size"])
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=agent_config["optimizer_config"]["step_size"])

        # Agent params
        self.alpha = agent_config.get('alpha', 0.005) # Entropy tradeoff
        self.num_actions = agent_config['network_config']['num_actions']
        self.num_replay = agent_config['num_replay_updates_per_step']
        self.gamma = agent_config['gamma']
        self.tau = agent_config['tau']
        self.polyak_tau = agent_config['polyak_tau']
        self.use_expected_sarsa = agent_config.get('use_expected_sarsa', False)
        self.use_softmax_policy = agent_config.get('use_softmax_policy', False)
        print("sarsa?", self.use_expected_sarsa, "softmax?", self.use_softmax_policy)

        self.rand_generator = np.random.RandomState(agent_config.get("seed"))
        
        self.last_state = None
        self.last_action = None
        
        self.sum_rewards = 0
        self.episode_steps = 0
        self.critic_losses = []

    def policy(self, states, deterministic = False):
        """
        Args:
            states (Numpy array): the possible next states (shape: [batch_size, state_dim]).
        Returns:
            action (np.array): the action sampled from the policy (shape: [batch_size, action_dim]).
        """
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states).float()
    
        # Get [mu1, mu2, log_sigma1, log_sigma2] for each state
        mu, std = self.actor(states)  # shape: [batch_size, n_states]
        
        dist_normal = dist.Normal(mu, std)
        
        # 1. Sample pre-squash action
        u = dist_normal.rsample() if not deterministic else mu
        
        # 2. Tanh squash
        a = torch.tanh(u)
        
        # 3. Scale and shift to env action space
        action = a * self.a_scale + self.a_bias
        
        # 4. Correct log-prob BEFORE scaling (i.e. correciton doesnt apply to scale)
        log_prob = dist_normal.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        return action, log_prob
    
    def start(self, state):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (Numpy array): the state from the
                environment's evn_start function.
        Returns:
            The first action the agent takes.
        """
        self.sum_rewards = 0
        self.episode_steps = 0
        self.last_state = state
        action, _ = self.policy(state)
        self.last_action = action.detach()
        self.last_reward = 0
        return self.last_action

    def step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (Numpy array): the state from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """
        
        self.sum_rewards += reward
        self.episode_steps += 1


        # Select action
        action, _ = self.policy(state)
        
        # Append new experience to replay buffer
        self.replay_buffer.append(self.last_state, self.last_action, reward, 0, state)
        
        # Perform replay steps:
        if self.replay_buffer.size() > self.replay_buffer.minibatch_size:
            for _ in range(self.num_replay):
                
                # Get sample experiences from the replay buffer
                experiences = self.replay_buffer.sample()
                
                # Call optimize_network to update the weights of the network (~1 Line)
                self.train_step(experiences)
                
        # Update the last state and last action.
        self.last_state = state
        self.last_action = action.detach()
        
        return action.detach()

    def end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        self.sum_rewards += reward
        self.episode_steps += 1

        # Set terminal state to an array of zeros
        state = np.zeros_like(self.last_state)

        # Append new experience to replay buffer
        self.replay_buffer.append(self.last_state, self.last_action, reward, 1, state)
        
        # Perform replay steps:
        if self.replay_buffer.size() > self.replay_buffer.minibatch_size:
            # current_q = deepcopy(self.network)
            for _ in range(self.num_replay):
                if self.use_expected_sarsa:
                    polyak_update(self.network.parameters(), self.target_network.parameters(), 1)

                # Get sample experiences from the replay buffer
                experiences = self.replay_buffer.sample()
                
                # Call optimize_network to update the weights of the network
                self.train_step(experiences)      
        
    def agent_message(self, message):
        if message == "get_sum_reward":
            return self.sum_rewards
        else:
            raise Exception("Unrecognized Message!")


    def train_step(self, experiences):
        states, actions, rewards, terminals, next_states = map(list, zip(*experiences))
        states = torch.from_numpy(np.vstack(states)).float()
        next_states = torch.from_numpy(np.vstack(next_states)).float()
        actions = torch.stack(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        terminals = torch.tensor(terminals, dtype=torch.float32)
        
        # --- Critic update ---
        self.critic_optimizer.zero_grad()
        q_loss = self.critic_loss(experiences)
        q_loss.backward()
        self.critic_optimizer.step()
        
        # Freeze Q-networks for efficiency (dont need grads)
        for p in self.crit_params:
            p.requires_grad = False

        
        # --- Actor update  ---
        self.actor_optimizer.zero_grad()
        pi_loss = self.actor_loss(experiences)
        pi_loss.backward()
        self.actor_optimizer.step()
        
        # Unfreeze Q-networks
        for p in self.crit_params:
            p.requires_grad = True
        
        # Update target networks
        with torch.no_grad():
            polyak_update(self.critic1.parameters(), self.target1.parameters(), self.polyak_tau)
            polyak_update(self.critic2.parameters(), self.target2.parameters(), self.polyak_tau)
        
       

    def critic_loss(self, experiences):
        states, actions, rewards, terminals, next_states = map(list, zip(*experiences))
        states = torch.from_numpy(np.vstack(states)).float()
        next_states = torch.from_numpy(np.vstack(next_states)).float()
        actions = torch.stack(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        terminals = torch.tensor(terminals, dtype=torch.float32)
        
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        
        with torch.no_grad():
            # Sample current policy for target (NOT FROM EXP BUFFER)
            next_actions, next_log_probs = self.policy(next_states)
            
            # Target values
            q1_targ = self.target1(next_states, next_actions)
            q2_targ = self.target2(next_states, next_actions)
            # use min target trick
            q_targ = torch.min(q1_targ, q2_targ)
            y = rewards + self.gamma*(1-terminals) * (q_targ - self.alpha * next_log_probs)
        
        # MSE loss
        loss_critic1 = ((q1 - y)**2).mean()
        loss_critic2 = ((q2 - y)**2).mean()
        loss = loss_critic1 + loss_critic2
        
        return loss
    
    def actor_loss(self, experiences):
        states, actions, rewards, terminals, next_states = map(list, zip(*experiences))
        states = torch.from_numpy(np.vstack(states)).float()
        
        actions, log_probs = self.policy(states)
        
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        # Use min target trick
        q = torch.min(q1, q2)
        
        # Entropy-regularized actor loss
        loss = (-q + self.alpha * log_probs).mean()
        
        return loss
    
    # --- Save checkpoint ---
    def save_checkpoint(self, filepath):
        checkpoint = {
            'critic1_state': self.critic1.state_dict(),
            'critic2_state': self.critic2.state_dict(),
            'target1_state': self.target1.state_dict(),
            'target2_state': self.target2.state_dict(),
            'actor_state': self.actor.state_dict(),
            'critic_optimizer_state': self.critic_optimizer.state_dict(),
            'actor_optimizer_state': self.actor_optimizer.state_dict(),
            'agent_config': self.agent_config,
        }
        torch.save(checkpoint, filepath)
        
        print(f"✅ Checkpoint saved to {filepath}")
        
    # --- Load checkpoint ---
    def load_checkpoint(self, filepath, map_location=None):
        checkpoint = torch.load(filepath, map_location=map_location)

        self.critic1.load_state_dict(checkpoint['critic1_state'])
        self.critic2.load_state_dict(checkpoint['critic2_state'])
        self.target1.load_state_dict(checkpoint['target1_state'])
        self.target2.load_state_dict(checkpoint['target2_state'])
        self.actor.load_state_dict(checkpoint['actor_state'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state'])
        self.agent_config = checkpoint.get('agent_config', {})

        print(f"✅ Checkpoint loaded from {filepath}")
        


