import torch
import itertools
import torch.distributions as dist
import numpy as np

from Models import Actor, Critic

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../', 'utils')))

from ReplayBuffer import ReplayBuffer


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
    def __init__(self, agent_config, amin, amax, expert_buffer):
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
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Save config to a member, to save it to file
        self.agent_config = agent_config
        # Replay Buffer
        state_dim = agent_config['network_config']["state_dim"]
        action_dim = agent_config['network_config']["num_actions"]
        self.replay_buffer = ReplayBuffer(agent_config['replay_buffer_size'], state_dim, action_dim,
                                          agent_config['minibatch_sz'], agent_config.get("seed"))
        self.expert_buffer = expert_buffer
        expert_buffer.actions[:,0] /=  5000. # scale expert actions

        # Networks
        self.critic1 = Critic(agent_config['network_config']).to(self.device)
        self.critic2 = Critic(agent_config['network_config']).to(self.device)
        self.target1 = Critic(agent_config['network_config']).to(self.device)
        self.target2 = Critic(agent_config['network_config']).to(self.device)
        self.actor = Actor(agent_config['network_config']).to(self.device)
        self.crit_params = itertools.chain(self.critic1.parameters(), self.critic2.parameters())
        # initialize target same as critic
        self.target1.load_state_dict(self.critic1.state_dict()) 
        self.target2.load_state_dict(self.critic2.state_dict())
        for p in self.target1.parameters():
            p.requires_grad = False
        for p in self.target2.parameters():
            p.requires_grad = False

        # Update freq
        self.critic_update_freq = agent_config.get('critic_update_freq',1)
        self.actor_update_freq = agent_config.get('actor_update_freq',1)
        self.target_update_freq = agent_config.get('target_update_freq',1)
        
        # Action limits
        self.a_scale = torch.tensor((amax - amin) / 2.0, dtype=torch.float32, device=self.device)
        self.a_bias = torch.tensor((amax + amin) / 2.0, dtype=torch.float32, device=self.device)

        # Optimizer
        self.critic_optimizer = torch.optim.Adam(self.crit_params, lr=agent_config["optimizer_config"]["step_size_c"])
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=agent_config["optimizer_config"]["step_size_a"])
        # Imitation learning weight
        self.IL_weight = agent_config.get("IL_weight", 2.)

        # Agent params
        self.alpha = agent_config.get('alpha', 0.005) # Entropy tradeoff
        self.num_actions = agent_config['network_config']['num_actions']
        self.num_replay = agent_config['num_replay_updates_per_step']
        self.gamma = agent_config['gamma']
        self.tau = agent_config['tau']
        self.polyak_tau = agent_config['polyak_tau']

        self.rand_generator = np.random.RandomState(agent_config.get("seed"))
        
        self.last_state = None
        self.last_action = None
        
        self.sum_rewards = 0
        self.episode_steps = 0
        self.episode = 0
        self.train_step_counter = 0
        self.crit_losses = []
        self.actor_losses = []

    def policy(self, states, deterministic = False, actions = None):
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states).float().to(self.device)
        else:
            states = states.float().to(self.device)
    
        # Get [mu1, mu2, log_sigma1, log_sigma2] for each state
        mu, std = self.actor(states)  # shape: [batch_size, n_states]
        dist_normal = dist.Normal(mu, std)
        
        if actions is None:
            # --- SAMPLE ----
            # 1. Sample pre-squash action
            u = dist_normal.rsample() if not deterministic else mu
            a = torch.tanh(u)
            
            # 3. Scale and shift to env action space
            action = a * self.a_scale + self.a_bias
            
            # 4. Correct log-prob BEFORE scaling (i.e. correciton doesnt apply to scale)
            log_prob = dist_normal.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        else:
            # --- BC Mode: compute logprob(actions) ---
            eps = 1e-6
            a = (actions - self.a_bias) / self.a_scale
            a = a.clamp(-1 + eps, 1 - eps)

            u = torch.atanh(a)

            # normal logprob
            log_prob = dist_normal.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            action = None
        
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
        self.episode += 1
        self.IL_weight *= 0.99 # becomes nearlyt 0 at 500
        self.last_state = state
        action, _ = self.policy(state)
        self.last_action = action.detach().cpu().numpy()
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
        action = action.detach().cpu().numpy()
        
        # Append new experience to replay buffer
        self.replay_buffer.append(self.last_state, self.last_action, reward, 0, state)
        
        # Perform replay steps:
        if self.replay_buffer.size() > self.replay_buffer.minibatch_size:
            for _ in range(self.num_replay):
                # Get sample experiences from the replay buffer
                experiences = self.replay_buffer.sample()
                expert_samp = self.expert_buffer.sample()
                
                # Call optimize_network to update the weights of the network (~1 Line)
                self.train_step(experiences, expert_samp)
                
        # Update the last state and last action.
        self.last_state = state
        self.last_action = action
        
        return action

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
                # Get sample experiences from the replay buffer
                experiences = self.replay_buffer.sample()
                expert_samp = self.expert_buffer.sample()
                
                # Call optimize_network to update the weights of the network
                self.train_step(experiences, expert_samp)      

    def train_step(self, experiences, expert_samp):
        
        self.train_step_counter += 1
        states, actions, rewards, terminals, next_states = experiences
        states = torch.from_numpy(states).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        actions = torch.from_numpy(actions).float().to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        terminals = torch.tensor(terminals, dtype=torch.float32).to(self.device)
        
        expert_states, expert_actions, _, _, _ = expert_samp
        expert_states = torch.from_numpy(expert_states).float().to(self.device)
        expert_actions = torch.from_numpy(expert_actions).float().to(self.device)
        
        # --- Critic update ---
        if self.train_step_counter % self.critic_update_freq == 0:
            self.critic_optimizer.zero_grad()
            q_loss = self.critic_loss(states, next_states, actions, rewards, terminals)
            q_loss.backward()
            self.critic_optimizer.step()
            self.crit_losses += [q_loss.item()]
        
        if self.train_step_counter % self.actor_update_freq == 0:
            # Freeze Q-networks for efficiency (dont need grads)
            # for p in self.crit_params:
            #     p.requires_grad = False
    
            
            # --- Actor update  ---
            self.actor_optimizer.zero_grad()
            pi_loss, log_pi = self.actor_loss(states, expert_states, expert_actions)
            pi_loss.backward()
            self.actor_optimizer.step()

            # Unfreeze Q-networks
            # for p in self.crit_params:
            #     p.requires_grad = True
            
            self.actor_losses += [pi_loss.item()]
        
        # Update target networks
        if self.episode_steps % self.target_update_freq == 0:
            with torch.no_grad():
                polyak_update(self.critic1.parameters(), self.target1.parameters(), self.polyak_tau)
                polyak_update(self.critic2.parameters(), self.target2.parameters(), self.polyak_tau)
        
  

    def critic_loss(self, states, next_states, actions, rewards, terminals):
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
    
    def actor_loss(self, states, s_E, a_E):
        actions, log_probs = self.policy(states)
        
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        # Use min target trick
        q = torch.min(q1, q2)
             
        # Entropy-regularized actor loss
        SAC_loss = (-q + self.alpha * log_probs).mean()
        
        # Imitation learning loss
        _, log_prob_E = self.policy(s_E, actions=a_E)
        # mu_E, std_E = self.actor(s_E)
        # dist_E = torch.distributions.Normal(mu_E, std_E)
        # log_prob_E = dist_E.log_prob(a_E).sum(dim=-1)
        bc_loss = -log_prob_E.mean()
        
        loss = SAC_loss + self.IL_weight*bc_loss
        return loss, log_probs


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
        


