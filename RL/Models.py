import torch
import torch.nn as nn
from torch.nn import functional as F

####### SIMPLE FFW ########

class Actor(nn.Module):
    def __init__(self, network_config):
        super(Actor, self).__init__()
        
        self.state_dim = network_config.get("state_dim")
        self.hidden_sizes = network_config.get("num_hidden_units")
        self.num_actions = network_config.get("num_actions")
        self.min_std = network_config.get("min_std", -20)
        self.max_std = network_config.get("max_std", 2)
        
        layers = []
        #Input layer
        layers.append(nn.Linear(self.state_dim, self.hidden_sizes[0]))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for i in range( len(self.hidden_sizes) - 1):
            layers.append(nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i+1]))
            layers.append(nn.ReLU())
        
        self.layers = nn.Sequential(*layers)
            
        # Output layers mu + log st.dev.
        self.mu = nn.Linear(self.hidden_sizes[-1], self.num_actions)
        self.log_std = nn.Linear(self.hidden_sizes[-1], self.num_actions)
        
    def forward(self, state):
        x = self.layers(state)
        mu = torch.tanh(self.mu(x))
        logstd = self.log_std(x)
        logstd = torch.clamp(logstd, self.min_std, self.max_std)
        std = torch.exp(logstd)

        return mu, std

class Critic(nn.Module):
    def __init__(self, network_config):
        super(Critic, self).__init__()
        
        self.state_dim = network_config.get("state_dim")
        self.hidden_sizes = network_config.get("num_hidden_units")
        self.num_actions = network_config.get("num_actions")
        
        layers = []
        #Input layer
        layers.append(nn.Linear(self.state_dim + self.num_actions, self.hidden_sizes[0]))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for i in range( len(self.hidden_sizes) - 1):
            layers.append(nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i+1]))
            layers.append(nn.ReLU())
            
        # Output layer: action value function
        layers.append( nn.Linear(self.hidden_sizes[-1], 1))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, state, action):
        x = torch.hstack((state, action))
        return self.model(x)
        
class MLP(nn.Module):
    def __init__(self, network_config):
        super(MLP, self).__init__()

        self.state_dim = network_config.get("state_dim")
        self.hidden_sizes = network_config.get("num_hidden_units")
        self.num_actions = network_config.get("num_actions")

        layers = []
        # Input layer
        layers.append(nn.Linear(self.state_dim, self.hidden_sizes[0]))
        layers.append(nn.ReLU())

        # Hidden layers
        for i in range(len(self.hidden_sizes) - 1):
            layers.append(nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i+1]))
            layers.append(nn.ReLU())

        # Output layer, mean + log st.dev per action 
        layers.append(nn.Linear(self.hidden_sizes[-1], 2*self.num_actions))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    

####### TRANSFORMER MODEL #######

class Head(nn.Module):
    """ one head of self attention"""
    
    def __init__(self, head_size, block_size, n_embed, dropout = 0.2):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        # self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # create tril
        self.register_buffer('eye', torch.eye(block_size)) # mask diagonal elements - dont attend self
        self.dropout = nn.Dropout(dropout)    
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B,T,C)
        q = self.query(x) # (B,T,C)
        # Compute attention scores (affinities)
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B,T,C) @ (B,C,T) -> (B,T,T) normalized by sqrt(dim) as in the paper
        # wei = wei.masked_fill(self.eye[:T,:T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei)
        # Perform weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v
        
        return out
    
class MultiHeadAttention(nn.Module):
    """ multiple heads of self attention"""
    
    def __init__(self, num_heads, head_size, block_size, n_embed, dropout=0.2):
        super().__init__()
        self.heads = nn.ModuleList((Head(head_size, block_size, n_embed, dropout) for _ in range(num_heads)))
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    
    def __init__(self, n_embed, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed), # factor 4 from attention paper
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed), # projection layer, going back to residual pathway
            nn.Dropout(dropout),
            )
        
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Trasnformer Block: communication followed by computation """
    def __init__(self, n_embed, n_head, block_size, dropout=0.2):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size, block_size, n_embed, dropout)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed) # from paper
        
    def forward(self, x):
        # add residual connection for better training
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x)) 
        return x

class Transformer(nn.Module):
    def __init__(self, network_config):
        super().__init__()

        self.state_dim = network_config.get("state_dim", 5)
        self.n_layer = network_config.get("num_layers", 1)
        self.n_head = network_config.get("num_attn_heads", 1)
        self.block_size = network_config.get("block_size", 5) # number of actors (including ego)
        self.n_embed = network_config.get("n_embed", 32) # akin to hidden units
        self.dropout = network_config.get("dropout", 0.05)
        self.num_actions = network_config.get("num_actions", 5)
        
        self.embed = nn.Linear(self.state_dim, self.n_embed) # Embed to higher space
        self.role_embed = nn.Embedding(2, self.n_embed)
        self.blocks = nn.Sequential(*[Block(self.n_embed, self.n_head, self.block_size, self.dropout) for _ in range(self.n_layer)])
        self.ln_f = nn.LayerNorm(self.n_embed)
        self.lm_head = nn.Linear(self.n_embed, self.num_actions)
        
    def forward(self, x, targets=None):
        
        x = x.reshape((-1,self.block_size,self.state_dim))

        # Batch, Number of neighbors
        B, N, C = x.shape
        
        x = self.embed(x) # Embed agents info to higher dim

        # Role embedding
        self.roles = torch.ones(N, dtype=torch.long) # roles for each agent. 0 = ego, 1 = neighbors
        self.roles[0] = 0 # ego is always the first element
        self.roles.expand(B,-1) # expand to include batch dimension
        x = x + self.role_embed(self.roles)# Add role embedding 0 - for ego, 1 - for all other (position-type embedding)

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B,N,states)

        logits = logits[:, 0, :] # Only care about ego scores

        return logits