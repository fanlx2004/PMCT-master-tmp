import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict, deque
import random
import math
from conf import conf

'''
    These functions define the structure of the Q-network that solves the state set partition problem, 
    and a DRL training agent that performs the training and verification of the Q-network.
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

    

class Encoder(nn.Module):
    def __init__(self, input_dim, encoded_dim):
        super().__init__()
        self.input_dim = input_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim*2, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, encoded_dim)
        )
        
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = np.array([np.array(e, dtype=np.float32) for e in x])
            x_tensor = torch.FloatTensor(x)
        else:
            x_tensor = x  
            
        mask = (~torch.isnan(x_tensor)).float()
        x_filled = torch.where(mask.bool(), x_tensor, torch.zeros_like(x_tensor))
        combined = torch.cat([x_filled, mask], dim=1)
        return self.net(combined)

class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(conf.STATE_DIM, conf.ENCODED_DIM)
        self.value_head = nn.Sequential(
            nn.Linear(conf.ENCODED_DIM, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1))
        self._init_output_layer()

    def _init_output_layer(self):
        output_layer = self.value_head[-1]
        nn.init.xavier_normal_(output_layer.weight, gain=0.1) 
        nn.init.constant_(output_layer.bias, float(conf.K+1))
    
    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = np.array(state, dtype=float)
            if state.ndim == 1:
                state = state[np.newaxis, :] 
            state = torch.FloatTensor(state).to(device)
            
        features = self.encoder(state)
        return self.value_head(features).squeeze(-1)


class CategorizedReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffers = defaultdict(lambda: deque(maxlen=capacity)) 
    
    def push(self, s_t, s_t1, r):
        self.buffers[int(r)].append((s_t, s_t1, r))
    
    def sample_r(self, r, batch_size):
        buffer = self.buffers.get(r, deque())
        if not buffer:
            return []
        if len(buffer) >= batch_size:
            return random.sample(buffer, batch_size)
        else:
            return random.choices(buffer, k=batch_size)
    
    def sample_balanced(self, batch_size):

        available_rs = [r for r in self.buffers.keys() if self.buffers[r]]
        if not available_rs:
            return None

        buffer_sizes = [len(self.buffers[r]) for r in available_rs]
        total_size = sum(buffer_sizes)
        
        allocations = []
        for size in buffer_sizes:
            alloc = math.ceil(batch_size * size / total_size)
            allocations.append(alloc)

        samples = []
        if len(samples) < batch_size:
            r_0 = random.choice(available_rs)
            samples.extend(self.sample_r(r_0, batch_size - len(samples)))
        random.shuffle(samples)
        return samples[:batch_size]

    def clear(self):
        self.buffers = defaultdict(lambda: deque(maxlen=self.capacity))

class DRLAgent:
    def __init__(self):
        self.q_net = QNetwork().to(device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=conf.LR)
        self.buffer = CategorizedReplayBuffer(conf.BUFFER_SIZE)
        
    def get_q_value(self, state):

     with torch.no_grad():
        if isinstance(state, (list, np.ndarray)) and np.array(state).ndim == 1:
            state = [state]
        
        state_arr = np.array(state, dtype=float)
        
        for i in range(len(state_arr)):

            first_element = state_arr[i, 0]
            
            for j in range(conf.STATE_DIM):  
                ## an operation to handle NaN values ##
                skip_processing = False
                if abs(first_element - 0) < 1e-6 and j in [10, 11, 12, 13]:
                    skip_processing = True
                
                if abs(first_element - 2) < 1e-6 and j in [6, 7, 8, 9]:
                    skip_processing = True

                if np.isnan(state_arr[i, j]) and not skip_processing:
                    mod = j % 4
                    if mod == 0:
                        state_arr[i, j] = -115
                    elif mod == 1:
                        state_arr[i, j] = 20
                    elif mod == 2:
                        state_arr[i, j] = 115
                    elif mod == 3:
                        state_arr[i, j] = 40
                        
        state_tensor = torch.FloatTensor(state_arr).to(device)
        
        q0 = self.q_net(state_tensor).to('cpu')
        q0 = q0.item() if q0.numel() == 1 else q0.numpy()
        
        result = np.where(q0 >= conf.K+0.5, 
                          conf.K+1, 
                          np.clip(np.floor(q0), 1, conf.K+1).astype(int))
        
        return result
                
    
    def update(self):
        batch = self.buffer.sample_balanced(conf.BATCH_SIZE)
        if not batch:
            return None
        
        s_t, s_t1, r = zip(*batch)
        
        s_t = torch.FloatTensor(np.array(s_t, dtype=float)).to(device)
        s_t1 = torch.FloatTensor(np.array(s_t1, dtype=float)).to(device)
        r = torch.FloatTensor(np.array(r)).to(device)
        
        q0_t = self.q_net(s_t)
        q0_t1 = self.q_net(s_t1).detach()
        
        mask = torch.isnan(s_t1).all(dim=1) 

        q0_t1[mask] = 0
        
        target = torch.clamp(torch.minimum(r, q0_t1 + 1), min=1, max=conf.K+1)
        error = q0_t - target
        loss = torch.mean(error ** 2)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
  

            
            
    

