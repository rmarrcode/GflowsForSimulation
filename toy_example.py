#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import tqdm
import matplotlib.pyplot as pp
import numpy as np

from sigma_graph.data.file_manager import check_dir, find_file_in_dir, load_graph_files
import matplotlib.pyplot as plt
import networkx as nx
import wandb

OUTPUT_DIMS = 5
NUM_EPOCHS = 500000
TRAJECTORY_LENGTH = 10
BATCH_SIZE = 100
START_NODE = 21
LEARNING_RATE = 3e-4

transitions = {
    # 0: no-op 
    # 1: N 
    # 2: S 
    # 3: W
    # 4: E
    1: {0: 1, 1: 2, 2: 1, 3: 3, 4: 1},
    2: {0: 2, 1: 12, 2: 1, 3: 2, 4: 2},
    3: {0: 3, 1: 3, 2: 3, 3: 4, 4: 1},
    4: {0: 4, 1: 5, 2: 4, 3: 6, 4: 3},
    5: {0: 5, 1: 13, 2: 4, 3: 5, 4: 5},
    6: {0: 6, 1: 6, 2: 6, 3: 7, 4: 4},
    7: {0: 7, 1: 8, 2: 7, 3: 9, 4: 6},
    8: {0: 8, 1: 14, 2: 7, 3: 8, 4: 8},
    9: {0: 9, 1: 9, 2: 9, 3: 10, 4: 7},
    10: {0: 10, 1: 11, 2: 10, 3: 10, 4: 9},
    11: {0: 11, 1: 15, 2: 10, 3: 16, 4: 11},
    12: {0: 12, 1: 18, 2: 2, 3: 12, 4: 12},
    13: {0: 13, 1: 19, 2: 5, 3: 13, 4: 13},
    14: {0: 14, 1: 20, 2: 8, 3: 14, 4: 14},
    15: {0: 15, 1: 21, 2: 11, 3: 17, 4: 15},
    16: {0: 16, 1: 17, 2: 16, 3: 16, 4: 11},
    17: {0: 17, 1: 22, 2: 16, 3: 23, 4: 15},
    18: {0: 18, 1: 18, 2: 12, 3: 25, 4: 18},
    19: {0: 19, 1: 19, 2: 13, 3: 26, 4: 25},
    20: {0: 20, 1: 20, 2: 14, 3: 27, 4: 26},
    21: {0: 21, 1: 21, 2: 15, 3: 22, 4: 27},
    22: {0: 22, 1: 22, 2: 17, 3: 24, 4: 21},
    23: {0: 23, 1: 24, 2: 17, 3: 23, 4: 17},
    24: {0: 24, 1: 22, 2: 23, 3: 24, 4: 22},
    25: {0: 25, 1: 25, 2: 25, 3: 19, 4: 18},
    26: {0: 26, 1: 26, 2: 26, 3: 20, 4: 19},
    27: {0: 27, 1: 27, 2: 27, 3: 21, 4: 20}
}

wandb.init(
    project="graph-training-simulation",
    config={
            "exp_config": {
                "learning_rate": LEARNING_RATE,
                "epocs": NUM_EPOCHS,
                "batch_size": BATCH_SIZE
        }
    },
)

# Model
class TBModel(nn.Module):
  def __init__(self, num_hid):
    super().__init__()

    self.mlp_forward = nn.Sequential(nn.Linear(TRAJECTORY_LENGTH, num_hid), 
                                     nn.LeakyReLU(),
                                     nn.Linear(num_hid, OUTPUT_DIMS))
    
    self.mlp_backward = nn.Sequential(nn.Linear(TRAJECTORY_LENGTH, num_hid), 
                                      nn.LeakyReLU(),
                                      nn.Linear(num_hid, OUTPUT_DIMS))
    
    self.logZ = nn.Parameter(torch.ones(1))

  def forward(self, x):
    P_F = self.mlp_forward(x)
    return P_F
  
def partial_trajectory(trajectory):
  print(trajectory)
  goal_trajectory = [21, 27, 20, 14, 8, 7, 9, 10, 11, 16]
  reward = 0
  for i in range(len(trajectory)):
    if trajectory[i] == goal_trajectory[i]:
      reward = reward + 1
  print(reward)
  return torch.tensor(reward)

def all_or_nothing_trajectory(trajectory):
  print(trajectory)
  goal_trajectory = [21, 27, 20, 14, 8]
  # reward = 0
  # for i in range(len(trajectory)):
  #   if trajectory[i] == goal_trajectory[i]:
  #     reward = reward + 1
  # return torch.tensor(reward)
  if trajectory == goal_trajectory:
    return torch.tensor([1])
  return torch.tensor([0])

def end_goal(trajectory):
  print(trajectory)
  if trajectory[-1] == 8:
    return torch.tensor([1])
  return torch.tensor([0])


model = TBModel(512)
opt = torch.optim.Adam(model.parameters(),  3e-4)

tb_losses = []
tb_rewards = []
logZs = []
minibatch_loss = 0
minibatch_reward = 0

for episode in tqdm.tqdm(range(NUM_EPOCHS), ncols=40):
  
  gflow_state = torch.zeros(TRAJECTORY_LENGTH)
  state = START_NODE
  total_P_F = 0
  total_P_B = 0
  trajectory = []

  for t in range(TRAJECTORY_LENGTH):
    trajectory.append(state)
    P_F_s = model.forward(gflow_state)
    P_B_s = torch.tensor([(1/5)]) #model.backward(gflow_state)

    cat = Categorical(logits=P_F_s)
    action = cat.sample()
    _gflow_state = gflow_state.clone()
    _gflow_state[t] = action
    gflow_state = _gflow_state.clone()

    new_state = transitions[state][action.item()]
    total_P_F += cat.log_prob(action)
    total_P_B += torch.log(P_B_s) #Categorical(logits=P_B_s).log_prob(action)

    state = new_state

  reward = partial_trajectory(trajectory)

  loss = (model.logZ + total_P_F - torch.log(reward).clip(-20) - total_P_B).pow(2)
  
  minibatch_loss += loss
  minibatch_reward += reward

  if (episode + 1) % BATCH_SIZE == 0:
    wandb.log({
      "loss": minibatch_loss.item(),
      "reward": minibatch_reward.item()/BATCH_SIZE,
      "Z": model.logZ.item()
    })
    minibatch_loss.backward()
    opt.step()
    opt.zero_grad()
    minibatch_loss = 0
    minibatch_reward = 0

