#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as pp
import numpy as np
from ipywidgets import interact, IntSlider, fixed

from sigma_graph.data.file_manager import load_graph_files, save_log_2_file, log_done_reward
from sigma_graph.envs.figure8.action_lookup import MOVE_LOOKUP, TURN_90_LOOKUP
from sigma_graph.envs.figure8.default_setup import OBS_TOKEN
from sigma_graph.envs.figure8.figure8_squad_rllib import Figure8SquadRLLib
from sigma_graph.envs.figure8.gflow_figure8_squad import GlowFigure8Squad
#from graph_scout.envs.base import ScoutMissionStdRLLib
import sigma_graph.envs.figure8.default_setup as default_setup
from sigma_graph.data.file_manager import check_dir, find_file_in_dir, load_graph_files
import model  # THIS NEEDS TO BE HERE IN ORDER TO RUN __init__.py!
import model.utils as utils
import model.gnn_gflow 
from trajectory import Trajectory
import losses

import torch.optim as optim
import wandb
import json
import random
import networkx as nx
import matplotlib.pyplot as plt

NUM_EPOCHS = 100000
# default = 34
BATCH_SIZE = 100
LEARNING_RATE = 3e-4
WANDB = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

local_action_move = {
    0: "NOOP",
    1: "N",
    2: "S",
    3: "W",
    4: "E",
}

def state_to_vec(state):
    result = [0]*27
    result[state-1] = 1
    return torch.tensor(result).float()

def compute_reward(state):
    if state == 17:
        return 1
    return 0

def convert_discrete_action_to_multidiscrete(action):
        return [action % len(local_action_move), action // len(local_action_move)]


# In[2]:


# Investigate loss rewar mirror
# Try real reward
# Make code cleaner 
# visualize flows 

config = {
    "custom_model_config": {
        "custom_model": "fcn", #fcn #attn_fcn
        "reward": "complex", #random_region random single complex
        "reward_interval": "step", #trajectory 
        "trajectory_per_reward": 1,
        "embedding": "number", #number #coordinate
        "is_dynamic_embedding": False,
        "trajectory_length": 5,
        "nred": 1,
        "nblue": 1,
        "start_node": 22,
        "aggregation_fn": "agent_node",
        "hidden_size": 15,
        "is_hybrid": False,
        "conv_type": "gcn",
        "layernorm": False,
        "graph_obs_token": {"embed_opt": False, "embed_dir": True},
    },
    "env_config": {
        "env_path": ".",
        "act_masked": True,
        "init_red": None,
        "init_blue": None,
        "init_health_red": 20,
        "init_health_blue": 20,
        "obs_embed": False,
        "obs_dir": False,
        "obs_team": True,
        "obs_sight": False,
        "log_on": True,
        "log_path": "logs/temp/",
        "fixed_start": -1,
        "penalty_stay": 0,
        "threshold_damage_2_blue": 2,
        "threshold_damage_2_red": 5,
    },
}

current_time = datetime.now()
run_name = f"{config['custom_model_config']['custom_model']}-{config['custom_model_config']['reward']}-{config['custom_model_config']['embedding']}-{current_time.strftime('%Y-%m-%d %H:%M:%S')}"
print(run_name)

if WANDB:
    wandb.init(
        project="graph-training-simulation",
        config={
                "model_config": config,
                "exp_config": {
                    "learning_rate": LEARNING_RATE,
                    "epocs": NUM_EPOCHS,
                    "batch_size": BATCH_SIZE
            }
        },
        name=run_name
    )

gflowfigure8 = GlowFigure8Squad(sampler_config=config)
optimizer = optim.AdamW(gflowfigure8.sampler_fig8_gat_coordinate_time.parameters(), lr=LEARNING_RATE)


# In[3]:


# FCN coordinate time

minibatch_loss = 0
minibatch_reward = 0
minibatch_z = 0
minibatch_pf = 0
minibatch_pb = 0

pbar = tqdm(total=NUM_EPOCHS)
episode = 0

while episode <= NUM_EPOCHS:
  
  TEMP_AGENT_INDEX = 0
  
  total_P_F = 0
  total_P_B = 0
  total_reward = 0
    
  trajectory_length = config['custom_model_config']['trajectory_length']

  trajectory_path = []
  action_path = []
  gflowfigure8.reset()
  gflowfigure8.reset_state_gat_coordinate_time()
  for t in range(trajectory_length):
    step = gflowfigure8.step_gat_coordinate_time(TEMP_AGENT_INDEX)  
    total_P_F += step['forward_prob']
    total_P_B += step['backward_prob']
    total_reward += step['step_reward']
    trajectory_path.append(step['red_node'])
    action_path.append(step['action'])

  logZ = gflowfigure8.sampler_fcn_coordinate_time.logZ
    
  clipped_reward = torch.log(torch.tensor(total_reward)).clip(-20)
  #last_node = gflowfigure8.team_red[0].get_info()["node"]
  #clipped_reward = torch.log(torch.tensor(compute_reward(last_node))).clip(-20)

  loss = (logZ + total_P_F - clipped_reward - total_P_B).pow(2)
  
  minibatch_loss += loss
  minibatch_reward += clipped_reward
  minibatch_z += logZ
  minibatch_pf += total_P_F
  minibatch_pb += total_P_B

  if (episode + 1) % BATCH_SIZE == 0:
    if WANDB:
      wandb.log({
          "loss": minibatch_loss/BATCH_SIZE, 
          "reward":  minibatch_reward/BATCH_SIZE,
          "pf": minibatch_pf/BATCH_SIZE,
          "pb": minibatch_pb/BATCH_SIZE,
          "z": minibatch_z/BATCH_SIZE
        })
      # for name, param in sampler.named_parameters():
      #     wandb.log({f"{name}_mean": param.data.mean().item(), f"{name}_std": param.data.std().item()})
    
    minibatch_loss.backward(retain_graph=True)
    optimizer.step()
    optimizer.zero_grad()
    minibatch_loss = 0
    minibatch_reward = 0
    minibatch_z = 0 
    minibatch_pf = 0 
    minibatch_pb = 0 

  pbar.update(1)
  episode = episode + 1


# In[ ]:


torch.save(gflowfigure8, f'models/{run_name}.pt')


# In[ ]:


gflowfigure8.reset()
red_path = []
blue_path = []
step_rewards = []
total_reward = 0
for r in range(34):   
    step = gflowfigure8.step_fcn_coordinate_time(0)
    red_path.append(step['red_node'])
    blue_path.append(step['blue_node'])
    step_rewards.append(step['step_reward'])
    total_reward += step['step_reward']

print(f'step_rewards {total_reward}')
episode_reward = gflowfigure8._episode_rewards_aggressive()[0]
print(f'episode_reward {episode_reward}')
total_reward += episode_reward
print(f'total_reward {total_reward}')


# In[ ]:


map_info, _ = load_graph_files(map_lookup="S")
col_map = ["gold"] * len(map_info.n_info)

def display_graph(index):
    fig, ax = plt.subplots(figsize=(8, 6))
    cur_col_map = col_map[:]
    cur_col_map[red_path[index]-1] = "red"
    cur_col_map[blue_path[index]-1] = "blue"
    nx.draw_networkx(map_info.g_acs, pos=map_info.n_info, node_color=cur_col_map, edge_color="blue", arrows=True, ax=ax)
    ax.set_title(f"step rewards {step_rewards[index]}")
    plt.axis('off')
    plt.show()

# Create an interactive widget to display different graphs
slider = IntSlider(min=0, max=33-1, step=1, value=0, description='Graph Index')
interact(display_graph, index=slider)


# In[ ]:


gflowfigure8.reset()
red_path = []
blue_path = []
step_rewards = []
total_reward = 0
for r in range(34):   
    step = gflowfigure8.step_fcn_coordinate_time(0)
    red_path.append(step['red_node'])
    blue_path.append(step['blue_node'])
    step_rewards.append(step['step_reward'])
    total_reward += step['step_reward']

print(f'step_rewards {total_reward}')
episode_reward = gflowfigure8._episode_rewards_aggressive()[0]
print(f'episode_reward {episode_reward}')
total_reward += episode_reward
print(f'total_reward {total_reward}')


# In[ ]:


map_info, _ = load_graph_files(map_lookup="S")
col_map = ["gold"] * len(map_info.n_info)

def display_graph(index):
    fig, ax = plt.subplots(figsize=(8, 6))
    cur_col_map = col_map[:]
    cur_col_map[red_path[index]-1] = "red"
    cur_col_map[blue_path[index]-1] = "blue"
    print(map_info.g_acs)
    print(map_info.n_info)
    nx.draw_networkx(map_info.g_acs, pos=map_info.n_info, node_color=cur_col_map, edge_color="blue", arrows=True, ax=ax)
    ax.set_title(f"step rewards {step_rewards[index+1]}")
    plt.axis('off')
    plt.show()

# Create an interactive widget to display different graphs
slider = IntSlider(min=0, max=33, step=1, value=0, description='Graph Index')
interact(display_graph, index=slider)

