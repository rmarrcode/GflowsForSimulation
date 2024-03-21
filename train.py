import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import tqdm
import matplotlib.pyplot as pp
import numpy as np

from sigma_graph.envs.figure8.action_lookup import MOVE_LOOKUP, TURN_90_LOOKUP
from sigma_graph.envs.figure8.default_setup import OBS_TOKEN
from sigma_graph.envs.figure8.figure8_squad_rllib import Figure8SquadRLLib
from sigma_graph.envs.figure8.gflow_figure8_squad import GlowFigure8Squad
#from graph_scout.envs.base import ScoutMissionStdRLLib
import sigma_graph.envs.figure8.default_setup as default_setup
import model  # THIS NEEDS TO BE HERE IN ORDER TO RUN __init__.py!
import model.utils as utils
import model.gnn_gflow 
from trajectory import Trajectory
import losses
import torch.optim as optim
import wandb
import json

INPUT_DIMS = 27
OUTPUT_DIMS = 5
NUM_EPOCHS = 100000
TRAJECTORY_LENGTH = 20
BATCH_SIZE = 100
START_NODE = 25
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
    if state == 10:
        return 2
    if state == 2:
        return 1
    return 0

def convert_discrete_action_to_multidiscrete(action):
        return [action % len(local_action_move), action // len(local_action_move)]

# Investigate loss rewar mirror
# Try real reward
# Make code cleaner 
# visualize flows 

if WANDB:
    wandb.init(
        project="graph-training-simulation",
        config={
            "learning_rate": LEARNING_RATE,
            "epocs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE
        }
    )

config = {
    "custom_model": "attn_fcn", #fcn #attn_fcn
    "reward": "random_region", #random_region
    "custom_model_config": {
        "nred": 1,
        "nblue": 1,
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
LEARNING_RATE = 3e-4
gflowfigure8 = GlowFigure8Squad(sampler_config=config)
# sampler_fcn -> sampler
# TODO: this is bad revisit and come up with something more concise

optimizer = optim.AdamW(gflowfigure8.sampler.parameters(), lr=LEARNING_RATE)

# Using sigma code fully

torch.autograd.set_detect_anomaly(True)

minibatch_loss = 0
minibatch_reward = 0
minibatch_z = 0
minibatch_pf = 0
minibatch_pb = 0

for episode in tqdm.tqdm(range(NUM_EPOCHS), ncols=40):
  
  TEMP_AGENT_INDEX = 0
  gflowfigure8._reset_agents()
  
  total_P_F = 0
  total_P_B = 0
  total_reward = 0
  for t in range(TRAJECTORY_LENGTH):

    step = gflowfigure8.step(TEMP_AGENT_INDEX)  
    total_P_F += step['forward_prob']
    total_P_B += step['backward_prob']
    #total_reward += torch.tensor(gflowfigure8._step_reward_test())
    total_reward += step['step_reward']

  logZ = gflowfigure8.sampler.logZ
  # TODO find more elegant solution to nan issue
  clipped_reward = torch.log(torch.tensor(total_reward).clip(0)).clip(-20)
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