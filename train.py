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
import random
import argparse




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

def run(config):

    TRAJECTORY_LENGTH = config['exp_config']['trajectory_length']
    NUM_EPOCHS = config['exp_config']['num_epochs']
    BATCH_SIZE = config['exp_config']['batch_size']
    START_NODE = config['exp_config']['start_node']
    LEARNING_RATE = config['exp_config']['learning_rate']
    WANDB = config['exp_config']['wandb']

    run_name = f"{config['custom_model_config']['custom_model']}-{config['custom_model_config']['reward']}-{config['custom_model_config']['embedding']}"

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

    LEARNING_RATE = 3e-4
    gflowfigure8 = GlowFigure8Squad(sampler_config=config)
    # sampler_fcn -> sampler
    # TODO: this is bad revisit and come up with something more concise

    optimizer = optim.AdamW(gflowfigure8.sampler.parameters(), lr=LEARNING_RATE)

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

        region_nodes = {16, 15, 11, 10, 9}

        if config['custom_model_config']['reward'] == 'random':
            reward_node_1i = [random.randint(1, 27)]
        elif config['custom_model_config']['reward'] == 'random_region':
            reward_node_1i = random.sample(region_nodes, 1)
        elif config['custom_model_config']['reward'] == '10':
            reward_node_1i = [10]
        elif config['custom_model_config']['reward'] == 'complex':
            reward_node_1i = []
                    
        for t in range(TRAJECTORY_LENGTH):

            step = gflowfigure8.step(TEMP_AGENT_INDEX, reward_node_1i)  
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

    torch.save(gflowfigure8, f'{run_name}.pt')

def separate_args(config):
    organized_config = {
        "custom_model_config": {
            "custom_model": config['custom_model'], #fcn #attn_fcn
            "reward": config['reward'], #random_region random 10 complex
            "embedding": config['embedding'], #number #coordinate
            "nred": config['nred'],
            "nblue": config['nblue'],
            "aggregation_fn":config['aggregation_fn'],
            "hidden_size": config['hidden_size'],
            "is_hybrid": config['is_hybrid'],
            "conv_type": config['conv_type'],
            "layernorm": config['layernorm'],
            "graph_obs_token": {"embed_opt": config['embed_opt'], "embed_dir": config['embed_dir']},
        },
        "env_config": {
            "env_path": config['env_path'],
            "act_masked": config['act_masked'],
            "init_red": config['init_red'],
            "init_blue": config['init_blue'],
            "init_health_red": config['init_health_red'],
            "init_health_blue": config['init_health_blue'],
            "obs_embed": config['obs_embed'],
            "obs_dir": config['obs_dir'],
            "obs_team": config['obs_team'],
            "obs_sight": config['obs_sight'],
            "log_on": config['log_on'],
            "log_path": config['log_path'],
            "fixed_start": config['fixed_start'],
            "penalty_stay": config['penalty_stay'],
            "threshold_damage_2_blue": config['threshold_damage_2_blue'],
            "threshold_damage_2_red": config['threshold_damage_2_red'],
        },
        "exp_config": {
            "trajectory_length": config['trajectory_length'],
            "num_epochs": config['num_epochs'],
            "batch_size": config['batch_size'],
            "start_node": config['start_node'],
            "learning_rate": config['learning_rate'],
            "wandb": config['wandb'],
        }
    }
    return organized_config


def parse_arguments():

    parser = argparse.ArgumentParser(description='Custom Model Configuration')
    
    # Custom Model Config Arguments
    parser.add_argument('--custom_model', type=str, default='attn_fcn')
    parser.add_argument('--reward', type=str, default='complex')
    parser.add_argument('--embedding', type=str, default='coordinate')
    parser.add_argument('--nred', type=int, default=1)
    parser.add_argument('--nblue', type=int, default=1)
    parser.add_argument('--aggregation_fn', type=str, default='agent_node')
    parser.add_argument('--hidden_size', type=int, default=15)
    parser.add_argument('--is_hybrid', type=bool, default=False)
    parser.add_argument('--conv_type', type=str, default='gcn')
    parser.add_argument('--layernorm', type=bool, default=False)
    parser.add_argument('--embed_opt', type=bool, default=False)
    parser.add_argument('--embed_dir', type=bool, default=True)

    # Environment Config Arguments
    parser.add_argument('--env_path', type=str, default='.')
    parser.add_argument('--act_masked', type=bool, default=True)
    parser.add_argument('--init_red', type=str, default=None)
    parser.add_argument('--init_blue', type=str, default=None)
    parser.add_argument('--init_health_red', type=int, default=20)
    parser.add_argument('--init_health_blue', type=int, default=20)
    parser.add_argument('--obs_embed', type=bool, default=False)
    parser.add_argument('--obs_dir', type=bool, default=False)
    parser.add_argument('--obs_team', type=bool, default=True)
    parser.add_argument('--obs_sight', type=bool, default=False)
    parser.add_argument('--log_on', type=bool, default=True)
    parser.add_argument('--log_path', type=str, default='logs/temp/')
    parser.add_argument('--fixed_start', type=int, default=-1)
    parser.add_argument('--penalty_stay', type=int, default=0)
    parser.add_argument('--threshold_damage_2_blue', type=int, default=2)
    parser.add_argument('--threshold_damage_2_red', type=int, default=5)

    # Experiment Config args
    parser.add_argument('--trajectory_length', type=int, default=20)
    parser.add_argument('--num_epochs', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=100)
    # TODO pretty sure this doesn't work
    parser.add_argument('--start_node', type=int, default=25)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--wandb', type=bool, default=False)
    
    args = parser.parse_args()
    
    # Convert Namespace object to dictionary
    return separate_args(vars(args))

if __name__ == "__main__":
    arguments_dict = parse_arguments()
    run(arguments_dict)