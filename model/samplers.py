"""
base class from https://github.com/ray-project/ray/blob/master/rllib/models/torch/fcnet.py
"""
import torch.nn as nn
import torch
import ray.rllib.models.torch.torch_modelv2 as TMv2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict
import gym

import dgl
from torch_geometric.nn.conv import GATv2Conv, GCNConv
from torch_geometric.nn.norm import BatchNorm
import networkx as nx
import numpy as np

from sigma_graph.data.graph.skirmish_graph import MapInfo
from sigma_graph.envs.figure8 import default_setup as env_setup
from sigma_graph.envs.figure8.action_lookup import MOVE_LOOKUP
from sigma_graph.envs.figure8.figure8_squad_rllib import Figure8SquadRLLib
import model.utils as utils

from torch.distributions import Categorical

local_action_move = env_setup.act.MOVE_LOOKUP

class Sampler(TMv2.TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        map: MapInfo,
        **kwargs
    ):
        TMv2.TorchModelV2.__init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
        )
        nn.Module.__init__(self)

        """
        values that we need to instantiate and use GNNs
        """
        utils.set_obs_token(kwargs["graph_obs_token"])
        (
            hiddens,
            activation,
            no_final_linear,
            self.vf_share_layers,
            self.free_log_std,
        ) = utils.parse_config(model_config)
        self.map = map
        self.num_red = kwargs["nred"]
        self.num_blue = kwargs["nblue"]
        self.aggregation_fn = kwargs["aggregation_fn"]
        self.hidden_size = kwargs["hidden_size"]
        self.is_hybrid = kwargs["is_hybrid"]  # is this a hybrid model or a gat-only model?
        self.conv_type = kwargs["conv_type"]
        self.layernorm = kwargs["layernorm"]
        self_shape, blue_shape, red_shape = env_setup.get_state_shapes(
            self.map.get_graph_size(),
            self.num_red,
            self.num_blue,
            env_setup.OBS_TOKEN,
        )
        self.obs_shapes = [
            self_shape,
            blue_shape,
            red_shape,
            self.num_red,
            self.num_blue,
        ]
        
        self._features = None  # current "base" output before logits
        self._last_flat_in = None  # last input

        """
        instantiate policy and value networks
        """
        self.GAT_LAYERS = 4
        self.N_HEADS = 1 if self.conv_type == "gcn" else 4
        self.HIDDEN_DIM = 4
        self.hiddens = [self.hidden_size, self.hidden_size] # TODO TEMP removed //2
        #self.hiddens = [169, 169]
        gat = GATv2Conv if self.conv_type == "gat" else GCNConv
        self.gats = nn.ModuleList([
            gat(
                in_channels=utils.NODE_EMBED_SIZE if i == 0 else self.HIDDEN_DIM*self.N_HEADS,
                out_channels=self.HIDDEN_DIM,
                heads=self.N_HEADS,
            )
            for i in range(self.GAT_LAYERS)
        ])
        if self.layernorm:
            self.norms = nn.ModuleList([
                BatchNorm(len(list(self.map.g_acs.adj.keys())))
                for _ in range(self.GAT_LAYERS)
            ])
        else:
            self.norms = [None]*self.GAT_LAYERS
        self.aggregator = utils.GeneralGNNPooling(
            aggregator_name=self.aggregation_fn,
            input_dim=self.HIDDEN_DIM*self.N_HEADS,
            output_dim=15
        )
        self.aggregator_backward = utils.GeneralGNNPooling(
            aggregator_name=self.aggregation_fn,
            input_dim=self.HIDDEN_DIM*self.N_HEADS,
            output_dim=15
        )
        self.aggregator_flow = utils.GeneralGNNPooling(
            aggregator_name=self.aggregation_fn,
            input_dim=self.HIDDEN_DIM*self.N_HEADS,
            output_dim=15
        )
        self._hiddens, self._logits = utils.create_policy_fc(
            hiddens=self.hiddens,
            activation=activation,
            num_outputs=num_outputs,
            no_final_linear=no_final_linear,
            num_inputs=85+num_outputs,
        )
        self._hiddens_backward, self._logits_backward = utils.create_policy_fc(
            hiddens=self.hiddens,
            activation=activation,
            num_outputs=num_outputs,
            no_final_linear=no_final_linear,
            num_inputs=85+num_outputs,
        )
        self._hiddens_flow, self._logits_flow = utils.create_policy_fc(
            hiddens=self.hiddens,
            activation=activation,
            num_outputs=1,
            no_final_linear=no_final_linear,
            num_inputs=85+num_outputs,
        )
        self._value_branch, self._value_branch_separate = utils.create_value_branch(
            num_inputs=85,
            num_outputs=num_outputs,
            vf_share_layers=self.vf_share_layers,
            activation=activation,
            hiddens=utils.VALUE_HIDDENS,
        )

        """
        produce debug output and ensure that model is on right device
        """
        #utils.count_model_params(self, print_model=True)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.adjacency = []
        for n in map.g_acs.adj:
            ms = map.g_acs.adj[n]
            for m in ms:
                self.adjacency.append([n-1, m-1])
        self.adjacency = torch.LongTensor(self.adjacency).t().contiguous()
        self.adjacency = self.adjacency.to(self.device)

        self.to(self.device)

    def convert_discrete_action_to_multidiscrete(self, action):
        return [action % len(local_action_move), action // len(local_action_move)]

    @override(TMv2.TorchModelV2)
    def forward(
        self,
        obs,
    ):
        x = utils.efficient_embed_obs_in_map(obs, self.map, self.obs_shapes)
        agent_nodes = [utils.get_loc(gx, self.map.get_graph_size()) for gx in obs]
        
        # inference
        for conv, norm in zip(self.gats, self.norms):
            x = torch.stack([conv(_x, self.adjacency) for _x in x], dim=0)
            if self.layernorm: x = norm(x)
        self._features = self.aggregator(x, self.adjacency, agent_nodes=agent_nodes)
        if self.is_hybrid:
            self._features = self._hiddens(torch.cat([self._features, obs], dim=1))
        probs = self._logits(self._features)

        # what does this do
        self._last_flat_in = obs.reshape(obs.shape[0], -1)

        # TODO: revisit this 
        # How does decision work??? What needs to be done
        logits = torch.nn.functional.log_softmax(probs, dim=1)
        dist = Categorical(logits)
        sample = dist.sample().tolist()[0]
        action = [self.convert_discrete_action_to_multidiscrete(sample)]
        
        return (logits[0][sample], action)
    
    def backward(
        self,
        obs,
    ):
        
        x = utils.efficient_embed_obs_in_map(obs, self.map, self.obs_shapes)
        agent_nodes = [utils.get_loc(gx, self.map.get_graph_size()) for gx in obs]
        
        # inference
        for conv, norm in zip(self.gats, self.norms):
            x = torch.stack([conv(_x, self.adjacency) for _x in x], dim=0)
            if self.layernorm: x = norm(x)
        self._features_backward = self.aggregator_backward(x, self.adjacency, agent_nodes=agent_nodes)
        if self.is_hybrid:
            self._features_backward = self._hiddens_backward(torch.cat([self._features_backward, obs], dim=1))
        probs = self._logits_backward(self._features_backward)

        # what does this do
        self._last_flat_in = obs.reshape(obs.shape[0], -1)

        logits = torch.nn.functional.log_softmax(probs, dim=1)
        dist = Categorical(logits)
        sample = dist.sample().tolist()[0]
        action = [self.convert_discrete_action_to_multidiscrete(sample)]
        
        return (logits[0][sample], action)
    
    def flow(
        self,
        obs,
    ):
        x = utils.efficient_embed_obs_in_map(obs, self.map, self.obs_shapes)
        agent_nodes = [utils.get_loc(gx, self.map.get_graph_size()) for gx in obs]
        
        # inference
        for conv, norm in zip(self.gats, self.norms):
            x = torch.stack([conv(_x, self.adjacency) for _x in x], dim=0)
            if self.layernorm: x = norm(x)
        self._features_flow = self.aggregator_flow(x, self.adjacency, agent_nodes=agent_nodes)
        if self.is_hybrid:
            self._features_flow = self._hiddens_flow(torch.cat([self._features_flow, obs], dim=1))
        prob = self._logits_flow(self._features_flow).log()
        
        return prob
    
    
class SamplerFCN(nn.Module):
    def __init__(
        self,
        self_size,
        num_hiddens,
        num_outputs
    ):
        self.self_size = self_size
        self.num_hiddens = num_hiddens
        self.num_outputs = num_outputs

        nn.Module.__init__(self)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.mlp_forward = nn.Sequential(nn.Linear(self_size, num_hiddens, dtype=float),
                                 nn.LeakyReLU(),
                                 nn.Linear(num_hiddens, num_outputs, dtype=float))
        self.mlp_backward = nn.Sequential(nn.Linear(self_size, num_hiddens, dtype=float), 
                                 nn.LeakyReLU(),
                                 nn.Linear(num_hiddens, num_outputs, dtype=float))

        self.to(self.device)

    def convert_discrete_action_to_multidiscrete(self, action):
        return [action % len(local_action_move), action // len(local_action_move)]

    def forward(
        self,
        obs,
    ):
        self_size = self.self_size
        self_obs = obs[0][:self_size].double()
        features = self.mlp_forward(self_obs)
        # double check activation
        probs = torch.nn.functional.softmax(features, dim=0)

        # sample = Categorical(probs).sample()
        # action = self.convert_discrete_action_to_multidiscrete(sample)
        # action[0] = action[0].cpu().tolist()
        # action[1] = action[1].cpu().tolist()

        return probs
    
    def backward(
        self,
        obs,
    ):
        
        self_size = self.self_size
        self_obs = obs[0][:self_size].double()
        features = self.mlp_backward(self_obs)
        # double check activation
        probs = torch.nn.functional.softmax(features, dim=0)

        # sample = Categorical(probs).sample()
        # # good idea to keep moving it
        # action = self.convert_discrete_action_to_multidiscrete(sample)
        # action[0] = action[0].cpu().tolist()
        # action[1] = action[1].cpu().tolist()

        return probs
    
    def flow(
        self,
        obs,
    ):
        return 0
    

class SimpleNetwork(nn.Module):
    def __init__(self):
        super(SimpleNetwork, self).__init__()
        self.linear1 = nn.Linear(27, 512)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
