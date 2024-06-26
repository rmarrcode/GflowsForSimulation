"""
base class from https://github.com/ray-project/ray/blob/master/rllib/models/torch/fcnet.py
"""
import torch.nn as nn
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops
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
from model.gat import GraphAttentionLayer

from torch.distributions import Categorical

local_action_move = env_setup.act.MOVE_LOOKUP

class SamplerGNN(TMv2.TorchModelV2, nn.Module):
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

        self.logZ = nn.Parameter(torch.ones(1))

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

        # # what does this do
        # self._last_flat_in = obs.reshape(obs.shape[0], -1)

        # # TODO: revisit this 
        # # How does decision work??? What needs to be done
        # logits = torch.nn.functional.log_softmax(probs, dim=1)
        # dist = Categorical(logits)
        # sample = dist.sample().tolist()[0]
        # action = [self.convert_discrete_action_to_multidiscrete(sample)]
        
        #return (logits[0][sample], action)
        return probs[0]
    
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
        # self._last_flat_in = obs.reshape(obs.shape[0], -1)

        # logits = torch.nn.functional.log_softmax(probs, dim=1)
        # dist = Categorical(logits)
        # sample = dist.sample().tolist()[0]
        # action = [self.convert_discrete_action_to_multidiscrete(sample)]
        
        # return (logits[0][sample], action)

        return probs[0]
    
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

class SamplerAttnFCN(nn.Module):
    def __init__(
        self,
        self_size,
        num_hiddens_action,
        num_outputs_action,
        out_features,
        n_heads,
        map,
        embedding,
        is_dynamic_embedding,
        **kwargs
    ):
        nn.Module.__init__(self)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.self_size = self_size
        self.num_hiddens_action = num_hiddens_action
        self.num_outputs_action = num_outputs_action
        self.out_features = out_features
        self.n_heads = n_heads
        self.embedding = embedding
        self.is_dynamic_embedding = is_dynamic_embedding
        self.map = map

        # acurate???
        #self.reward_nodes = [2]

        adj_matrix = torch.tensor(nx.adjacency_matrix(self.map.g_acs).toarray(), device=self.device)
        self.adj_matrix = adj_matrix.reshape((27, 27, 1))

        if self.embedding == "number":
            in_features = self_size
        elif self.embedding == "coordinate":
            in_features = 2

        self.mlp_forward = nn.Sequential(
            nn.Linear(in_features+2, num_hiddens_action, dtype=float),
            nn.LeakyReLU(),
            nn.Linear(num_hiddens_action, num_outputs_action, dtype=float))
        self.mlp_backward = nn.Sequential(
            nn.Linear(self_size, num_hiddens_action, dtype=float), 
            nn.LeakyReLU(),
            nn.Linear(num_hiddens_action, num_outputs_action, dtype=float))
        
        self.logZ = nn.Parameter(torch.ones(1))

        n_hidden = 32 
        dropout = 0.6
        self.layer1 = GraphAttentionLayer(in_features+1, n_hidden, n_heads, is_concat=True, dropout=dropout)
        self.activation = nn.ELU()
        self.layer2 = GraphAttentionLayer(n_hidden, n_hidden, n_heads, is_concat=True, dropout=dropout)
        self.output = GraphAttentionLayer(n_hidden, in_features+1, 1, is_concat=False, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        

    def update_reward(self, reward_nodes):
        if self.embedding == "number":
            self.dynamic_embedding = torch.stack([
                torch.cat(
                    (   
                        F.one_hot(torch.tensor(node, device=self.device), self.self_size),
                        torch.tensor([1.0] if (node + 1) in reward_nodes else [0.0], dtype=float, device=self.device)
                    ), dim=0
                )
                for node in range(self.self_size)
            ])  
        elif self.embedding == "coordinate":
            self.dynamic_embedding = torch.stack([
                torch.cat(
                    (   
                        torch.tensor([self.map.n_info[node+1][0], self.map.n_info[node+1][1]], device=self.device),
                        torch.tensor([1.0] if (node + 1) in reward_nodes else [0.0], dtype=float, device=self.device)
                    ), dim=0
                )
                for node in range(self.self_size)
            ])  
        else:
            #TODO: throw some error 
            pass

        self.to(self.device)

    def convert_discrete_action_to_multidiscrete(self, action):
        return [action % len(local_action_move), action // len(local_action_move)]

    def forward(self, obs, reward_nodes):
        
        x = self.dropout(self.dynamic_embedding)
        x = self.layer1(x, self.adj_matrix)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer2(x, self.adj_matrix)
        x = self.activation(x)
        x = self.output(x, self.adj_matrix)

        # TODO: Examine and fix
        if self.is_dynamic_embedding:
            self.dynamic_embedding = x.clone()

        bool_obs = obs.bool()[0]
        cur_node = utils.get_loc(bool_obs, self.self_size) + 1

        reward_nodes = torch.stack([torch.tensor([1.0] if (node + 1) in reward_nodes else [0.0], dtype=float, device=self.device) for node in range(self.self_size)])
        final_embeddings = torch.cat((self.dynamic_embedding, reward_nodes), dim=1)
        probs = self.mlp_forward(final_embeddings[cur_node-1])

        return probs

    def backward(
        self,
        obs,
    ):  
        self_size = self.self_size
        self_obs = obs[0][:self_size].double()
        probs = self.mlp_backward(self_obs)

        return probs
    
    def flow(
        self,
        obs,
    ):
        return 0

class SamplerFCNSimple(nn.Module):
    def __init__(
        self,
        num_hiddens,
        num_outputs,
        embedding,
        map
    ):
        nn.Module.__init__(self)

        self.num_hiddens = num_hiddens
        self.num_outputs = num_outputs
        self.embedding = embedding
        self.map = map

        if self.embedding == "coordinate":
            self.embedding_size = 2
        if self.embedding == "number":
            self.embedding_size = 27

        # is softmax needed for coordinate embedding
        self.mlp_forward = nn.Sequential(nn.Linear(self.embedding_size, num_hiddens, dtype=float),
                                 nn.LeakyReLU(),
                                 nn.Linear(num_hiddens, num_outputs, dtype=float))
        self.mlp_backward = nn.Sequential(nn.Linear(self.embedding_size, num_hiddens, dtype=float), 
                                 nn.LeakyReLU(),
                                 nn.Linear(num_hiddens, num_outputs, dtype=float))
        
        self.logZ = nn.Parameter(torch.ones(1))

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.to(self.device)

    def convert_discrete_action_to_multidiscrete(self, action):
        return [action % len(local_action_move), action // len(local_action_move)]

    def forward(
        self,
        obs,
    ):
        self_obs = obs[0][:27].double()
        bool_obs = obs.bool()[0]
        cur_node = utils.get_loc(bool_obs, 27)

        if self.embedding == "number":
            self.embedding = self_obs
        elif self.embedding == "coordinate":
            map_tensor = torch.tensor([self.map.n_info[cur_node][0], self.map.n_info[cur_node][1]], dtype=float, device=self.device)
            self.embedding = map_tensor

        probs = self.mlp_forward(self.embedding)

        return probs
    
    def backward(
        self,
        obs,
    ):
        self_obs = obs[0][:27].double()
        bool_obs = obs.bool()[0]
        cur_node = utils.get_loc(bool_obs, 27) + 1

        if self.embedding == "number":
            self.embedding = self_obs
        elif self.embedding == "coordinate":
            self.embedding = torch.tensor([self.map.n_info[cur_node][0], self.map.n_info[cur_node][1]], dtype=float, device=self.device)

        probs = self.mlp_backward(self.embedding)

        return probs
    
    def flow(
        self,
        obs,
    ):
        return 0

class SamplerFig8GATCoordinateTime(nn.Module):
    def __init__(
        self,
        num_hiddens_action,
        num_outputs_action,
        trajectory_length,
        map
    ):
        nn.Module.__init__(self)
        self.trajectory_length = trajectory_length
        self.in_features = 2*trajectory_length+1
        self.num_hiddens_action = num_hiddens_action
        self.num_outputs_action = num_outputs_action
        self.n_heads = 1
        self.map = map

        self.mlp_forward = nn.Sequential(
            nn.Linear(self.in_features, self.num_hiddens_action, dtype=float),
            nn.LeakyReLU(),
            nn.Linear(self.num_hiddens_action, self.num_outputs_action, dtype=float))
        
        self.logZ = nn.Parameter(torch.ones(1))

        n_hidden = 32 
        dropout = 0.6
        self.layer1 = GraphAttentionLayer(in_features=self.in_features, out_features=n_hidden, n_heads=1, is_concat=True, dropout=dropout)
        self.activation = nn.ELU()
        self.layer2 = GraphAttentionLayer(in_features=n_hidden, out_features=n_hidden, n_heads=1, is_concat=True, dropout=dropout)
        self.output = GraphAttentionLayer(in_features=n_hidden, out_features=self.in_features, n_heads=1, is_concat=False, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def convert_discrete_action_to_multidiscrete(self, action):
        return [action % len(local_action_move), action // len(local_action_move)]

    def reset_state(self):
        self.cur_gflow_state = torch.zeros(2 * self.trajectory_length + 1, device=self.device, dtype=float)

    def forward(self, 
                obs, 
                cur_blue,
                red_neighbors,
                prev_red,
                prev_blue,
                next_blue,
                step_counter):

        node_embeddings_lst = []
        is_next_red = not self.trajectory_length - 1 == step_counter

        prev_state = self.cur_gflow_state.clone()
        prev_state[(step_counter-1)*2] = 0.
        prev_state[(step_counter-1)*2+1] = 0.
        prev_state[-1] = prev_red == prev_blue
        
        if prev_blue:
            node_embeddings_lst.append(prev_state)

        bool_obs = obs.bool()[0]
        cur_node = utils.get_loc(bool_obs, 27) + 1
        self.embedding = self.cur_gflow_state.clone()
        self.embedding[step_counter*2] = self.map.n_info[cur_node][0]
        self.embedding[step_counter*2+1] = self.map.n_info[cur_node][1]
        self.embedding[-1] = cur_blue == cur_node
        self.cur_gflow_state = self.embedding.clone()

        node_embeddings_lst.append(self.cur_gflow_state)

        if is_next_red:
            next_steps = [torch.cat(
                    (   
                        self.cur_gflow_state[:step_counter*2+2],
                        torch.tensor([self.map.n_info[n][0], self.map.n_info[n][1]], dtype=float, device=self.device),
                        torch.zeros((2*self.trajectory_length)-(2*step_counter+4), dtype=float, device=self.device),
                        torch.tensor([1.0] if n == next_blue else [0.0], dtype=float, device=self.device)
                    ), dim=0
                )
            for n in red_neighbors]
            node_embeddings_lst = node_embeddings_lst + next_steps
        
        node_embeddings = torch.stack(node_embeddings_lst)
        no_nodes = node_embeddings.shape[0]

        adj_matrix = torch.zeros((no_nodes, no_nodes, 1), device=self.device)
        if prev_blue:
            adj_matrix[0][1] = 1
            adj_matrix[1][0] = 1
            if is_next_red:
                for i in range(2, no_nodes):
                    adj_matrix[1][i] = 1
                    adj_matrix[i][1] = 1
        else:
            for i in range(1, no_nodes):
                adj_matrix[0][i] = 1
                adj_matrix[i][0] = 1

        x = self.dropout(node_embeddings)
        x = self.layer1(x, adj_matrix)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer2(x, adj_matrix)
        x = self.activation(x)
        x = self.output(x, adj_matrix)

        probs = self.mlp_forward(x[0])

        return probs

    def backward(
        self,
    ):  
        probs = torch.tensor( ([1/27]*27), device=self.device, dtype=float )

        return probs
    
    def flow(
        self,
        obs,
    ):
        return 0

class SamplerFcnCoordinateTime(nn.Module):    
    def __init__(
        self,
        num_hiddens,
        num_outputs,
        embedding,
        trajectory_length,
        map
    ):
        nn.Module.__init__(self)

        self.embedding_size = 2 * trajectory_length

        self.num_hiddens = num_hiddens
        self.num_outputs = num_outputs
        self.map = map
        self.embedding = embedding
        self.trajectory_length = trajectory_length

        self.mlp_forward = nn.Sequential(
                                nn.Linear(self.embedding_size, num_hiddens, dtype=float),
                                nn.LeakyReLU(),
                                nn.Linear(num_hiddens, num_outputs, dtype=float),
                                nn.ReLU())
        self.mlp_backward = nn.Sequential(
                                nn.Linear(27, num_hiddens, dtype=float), 
                                nn.LeakyReLU(),
                                nn.Linear(num_hiddens, num_outputs, dtype=float))
        
        self.logZ = nn.Parameter(torch.tensor([10], dtype=float))

        self.gflow_state = torch.zeros(2 * self.trajectory_length)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.to(self.device)

    def convert_discrete_action_to_multidiscrete(self, action):
        return [action % len(local_action_move), action // len(local_action_move)]

    def reset_state(self):
        self.gflow_state = torch.zeros(2 * self.trajectory_length, device=self.device, dtype=float)

    def forward(
        self,
        obs,
        step_counter
    ):
        bool_obs = obs.bool()[0]
        cur_node = utils.get_loc(bool_obs, 27) + 1
        self.embedding = self.gflow_state.clone()
        self.embedding[step_counter*2] = self.map.n_info[cur_node][0]
        self.embedding[step_counter*2+1] = self.map.n_info[cur_node][1]

        self.gflow_state = self.embedding.clone()

        probs = self.mlp_forward(self.gflow_state)

        return probs
    
    def backward(
        self,
        obs
    ):
        # self_size = 27
        # self_obs = obs[0][:self_size].double()
        # probs = self.mlp_backward(self_obs)
        probs = torch.tensor( ([1/27]*27), device=self.device, dtype=float )

        return probs
    
    def flow(
        self,
        obs,
    ):
        return 0
    
class SamplerFCNFig8(nn.Module):
    def __init__(
        self,
        num_hiddens,
        num_outputs,
        embedding,
        trajectory_length,
        map
    ):
        nn.Module.__init__(self)
        if embedding == "coordinate":
            self.embedding_size = 2
        elif embedding == "number":
            self.embedding_size = 27

        self.embedding_size += trajectory_length

        self.num_hiddens = num_hiddens
        self.num_outputs = num_outputs
        self.map = map
        self.embedding = embedding
        self.trajectory_length = trajectory_length

        self.mlp_forward = nn.Sequential(
                                nn.Dropout(),
                                nn.Linear(self.embedding_size, num_hiddens, dtype=float),
                                nn.LeakyReLU(),
                                nn.Dropout(),
                                nn.Linear(num_hiddens, num_outputs, dtype=float),
                                nn.ReLU())
        self.mlp_backward = nn.Sequential(
                                nn.Dropout(),
                                nn.Linear(self.embedding_size, num_hiddens, dtype=float), 
                                nn.LeakyReLU(),
                                nn.Linear(num_hiddens, num_outputs, dtype=float))
        
        self.logZ = nn.Parameter(torch.tensor([1]))

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.to(self.device)

    def convert_discrete_action_to_multidiscrete(self, action):
        return [action % len(local_action_move), action // len(local_action_move)]

    def forward(
        self,
        obs,
        step_counter
    ):
        self_obs = obs[0][:27].double()
        bool_obs = obs.bool()[0]
        cur_node = utils.get_loc(bool_obs, 27) + 1
        if self.embedding == "number":
            embedding = torch.cat(
                    (   
                        self_obs,
                        F.one_hot(torch.tensor(step_counter, device=self.device),self.trajectory_length)
                    ), dim=0
                )

        # focusing on fig 8 for now where there is no reward node
        elif self.embedding == "coordinate":
            embedding = torch.cat(
                        (
                            torch.tensor([self.map.n_info[cur_node][0], self.map.n_info[cur_node][1]], dtype=float, device=self.device),
                            F.one_hot(torch.tensor(step_counter, device=self.device), self.trajectory_length)
                        ), dim=0
                    )

        probs = self.mlp_forward(embedding)

        return probs
    
    def backward(
        self,
        obs,
        step_counter
    ):
        self_obs = obs[0][:27].double()
        bool_obs = obs.bool()[0]
        cur_node = utils.get_loc(bool_obs, 27) + 1
        if self.embedding == "number":
            embedding = torch.cat(
                    (   
                        self_obs,
                        F.one_hot(torch.tensor(step_counter, device=self.device),self.trajectory_length)
                    ), dim=0
                )

        # focusing on fig 8 for now where there is no reward node
        elif self.embedding == "coordinate":
            embedding = torch.cat(
                        (
                            torch.tensor([self.map.n_info[cur_node][0], self.map.n_info[cur_node][1]], dtype=float, device=self.device),
                            F.one_hot(torch.tensor(step_counter, device=self.device), self.trajectory_length)
                        ), dim=0
                    )

        probs = self.mlp_backward(embedding)

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


    
