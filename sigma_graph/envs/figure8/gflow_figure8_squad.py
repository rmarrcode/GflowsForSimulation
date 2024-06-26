import sys
import numpy as np
import random

from  model.samplers import *

from sigma_graph.data.file_manager import load_graph_files, save_log_2_file, log_done_reward
from sigma_graph.data.graph.skirmish_graph import MapInfo
from sigma_graph.data.data_helper import get_emb_from_name

#from sigma_graph.envs.utils.multiagent_space import ActionSpaces, ObservationSpaces
from ..utils.multiagent_space import ActionSpaces, ObservationSpaces
from .agents.skirmish_agents import AgentRed, AgentBlue
from .rewards.rewards_simple import get_step_engage, get_step_overlay, get_episode_reward_agent, get_episode_reward_agent_agressive
from . import default_setup as env_setup

import gym
from gym import spaces
import torch
import torch.nn as nn

from torch.distributions import Categorical

local_action_move = env_setup.act.MOVE_LOOKUP
local_action_turn = env_setup.act.TURN_90_LOOKUP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GlowFigure8Squad():
    def __init__(self, sampler_config, max_step=40, n_red=1, n_blue=1):#, **kwargs):
        self.sampler_config = sampler_config
        # setup configs
        self.max_step = max_step
        self.num_red = n_red
        self.num_blue = n_blue
        self.step_counter = 0
        self.done_counter = 0
        # "in_eval" will be in kwargs if we are in eval mode. check train.py:create_trainer_config for more information.
        self.in_eval = "in_eval" in sampler_config #kwargs
        self.n_eval_episodes = 0

        # manage environment config arguments
        self.configs = {}
        self.logs = {}
        self.obs_token = {}
        self.rewards = {}
        self.action_mask = []
        # load default local configs and parse outer arguments
        self.logger = True
        self.invalid_masked = True
        self._init_env_config(**sampler_config['env_config']) #kwargs)

        # load env map and patrol routes (loading connectivity & visibility graphs)
        self.map = MapInfo()
        
        self.routes = []  # list of RouteInfo instances
        self._load_map_data()
        
        # generate lists of agent instances
        self.team_red = []
        self.team_blue = []
        self.learning_agent = []  # default training agents are agents in team red
        self._init_agents()

        # init spaces
        self.__action_space = ActionSpaces([spaces.MultiDiscrete([len(local_action_move), len(local_action_turn)])
                                          for _ in range(len(self.learning_agent))])
        # <default>: all agents have identical observation shape
        self.state_shape = env_setup.get_state_shape(self.map.get_graph_size(), n_red, n_blue, self.obs_token)
        # agent obs size: [self: pos(6or27)+dir(4)+flags(2*B) +teamB: pos+next_move(4)+flags(2*B) +teamR: 6*(n_R-1)or27]
        self.__observation_space = ObservationSpaces([spaces.Box(low=0, high=1, shape=(self.state_shape,), dtype=np.int8)
                                                    for _ in range(len(self.learning_agent))])
        self.obs = [[] for _ in range(len(self.learning_agent))]
        self.states = [[] for _ in range(len(self.learning_agent))]
        self.start_node = sampler_config['custom_model_config']['start_node']
        #self.positions = [[] for _ in range(len(self.learning_agent))]

        self.reward_type = sampler_config['custom_model_config']['reward']
        self.custom_model = sampler_config['custom_model_config']['custom_model']
        self.embedding = sampler_config['custom_model_config']['embedding']
        self.is_dynamic_embedding = sampler_config['custom_model_config']['is_dynamic_embedding']
        self.trajectory_length = sampler_config['custom_model_config']['trajectory_length']

        self.sampler_fcn_simple = SamplerFCNSimple(
            num_hiddens=512,
            num_outputs=15,
            embedding=sampler_config["custom_model_config"]["embedding"],
            map = self.map
        )
        self.sampler_fcn_coordinate_time = SamplerFcnCoordinateTime(
            num_hiddens=512,
            num_outputs=15,
            embedding=self.embedding,
            trajectory_length=self.trajectory_length,
            map=self.map
        )
        self.sampler_fig8_gat_coordinate_time = SamplerFig8GATCoordinateTime(
            num_hiddens_action=512,
            num_outputs_action=15,
            trajectory_length=self.trajectory_length,
            map=self.map
        )
        self.total_reward = 0
        if sampler_config['custom_model_config']['custom_model'] == 'gnn':
            self.sampler = SamplerGNN(
                obs_space=self.__observation_space,
                action_space=self.__action_space,
                num_outputs=15,
                model_config={},
                name='str',
                map=self.map,
                graph_obs_token=sampler_config['custom_model_config']['graph_obs_token'],
                nred=sampler_config['custom_model_config']['nred'],
                nblue=sampler_config['custom_model_config']['nblue'],
                aggregation_fn=sampler_config['custom_model_config']['aggregation_fn'],
                hidden_size=sampler_config['custom_model_config']['hidden_size'],
                is_hybrid=sampler_config['custom_model_config']['is_hybrid'],
                conv_type=sampler_config['custom_model_config']['conv_type'],
                layernorm=sampler_config['custom_model_config']['layernorm'],
                activation=torch.nn.functional.log_softmax
            )
        if sampler_config['custom_model_config']['custom_model'] == 'gnn_custom':
            self.sampler = SamplerGCNCustom(
                map=self.map,
                nred=sampler_config['custom_model_config']['nred'],
                nblue=sampler_config['custom_model_config']['nblue']
            )
        elif sampler_config['custom_model_config']['custom_model'] == 'attn_fcn':
            self.sampler = SamplerAttnFCN(
                self_size=27,
                num_hiddens_action=512,
                num_outputs_action=15,
                out_features=28,
                n_heads=1,
                map=self.map,
                nred=sampler_config['custom_model_config']['nred'],
                nblue=sampler_config['custom_model_config']['nblue'],
                embedding = self.embedding,
                is_dynamic_embedding = self.is_dynamic_embedding
            )
        
    def reset(self, force=False):
        if self.in_eval:
            self.n_eval_episodes += 1
        self.step_counter = 0
        if force:
            self.done_counter = 0
        self._reset_agents()
        self._update()
        return np.array(self.states, dtype=np.int8)

    def reset_step(self):
        self.step_counter = 0

    def _get_step_done(self):
        # reach to max_step
        if self.step_counter >= self.max_step:
            return [True] * self.num_red
        # all Blue agents got terminated
        if all([self.team_blue[_b].get_health() <= 0 for _b in range(self.num_blue)]):
            return [True] * self.num_red
        # done for each Red agent
        return [self.team_red[_r].get_health() <= 0 for _r in range(self.num_red)]

    def convert_discrete_action_to_multidiscrete(self, action):
        return [action % len(local_action_move), action // len(local_action_move)]

    # TODO write this better
    def probs_to_action(self, probs):
        cat = Categorical(logits=probs)
        action = cat.sample()
        action_prob = cat.log_prob(action)
        action = self.convert_discrete_action_to_multidiscrete(action)
        action[0] = action[0].cpu().tolist()
        action[1] = action[1].cpu().tolist()
        return (action_prob, [action])



    # use observation from state var
    # make actions
    # update state 
    # update observation
    def step(self, a_id, reward_nodes):

        red_node = self.team_red[0].get_info()["node"]
        blue_node = self.team_blue[0].get_info()["node"]

        prev_obs = self._log_step_prev()
        
        R_engage_B, B_engage_R, R_overlay = self._update()

        if self.sampler_config['custom_model_config']['custom_model'] == 'complex':
            probs_forward = self.sampler.forward(torch.tensor(np.array([self.states[a_id],], dtype=np.int8), device=device),
                                                 self.step_counter)
        else:
            probs_forward = self.sampler.forward(torch.tensor(np.array([self.states[a_id],], dtype=np.int8), device=device),
                                                reward_nodes=reward_nodes)

        # TODO fix this
        cat = Categorical(logits=probs_forward)
        discrete_action = cat.sample()
        forward_prob = cat.log_prob(discrete_action)
        action = self.convert_discrete_action_to_multidiscrete(discrete_action)
        action[0] = action[0].cpu().tolist()
        action[1] = action[1].cpu().tolist()
        action = [action]

        # TODO check if backward action is its own thing
        if self.sampler_config['custom_model_config']['custom_model'] == 'fcn':
            probs_backward = self.sampler.backward(torch.tensor(np.array([self.states[a_id],], dtype=np.int8), device=device),
                                                 self.step_counter)
        else:
            probs_backward = self.sampler.backward(torch.tensor(np.array([self.states[a_id],], dtype=np.int8), device=device),
                                                reward_nodes=reward_nodes)
            
        backward_prob = Categorical(logits=probs_backward).log_prob(discrete_action)
        flow = self.sampler.flow(torch.tensor(np.array([self.states[a_id],], dtype=np.int8), device=device))

        # update log
        self._log_step_update(prev_obs, [action,], [0,])

        action_penalty_red = self._take_action_red(action)
        self._take_action_blue()
        R_engage_B, B_engage_R, R_overlay = self._update()
        self.agent_interaction(R_engage_B, B_engage_R)

        self.step_counter += 1

        if self.reward_type == 'complex':
            step_reward = self._step_rewards_aggresssive(action_penalty_red, R_engage_B, B_engage_R, R_overlay)[0] 
            #print(f'step_reward {step_reward}')
        else:
            step_reward = self._step_reward_test(reward_nodes)

        n_done = self._get_step_done()

        return ({
            'done': False,
            'forward_prob': forward_prob,
            'backward_prob': backward_prob,
            'flow': flow,
            'action': action,
            'step_reward': step_reward,
            'red_node': red_node,
            'blue_node': blue_node
        })
    
    def reset_state_fcn_coordinate_time(self):
        self.reset()
        self.sampler_fcn_coordinate_time.reset_state()
        self.total_reward = 0

    def step_fcn_coordinate_time(self, a_id):
        red_node = self.team_red[0].get_info()["node"]
        blue_node = self.team_blue[0].get_info()["node"]
        prev_obs = self._log_step_prev()
        R_engage_B, B_engage_R, R_overlay = self._update()
        obs = torch.tensor(np.array([self.states[a_id],], dtype=np.int8), device=device)

        probs_forward = self.sampler_fcn_coordinate_time.forward(obs, self.step_counter)

        norms_prob_forward = F.normalize(probs_forward, p=1, dim=0)
        discrete_action = torch.multinomial(norms_prob_forward, 1, replacement=True)[0]
        forward_prob = torch.log(norms_prob_forward[discrete_action])

        # cat = Categorical(logits=probs_forward)
        # discrete_action = cat.sample()
        # forward_prob = cat.log_prob(discrete_action)

        action = self.convert_discrete_action_to_multidiscrete(discrete_action)
        action[0] = action[0].cpu().tolist()
        action[1] = action[1].cpu().tolist()
        action = [action]

        probs_backward = self.sampler_fcn_coordinate_time.backward(obs)
        backward_prob = Categorical(logits=probs_backward).log_prob(discrete_action)

        flow = self.sampler_fcn_coordinate_time.flow(torch.tensor(np.array([self.states[a_id],], dtype=np.int8), device=device))

        # update log
        self._log_step_update(prev_obs, [action,], [0,])
        action_penalty_red = self._take_action_red(action)
        self._take_action_blue()
        R_engage_B, B_engage_R, R_overlay = self._update()
        self.agent_interaction(R_engage_B, B_engage_R)

        self.step_counter += 1

        step_reward = int(red_node == blue_node) #int(R_overlay[0])
        self.total_reward += step_reward
        #step_reward = self._step_rewards_aggresssive(action_penalty_red, R_engage_B, B_engage_R, R_overlay)[0] 

        return ({
            'done': False,
            'forward_prob': forward_prob,
            'backward_prob': backward_prob,
            'flow': flow,
            'action': action,
            'step_reward': step_reward,
            'red_node': red_node,
            'blue_node': blue_node
        })

    def step_fcn_coordinate_time_deterministic(self, a_id):
        red_node = self.team_red[0].get_info()["node"]
        blue_node = self.team_blue[0].get_info()["node"]
        prev_obs = self._log_step_prev()
        R_engage_B, B_engage_R, R_overlay = self._update()
        obs = torch.tensor(np.array([self.states[a_id],], dtype=np.int8), device=device)

        probs_forward = self.sampler_fcn_coordinate_time.forward(obs, self.step_counter)

        max_val, discrete_action = torch.max(probs_forward, dim=0)

        # norms_prob_forward = F.normalize(probs_forward, p=1, dim=0)
        # discrete_action = torch.multinomial(norms_prob_forward, 1, replacement=True)[0]
        # forward_prob = torch.log(norms_prob_forward[discrete_action])

        # cat = Categorical(logits=probs_forward)
        # discrete_action = cat.sample()
        # forward_prob = cat.log_prob(discrete_action)

        action = self.convert_discrete_action_to_multidiscrete(discrete_action)
        action[0] = action[0].cpu().tolist()
        action[1] = action[1].cpu().tolist()
        action = [action]

        probs_backward = self.sampler_fcn_coordinate_time.backward(obs)
        backward_prob = Categorical(logits=probs_backward).log_prob(discrete_action)

        flow = self.sampler_fcn_coordinate_time.flow(torch.tensor(np.array([self.states[a_id],], dtype=np.int8), device=device))

        # update log
        self._log_step_update(prev_obs, [action,], [0,])
        action_penalty_red = self._take_action_red(action)
        self._take_action_blue()
        R_engage_B, B_engage_R, R_overlay = self._update()
        self.agent_interaction(R_engage_B, B_engage_R)

        self.step_counter += 1

        step_reward = int(red_node == blue_node) #int(R_overlay[0])
        self.total_reward += step_reward
        #step_reward = self._step_rewards_aggresssive(action_penalty_red, R_engage_B, B_engage_R, R_overlay)[0] 

        return ({
            'done': False,
            'backward_prob': backward_prob,
            'flow': flow,
            'action': action,
            'step_reward': step_reward,
            'red_node': red_node,
            'blue_node': blue_node
        })

    
    def reset_state_gat_coordinate_time(self):
        self.reset()
        self.sampler_fig8_gat_coordinate_time.reset_state()
        self.total_reward = 0
        self.prev_blue = None
        self.step_counter = 0

    # cur time is defined by conditions before function iscalled
    def step_gat_coordinate_time(self, a_id):
        red_node = self.team_red[0].get_info()["node"]
        blue_node = self.team_blue[0].get_info()["node"]

        agent_encoding = self.team_red[0].agent_code
        # is prev_red real????
        prev_red, list_neighbors, list_acts = self.map.get_all_states_by_node(agent_encoding)

        _route = self.team_blue[0].get_route()
        _idx = (self.step_counter + self.blue_offsets[0] + 1) % self.routes[_route].get_route_length()
        next_blue_node, _, _ = self.routes[_route].get_location_by_index(_idx)

        prev_obs = self._log_step_prev()
        R_engage_B, B_engage_R, R_overlay = self._update()
        obs = torch.tensor(np.array([self.states[a_id],], dtype=np.int8), device=device)

        probs_forward = self.sampler_fig8_gat_coordinate_time.forward(
                                                                obs=obs, 
                                                                cur_blue=blue_node,
                                                                red_neighbors=list_neighbors,
                                                                prev_red=prev_red,
                                                                prev_blue=self.prev_blue,
                                                                next_blue=next_blue_node,
                                                                step_counter=self.step_counter)

        # norms_prob_forward = F.normalize(probs_forward, p=1, dim=0)
        # discrete_action = torch.multinomial(norms_prob_forward, 1, replacement=True)[0]
        # forward_prob = torch.log(norms_prob_forward[discrete_action])

        cat = Categorical(logits=probs_forward)
        discrete_action = cat.sample()
        forward_prob = cat.log_prob(discrete_action)

        action = self.convert_discrete_action_to_multidiscrete(discrete_action)
        action[0] = action[0].cpu().tolist()
        action[1] = action[1].cpu().tolist()
        action = [action]

        probs_backward = self.sampler_fcn_coordinate_time.backward(obs)
        backward_prob = Categorical(logits=probs_backward).log_prob(discrete_action)

        flow = self.sampler_fcn_coordinate_time.flow(torch.tensor(np.array([self.states[a_id],], dtype=np.int8), device=device))

        # update log
        self._log_step_update(prev_obs, [action,], [0,])
        action_penalty_red = self._take_action_red(action)
        self._take_action_blue()
        R_engage_B, B_engage_R, R_overlay = self._update()
        self.agent_interaction(R_engage_B, B_engage_R)

        self.prev_blue = blue_node

        self.step_counter += 1

        step_reward = int(red_node == blue_node) #int(R_overlay[0])
        self.total_reward += step_reward
        #step_reward = self._step_rewards_aggresssive(action_penalty_red, R_engage_B, B_engage_R, R_overlay)[0] 

        return ({
            'done': False,
            'forward_prob': forward_prob,
            'backward_prob': backward_prob,
            'flow': flow,
            'action': action,
            'step_reward': step_reward,
            'red_node': red_node,
            'blue_node': blue_node
        })

    def step_gat_coordinate_time_deterministic(self, a_id):
        red_node = self.team_red[0].get_info()["node"]
        blue_node = self.team_blue[0].get_info()["node"]

        agent_encoding = self.team_red[0].agent_code
        # is prev_red real????
        prev_red, list_neighbors, list_acts = self.map.get_all_states_by_node(agent_encoding)

        _route = self.team_blue[0].get_route()
        _idx = (self.step_counter + self.blue_offsets[0] + 1) % self.routes[_route].get_route_length()
        next_blue_node, _, _ = self.routes[_route].get_location_by_index(_idx)

        prev_obs = self._log_step_prev()
        R_engage_B, B_engage_R, R_overlay = self._update()
        obs = torch.tensor(np.array([self.states[a_id],], dtype=np.int8), device=device)

        probs_forward = self.sampler_fig8_gat_coordinate_time.forward(
                                                                obs=obs, 
                                                                cur_blue=blue_node,
                                                                red_neighbors=list_neighbors,
                                                                prev_red=prev_red,
                                                                prev_blue=self.prev_blue,
                                                                next_blue=next_blue_node,
                                                                step_counter=self.step_counter)

        max_val, discrete_action = torch.max(probs_forward, dim=0)

        # norms_prob_forward = F.normalize(probs_forward, p=1, dim=0)
        # discrete_action = torch.multinomial(norms_prob_forward, 1, replacement=True)[0]
        # forward_prob = torch.log(norms_prob_forward[discrete_action])

        # cat = Categorical(logits=probs_forward)
        # discrete_action = cat.sample()
        # forward_prob = cat.log_prob(discrete_action)

        action = self.convert_discrete_action_to_multidiscrete(discrete_action)
        action[0] = action[0].cpu().tolist()
        action[1] = action[1].cpu().tolist()
        action = [action]

        probs_backward = self.sampler_fcn_coordinate_time.backward(obs)
        backward_prob = Categorical(logits=probs_backward).log_prob(discrete_action)

        flow = self.sampler_fcn_coordinate_time.flow(torch.tensor(np.array([self.states[a_id],], dtype=np.int8), device=device))

        # update log
        self._log_step_update(prev_obs, [action,], [0,])
        action_penalty_red = self._take_action_red(action)
        self._take_action_blue()
        R_engage_B, B_engage_R, R_overlay = self._update()
        self.agent_interaction(R_engage_B, B_engage_R)

        self.prev_blue = blue_node

        self.step_counter += 1

        step_reward = int(red_node == blue_node) #int(R_overlay[0])
        self.total_reward += step_reward
        #step_reward = self._step_rewards_aggresssive(action_penalty_red, R_engage_B, B_engage_R, R_overlay)[0] 

        return ({
            'done': False,
            'backward_prob': backward_prob,
            'flow': flow,
            'action': action,
            'step_reward': step_reward,
            'red_node': red_node,
            'blue_node': blue_node
        })


    def step_fcn_simple(self, a_id):

        red_node = self.team_red[0].get_info()["node"]

        prev_obs = self._log_step_prev()
        R_engage_B, B_engage_R, R_overlay = self._update()

        obs = torch.tensor(np.array([self.states[a_id],], dtype=np.int8), device=device)
        probs_forward = self.sampler_fcn_simple.forward(obs)

        # TODO fix this
        cat = Categorical(logits=probs_forward)
        discrete_action = cat.sample()
        forward_prob = cat.log_prob(discrete_action)
        action = self.convert_discrete_action_to_multidiscrete(discrete_action)
        action[0] = action[0].cpu().tolist()
        action[1] = action[1].cpu().tolist()
        action = [action]

        # TODO check if backward action is its own thing
        probs_backward = self.sampler_fcn_simple.backward(obs)
        backward_prob = Categorical(logits=probs_backward).log_prob(discrete_action)

        flow = self.sampler_fcn_simple.flow(obs)

        # update log
        self._log_step_update(prev_obs, [action,], [0,])

        action_penalty_red = self._take_action_red(action)
        self._take_action_blue()
        R_engage_B, B_engage_R, R_overlay = self._update()
        self.agent_interaction(R_engage_B, B_engage_R)

        self.step_counter += 1

        return ({
            'done': False,
            'forward_prob': forward_prob,
            'backward_prob': backward_prob,
            'flow': flow,
            'action': action,
            'red_node': red_node,
        })

    def step_simple(self, a_id):

        node = self.team_red[0].get_info()["node"]

        probs_forward = self.sampler_fcn.forward(torch.tensor(np.array([self.states[a_id],], dtype=np.int8), device=device))
        (forward_prob, action) = self.probs_to_action(probs_forward)
        probs_backward = self.sampler_fcn.backward(torch.tensor(np.array([self.states[a_id],], dtype=np.int8), device=device))
        (backward_prob, _) = self.probs_to_action(probs_backward)
        flow = self.sampler_fcn.flow(torch.tensor(np.array([self.states[a_id],], dtype=np.int8), device=device))

        action_penalty_red = self._take_action_red(action)
        step_reward = self._step_reward_test()

        return ({
            'done': False,
            'forward_prob': forward_prob,
            'backward_prob': backward_prob,
            'flow': flow,
            'action': action,
            'step_reward': step_reward,
            'node': node,
        })

    def _take_action_red(self, n_actions):
        action_penalty = [0] * self.num_red
        _action_stay_penalty = self.configs["penalty_stay"] if "penalty_stay" in self.configs else 0
        for agent_i, actions in enumerate(n_actions):
            # check input action if in range of the desired discrete action space
            assert self.__action_space[agent_i].contains(actions), f"{actions}: action out of range"
            if self.team_red[agent_i].is_frozen():
                continue
            action_move, action_turn = actions
            # find all 1st ordered neighbors of the current node
            agent_encoding = self.team_red[agent_i].agent_code
            prev_node, list_neighbors, list_acts = self.map.get_all_states_by_node(agent_encoding)
            # validate actions with masks
            if action_move != 0 and action_move not in list_acts:
                # if the action_mask turns on in the learning, invalid actions should not appear.
                if self.invalid_masked:
                    assert f"[ActError] action{action_move} node{prev_node} masking{self.action_mask[agent_i]}"
                # if the learning process doesn't have action masking, then invalid Move should be replaced by NOOP.
                else:
                    action_move = 0
                    action_penalty[agent_i] = self.configs["penalty_invalid"]

            # only make 'Turn' when the move action is NOOP
            if action_move == 0:
                if _action_stay_penalty:
                    action_penalty[agent_i] += _action_stay_penalty
                agent_dir = self.team_red[agent_i].agent_dir
                if action_turn == 1:
                    agent_dir = env_setup.act.TURN_L[agent_dir]
                elif action_turn == 2:
                    agent_dir = env_setup.act.TURN_R[agent_dir]
                self.team_red[agent_i].agent_dir = agent_dir
            # make 'Move' and then 'Turn' actions
            elif action_move in list_acts:
                _node = list_neighbors[list_acts.index(action_move)]
                _code = self.map.get_name_by_index(_node)
                _dir = action_move
                # 'Turn' condition for turning left or right
                if action_turn:
                    _dir = env_setup.act.TURN_L[action_move] if action_turn == 1 else env_setup.act.TURN_R[action_move]
                self.team_red[agent_i].set_location(_node, _code, _dir)
        return action_penalty

    def _take_action_blue(self, n_actions=None):
        for agent_i in range(self.num_blue):
            if self.team_blue[agent_i].is_frozen():
                continue
            _route = self.team_blue[agent_i].get_route()
            _idx = (self.step_counter + self.blue_offsets[agent_i]) % self.routes[_route].get_route_length()
            _node, _code, _dir = self.routes[_route].get_location_by_index(_idx)
            self.team_blue[agent_i].update_index(_idx, _node, _code, _dir)
        # return [0] * self.num_blue

    # update local states after all agents finished actions
    def _update(self):
        # generate binary matrices for pair-wised inSight and inRange indicators
        R_see_B = np.zeros((self.num_red, self.num_blue), dtype=np.bool_)
        R_engage_B = np.zeros((self.num_red, self.num_blue), dtype=np.bool_)
        B_see_R = np.zeros((self.num_blue, self.num_red), dtype=np.bool_)
        B_engage_R = np.zeros((self.num_blue, self.num_red), dtype=np.bool_)
        R_nodes = [0] * self.num_red
        R_overlay = [False] * self.num_red
        if self.invalid_masked:
            self.action_mask = [np.zeros(sum(tuple(self.__action_space[_].nvec)), dtype=np.bool_)
                                for _ in range(len(self.learning_agent))]
        for _r in range(self.num_red):
            node_r, _ = self.team_red[_r].get_pos_dir()
            R_nodes[_r] = node_r
            for _b in range(self.num_blue):
                node_b, _ = self.team_blue[_b].get_pos_dir()
                R_see_B[_r, _b] = self.is_in_sight(node_r, node_b, self.team_red[_r].agent_dir)
                R_engage_B[_r, _b] = self.is_in_range(node_r, node_b, self.team_red[_r].agent_dir)
                B_see_R[_b, _r] = self.is_in_sight(node_b, node_r, self.team_blue[_b].agent_dir)
                B_engage_R[_b, _r] = self.is_in_range(node_b, node_r, self.team_blue[_b].agent_dir)
            # update action masking
            if self.invalid_masked:
                # self.action_mask[_r] = np.zeros(sum(tuple(self.__action_space[_r].nvec)), dtype=np.bool_)
                mask_idx = self.learning_agent.index(self.team_red[_r].get_id())
                # masking invalid movements on the given node
                acts = set(local_action_move.keys())
                valid = set(self.map.get_actions_by_node(self.team_red[_r].get_encoding()) + [0])
                invalid = [_ for _ in acts if _ not in valid]
                for masking in invalid:
                    self.action_mask[mask_idx][masking] = True

        # update overlap list for team red
        for _s in range(self.num_red - 1):
            for _t in range(_s + 1, self.num_red):
                if R_nodes[_s] == R_nodes[_t]:
                    R_overlay[_s] = True
                    R_overlay[_t] = True

        # update states for all agents in team red
        look_dir_shape = len(env_setup.ACT_LOOK_DIR)
        _obs_self_dir = self.obs_token["obs_dir"]
        _state_R_dir = []
        # get looking direction encodings for all red agents
        if _obs_self_dir:
            _state_R_dir = np.zeros((self.num_red, look_dir_shape))
            for _r in range(self.num_red):
                _, _dir = self.team_red[_r].get_pos_dir()
                _state_R_dir[_r, (_dir - 1)] = 1

        # get next_move_dir encodings for all blue agents
        _state_B_next = np.zeros((self.num_blue, look_dir_shape))
        for _b in range(self.num_blue):
            _route = self.team_blue[_b].get_route()
            _index = self.team_blue[_b].get_index()
            _dir = self.routes[_route].list_next[_index]
            _state_B_next[_b, (_dir - 1)] = 1

        ''' update state for each agent '''
        if self.obs_token["obs_embed"]:
            # condition: generate binary encoded position embedding for all agents
            _state_R_embed = []
            for agent_i in range(self.num_red):
                _state_R_embed.append(get_emb_from_name(self.team_red[agent_i].get_encoding()))
            _state_B_embed = []
            for agent_i in range(self.num_blue):
                _state_B_embed.append(get_emb_from_name(self.team_blue[agent_i].get_encoding()))

            # concatenate state_self + state_blues (+ state_reds)
            for _r in range(self.num_red):
                _state = []
                _state += _state_R_embed[_r]
                if _obs_self_dir:
                    _state += _state_R_dir[_r, :].tolist()
                if self.obs_token["obs_sight"]:
                    _state += R_see_B[_r, :].tolist()
                if self.obs_token["obs_range"]:
                    _state += R_engage_B[_r, :].tolist()

                # add team blue info
                for _b in range(self.num_blue):
                    _state += _state_B_embed[_b] + _state_B_next[_b, :].tolist()
                if self.obs_token["obs_sight"]:
                    _state += B_see_R[:, _r].tolist()
                if self.obs_token["obs_range"]:
                    _state += B_engage_R[:, _r].tolist()

                # add teammates info if True
                if self.obs_token["obs_team"]:
                    for agent in range(self.num_red):
                        if agent != _r:
                            _state += _state_R_embed[agent]
                self.states[_r] = _state
            # [Debug] test state shape
            # self.states = [[[(_ + self.step_counter) % 2] * self.state_shape] for _ in range(self.num_red)]
        else:
            # condition: multi-hot pos encodings for self, 'team_blue' and 'team_red' in the shape of len(G.all_nodes())
            pos_obs_size = self.map.get_graph_size()
            # concatenate state_self + state_blues (+ state_reds)
            for _r in range(self.num_red):
                # add self position one-hot embedding
                _state = [0] * pos_obs_size
                _state[R_nodes[_r] - 1] = 1
                # add self direction if True
                if _obs_self_dir:
                    _state += _state_R_dir[_r, :].tolist()
                # add self interaction indicators
                if self.obs_token["obs_sight"]:
                    _state += R_see_B[_r, :].tolist()
                if self.obs_token["obs_range"]:
                    _state += R_engage_B[_r, :].tolist()
                
                # add team blue info
                _state_B = [0] * pos_obs_size
                for _b in range(self.num_blue):
                    _node, _ = self.team_blue[_b].get_pos_dir()
                    _state_B[_node - 1] = 1
                    # add next move dir
                    _state_B += _state_B_next[_b, :].tolist()
                if self.obs_token["obs_sight"]:
                    _state_B += B_see_R[:, _r].tolist()
                if self.obs_token["obs_range"]:
                    _state_B += B_engage_R[:, _r].tolist()
                _state += _state_B

                # add teammates pos if True
                if self.obs_token["obs_team"]:
                    _state_R = [0] * pos_obs_size
                    for _agent in range(self.num_red):
                        if _agent != _r:
                            _state_R[R_nodes[_agent] - 1] = 1
                    _state += _state_R
                # update the local state attribute
                self.states[_r] = _state
            # [Debug] test state shape
            # self.states = [[[(_ + self.step_counter) % 2] * self.state_shape] for _ in range(self.num_red)]
        return R_engage_B, B_engage_R, R_overlay

    # update health points for all agents
    def agent_interaction(self, R_engage_B, B_engage_R):
        # update health and damage points for all agents
        _step_damage = env_setup.INTERACT_LOOKUP["engage_behavior"]["damage"]
        for _b in range(self.num_blue):
            for _r in range(self.num_red):
                if R_engage_B[_r, _b]:
                    self.team_red[_r].damage_add(_step_damage)
                    self.team_blue[_b].take_damage(_step_damage)
                if B_engage_R[_b, _r]:
                    #if self.team_red[_r].agent_node == 4 or self.team_red[_r].agent_node == 5: # TODO
                    self.team_red[_r].take_damage(_step_damage)
                    '''this shooting code was commented out before'''
            # update end time for blue agents
            if self.team_blue[_b].get_end_step() > 0:
                continue
            _damage_taken_blue = self.configs["init_health_blue"] - self.team_blue[_b].get_health()
            if _damage_taken_blue >= self.configs["threshold_damage_2_blue"]:
                self.team_blue[_b].set_end_step(self.step_counter)

    
    def _step_rewards(self, penalties, R_engage_B, B_engage_R, R_overlay):
        rewards = penalties
        # TODO: Revisit
        # if self.rewards["step"]["reward_step_on"] is False:
        #     return rewards
        for agent_r in range(self.num_red):
            rewards[agent_r] += get_step_overlay(R_overlay[agent_r], **self.rewards["step"])
            for agent_b in range(self.num_blue):
                rewards[agent_r] += get_step_engage(r_engages_b=R_engage_B[agent_r, agent_b],
                                                    b_engages_r=B_engage_R[agent_b, agent_r],
                                                    team_switch=False, **self.rewards["step"])
        return rewards
    
    def _step_rewards_aggresssive(self, penalties, R_engage_B, B_engage_R, R_overlay):
        rewards_aggressive = {'reward_step_on': True, 'red_2_blue': 4, 'blue_2_red': 0, 'red_overlay': 0}
        rewards = penalties
        for agent_r in range(self.num_red):
            rewards[agent_r] += get_step_overlay(R_overlay[agent_r], **rewards_aggressive)
            for agent_b in range(self.num_blue):
                rewards[agent_r] += get_step_engage(r_engages_b=R_engage_B[agent_r, agent_b],
                                                    b_engages_r=B_engage_R[agent_b, agent_r],
                                                    team_switch=False, **rewards_aggressive)
        return rewards

    def _step_reward_test(self, reward_nodes):
        reward = 0
        for red_i in range(self.num_red):
            # 0 or 1 indexed
            cur_node = self.team_red[red_i].get_info()["node"]
            if cur_node in reward_nodes:
                reward += 100
            # if cur_node == 2:
            #     reward += 1
        return reward

    def _episode_rewards_test(self, trajectory):
        return trajectory.rewards.sum()

    def _episode_rewards(self):
        # gather final states
        _HP_full_r = self.configs["init_health_red"]
        _HP_full_b = self.configs["init_health_blue"]
        _threshold_r = self.configs["threshold_damage_2_red"]
        _threshold_b = self.configs["threshold_damage_2_blue"]

        _health_lost_r = [_HP_full_r - self.team_red[_r].get_health() for _r in range(self.num_red)]
        _damage_cost_r = [self.team_red[_r].damage_total() for _r in range(self.num_red)]
        _health_lost_b = [_HP_full_b - self.team_blue[_b].get_health() for _b in range(self.num_blue)]
        _end_step_b = [self.team_blue[_b].get_end_step() for _b in range(self.num_blue)]

        rewards = [0] * self.num_red
        if self.rewards["episode"]["reward_episode_on"] is False:
            return rewards
        # If any Red agent got terminated, the whole team would not receive the episode rewards
        if any([_health_lost_r[_r] > _threshold_r for _r in range(self.num_red)]):
            return rewards
        for agent_r in range(self.num_red):
            for agent_b in range(self.num_blue):
                rewards[agent_r] += get_episode_reward_agent(_health_lost_r[agent_r], _health_lost_b[agent_b],
                                                             _threshold_r, _threshold_b, _damage_cost_r[agent_r],
                                                             _end_step_b[agent_b], **self.rewards["episode"])
        return rewards
    
    def _episode_rewards_aggressive(self):
        # gather final states
        _HP_full_r = self.configs["init_health_red"]
        _HP_full_b = self.configs["init_health_blue"]
        _threshold_r = self.configs["threshold_damage_2_red"]
        _threshold_b = self.configs["threshold_damage_2_blue"]

        _health_lost_r = [_HP_full_r - self.team_red[_r].get_health() for _r in range(self.num_red)]
        _damage_cost_r = [self.team_red[_r].damage_total() for _r in range(self.num_red)]
        _health_lost_b = [_HP_full_b - self.team_blue[_b].get_health() for _b in range(self.num_blue)]
        _end_step_b = [self.team_blue[_b].get_end_step() for _b in range(self.num_blue)]

        rewards = [0] * self.num_red
        if self.rewards["episode"]["reward_episode_on"] is False:
            return rewards
        # If any Red agent got terminated, the whole team would not receive the episode rewards
        if any([_health_lost_r[_r] > _threshold_r for _r in range(self.num_red)]):
            return rewards
        for agent_r in range(self.num_red):
            for agent_b in range(self.num_blue):
                rewards[agent_r] += get_episode_reward_agent_agressive(_health_lost_r[agent_r], _health_lost_b[agent_b],
                                                             _threshold_r, _threshold_b, _damage_cost_r[agent_r],
                                                             _end_step_b[agent_b], **self.rewards["episode"])
        return rewards


    def _get_step_done(self):
        # reach to max_step
        if self.step_counter >= self.max_step:
            return [True] * self.num_red
        # all Blue agents got terminated
        if all([self.team_blue[_b].get_health() <= 0 for _b in range(self.num_blue)]):
            return [True] * self.num_red
        # done for each Red agent
        return [self.team_red[_r].get_health() <= 0 for _r in range(self.num_red)]

    def is_in_half(self, route_pos_index, route_id):
        return not (route_pos_index < (self.routes[route_id].get_route_length() // 2))

    def is_in_sight(self, s_idx, t_idx, s_dir):
        """ field of view check
            if there is an edge in the visibility FOV graph;
                if so, check if it is inside the sight range
            <!> no self-loop in the visibility graph for now, check if two agents are on the same node first
        """
        if s_idx == t_idx:
            return True
        if self.map.g_vis.has_edge(s_idx, t_idx):
            _distance = self.map.get_edge_attr_vis_fov_by_idx(s_idx, t_idx, s_dir)
            # -1 indicates there is no visibility edge in the 's_dir' direction
            _range = env_setup.INTERACT_LOOKUP["sight_range"]
            if _distance == -1:
                return False
            if _range < 0 or _distance < _range:
                return True
        return False

    def is_in_range(self, s_idx, t_idx, s_dir):
        """ engage behavior indicator check
            if there is an edge in visibility graph;
                if so, check if the distance is below the engaging range limit
        """
        if s_idx == t_idx:
            return True
        if self.map.g_vis.has_edge(s_idx, t_idx):
            _distance = self.map.get_edge_attr_vis_fov_by_idx(s_idx, t_idx, s_dir)
            if _distance == -1:
                return False
            if _distance < env_setup.INTERACT_LOOKUP["engage_range"]:
                return True
        return False

    # load configs and update local defaults [!!] need to know num_red and num_blue
    def _init_env_config(self, **kwargs):
        """ set default env config values if not specified in outer configs """
        from copy import deepcopy
        self.configs = deepcopy(env_setup.INIT_CONFIGS)
        self.obs_token = deepcopy(env_setup.OBS_TOKEN)
        self.rewards = deepcopy(env_setup.INIT_REWARDS)

        _config_local_args = env_setup.INIT_CONFIGS_LOCAL
        _config_args = _config_local_args + list(self.configs.keys())
        _obs_shape_args = list(self.obs_token.keys())
        _reward_step_args = list(self.rewards["step"].keys())
        _reward_done_args = list(self.rewards["episode"].keys())
        _log_args = list(env_setup.INIT_LOGS.keys())
        # loading outer args and overwrite env configs
        
        for key, value in kwargs.items():
            # assert env_setup.check_args_value(key, value)
            if key in _config_args:
                self.configs[key] = value
            elif key in _obs_shape_args:
                self.obs_token[key] = value
            elif key in _reward_step_args:
                self.rewards["step"][key] = value
            elif key in _reward_done_args:
                self.rewards["episode"][key] = value
            elif key in _log_args:
                self.logs[key] = value
            else:
                print(f"Invalid config argument \'{key}:{value}\'")


        # set local defaults if not predefined or loaded
        for key in _config_local_args:
            if key in self.configs:
                continue
            if key == "threshold_damage_2_red":
                self.configs[key] = self.configs["damage_maximum"]
            elif key == "threshold_damage_2_blue":
                # grant blue agents a higher damage threshold when more reds on the map
                self.configs[key] = self.configs["damage_maximum"] * self.num_red
            elif key == "act_masked":
                self.configs[key] = env_setup.ACT_MASKED["mask_on"]
        # setup penalty for invalid action in unmasked conditions
        self.invalid_masked = self.configs["act_masked"]
        if self.invalid_masked is False:
            self.configs["penalty_invalid"] = env_setup.ACT_MASKED["unmasked_invalid_action_penalty"]

        # check init_red init configs, must have attribute "pos":[str/tuple] for resetting agents
        self.configs["init_red"] = env_setup.check_agent_init("red", self.num_red, self.configs["init_red"])
        # check init_blue, must have attribute "route":[str] used in loading graph files. default: '0'
        self.configs["init_blue"] = env_setup.check_agent_init("blue", self.num_blue, self.configs["init_blue"])
        # get all unique routes. (blue agents might share patrol route)
        self.configs["route_lookup"] = list(set(_blue["route"] for _blue in self.configs["init_blue"]))
        
        # setup log inits if not provided
        if "log_on" in self.logs:
            self.logger = self.logs["log_on"]
            # turn on logger if True
            if self.logger is True:
                self.logs["root_path"] = self.configs["env_path"]
                for item in _log_args[1:]:
                    if item not in self.logs:
                        self.logs[item] = env_setup.INIT_LOGS[item]
        return True

    # load scenario map graphs and generate routes [!!] executable after _init_env_config
    def _load_map_data(self):
        # load graphs
        self.map, self.routes = load_graph_files(env_path=self.configs["env_path"],
                                                 map_lookup=self.configs["map_id"],
                                                 route_lookup=self.configs["route_lookup"],
                                                 is_pickle_graph=self.configs["load_pickle"])
        # generate pos & dirs for patrol routes
        for _route in range(len(self.routes)):
            route_len = self.routes[_route].get_route_length()
            self.routes[_route].list_node = [0] * route_len
            for _idx in range(route_len):
                index_by_name = self.map.get_index_by_name(self.routes[_route].list_code[_idx])
                self.routes[_route].list_node[_idx] = index_by_name
            list_dir = [0] * (route_len - 1)
            for _idx in range(route_len - 1):
                list_dir[_idx] = self.map.get_edge_attr_acs_by_idx(self.routes[_route].list_node[_idx],
                                                                   self.routes[_route].list_node[_idx + 1])
            off_dir = self.map.get_edge_attr_acs_by_idx(self.routes[_route].list_node[-1],
                                                        self.routes[_route].list_node[0])
            # verbose direction info for fast retrieval
            self.routes[_route].list_move = [off_dir] + list_dir
            self.routes[_route].list_next = list_dir + [off_dir]

    # initialize team red and team blue agents [!!] executable after _init_env_config
    def _init_agents(self):
        for idx, init_red in enumerate(self.configs["init_red"]):
            r_uid = idx
            learn = init_red["learn"] if "learn" in init_red else True
            if learn is True:
                self.learning_agent.append(r_uid)
            self.team_red.append(AgentRed(_uid=r_uid, _learn=learn))

        for idx, init_blue in enumerate(self.configs["init_blue"]):
            b_uid = (idx + self.num_red)
            learn = init_blue["learn"] if "learn" in init_blue else False
            if learn is True:
                self.learning_agent.append(b_uid)
            b_route = self.configs["route_lookup"].index(init_blue["route"])  # int: index in the route lookup list
            self.team_blue.append(AgentBlue(_uid=b_uid, _learn=learn, _route=b_route))

    # reset agents to init status for each new episode
    def _reset_agents(self):
        HP_red = self.configs["init_health_red"]
        for idx, init_red in enumerate(self.configs["init_red"]):
            r_eval_start_pos = self.n_eval_episodes if self.in_eval else None

            r_node = self.start_node
            # TODO: Fix this
            r_code = env_setup.get_default_red_encoding_simple(r_node)
            #print(f'r_node {r_node}')
            #print(f'r_code {r_code}')
            #r_code = '11_0100' #env_setup.get_default_red_encoding(idx, 25, r_eval_start_pos)
            # why is this so complicated???
            r_dir = env_setup.get_default_dir(init_red["dir"])
            self.team_red[idx].reset(_node=r_node, _code=r_code, _dir=r_dir, _health=HP_red)

            if self.configs["fixed_start"] != -1:
                pos = self.configs["fixed_start"]
                self.team_red[idx].set_location(pos, self.map.get_name_by_index(pos), 1)

        HP_blue = self.configs["init_health_blue"]
        self.blue_offsets = []
        for idx, init_blue in enumerate(self.configs["init_blue"]):
            b_route = self.team_blue[idx].get_route()
            #b_index = init_blue["idx"]  # int: index of the position on the given route
            route_len = self.routes[b_route].get_route_length()
            b_index = int(idx * route_len / self.num_blue) % route_len
            self.blue_offsets.append(b_index)
            b_node, b_code, b_dir = self.routes[b_route].get_location_by_index(b_index)
            self.team_blue[idx].reset(_node=b_node, _code=b_code, _dir=b_dir, _health=HP_blue, _index=b_index, _end=-1)

    def _log_step_prev(self):
        return self.states if self.logger else []

    def _log_step_update(self, prev_obs, actions, rewards):
        # not really sure why this is called when it is
        # using current team positions so this should be called before action to get first action
        # but if called before first action we dont have reward        
        if self.logger is True:
            _r = [[f"red:{_agent.get_id()}", _agent.get_pos_dir(), _agent.get_encoding(), _agent.get_health()]
                  for _agent in self.team_red]
            _b = [[f"blue:{_agent.get_id()}", _agent.get_pos_dir(), _agent.get_encoding(), _agent.get_health()]
                  for _agent in self.team_blue]
            save_log_2_file(self.logs, self.step_counter, self.done_counter, _r + _b,
                            prev_obs, actions, self.states, rewards)

    def _log_episode(self, episode_rewards):
        if self.logger is True:
            log_done_reward(self.logs, self.done_counter, episode_rewards)

    def render(self, mode='human'):
        pass

    # delete local instances
    def close(self):
        if self.team_red is not None:
            del self.team_red
        if self.team_blue is not None:
            del self.team_blue
        if self.routes is not None:
            del self.routes
        if self.map is not None:
            del self.map
