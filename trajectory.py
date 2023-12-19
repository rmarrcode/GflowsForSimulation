import torch

class Trajectory():
    def __init__(self) -> None:
        self.log_norm_constant = 1
        self.forward_probs = torch.tensor([], requires_grad=True)
        self.backward_probs = torch.tensor([], requires_grad=True)
        self.flows = torch.tensor([], requires_grad=True)
        self.actions = torch.tensor([], requires_grad=True)
        self.rewards = torch.tensor([], requires_grad=True)
        self.log_norm_constant = 1
    def add_step(self, forward_prob, backward_prob, flow, action, reward):
        self.forward_probs = torch.cat((self.forward_probs, torch.tensor([forward_prob]))) 
        self.backward_probs = torch.cat((self.backward_probs, torch.tensor([backward_prob]))) 
        self.flows = torch.cat((self.flows, torch.tensor([flow]))) 
        self.actions = torch.cat((self.actions, torch.tensor([action]))) 
        self.rewards = torch.cat((self.rewards, torch.tensor([reward]))) 
    def episode_reward(self, episode_reward):
        self.rewards = torch.cat((self.rewards, torch.tensor([episode_reward]))) 
