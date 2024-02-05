import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class Trajectory():
    def __init__(self) -> None:
        self.log_norm_constant = 5
        self.forward_probs = torch.tensor([1.], requires_grad=True, device=device)
        self.backward_probs = torch.tensor([1.], requires_grad=True, device=device)
        #self.flows = torch.tensor([1.], requires_grad=True)
        #self.actions = torch.tensor([], requires_grad=True)
        self.rewards = torch.tensor([1.], requires_grad=True, device=device)
        #self.nodes = torch.tensor([], requires_grad=True)
    # def add_step(self, forward_prob, backward_prob, flow, action, reward, node):
    #     self.forward_probs = torch.cat((self.forward_probs, torch.tensor([forward_prob]))) 
    #     self.backward_probs = torch.cat((self.backward_probs, torch.tensor([backward_prob]))) 
    #     self.flows = torch.cat((self.flows, torch.tensor([flow]))) 
    #     self.nodes = torch.cat((self.nodes, torch.tensor([node]))) 
    #     self.actions = torch.cat((self.actions, torch.tensor([action]))) 
    #     self.rewards = torch.cat((self.rewards, torch.tensor([reward]))) 
    def add_step(self, forward_prob, backward_prob, reward):
        self.forward_probs = self.forward_probs * forward_prob
        self.backward_probs = self.backward_probs * backward_prob
        self.rewards = self.rewards + reward

    def episode_reward(self, episode_reward):
        self.rewards = torch.cat((self.rewards, torch.tensor([episode_reward]))) 
