import numpy as np
import torch

class Losses():
    def log_trajectory_balance(trajectory):
        loss =  trajectory.log_norm_constant + trajectory.forward_probs.sum() - trajectory.backward_probs.sum() 
        return loss
    
    def trajectory_balance(trajectory):
        # loss =  (trajectory.log_norm_constant * torch.prod(trajectory.forward_probs, 0)) / (torch.prod(trajectory.backward_probs, 0) * trajectory.rewards.sum())
        # return loss
        # log_norm_constant = trajectory.log_norm_constant
        # forward_probs = torch.prod(trajectory.forward_probs, 0)
        # backward_probs = torch.prod(trajectory.backward_probs, 0)
        # reward = trajectory.rewards.sum()
        # loss = ( (log_norm_constant *  forward_probs) /  (backward_probs * reward) )
        loss =  (trajectory.log_norm_constant * trajectory.forward_probs) / (trajectory.backward_probs * trajectory.rewards)
        return loss

    def step_test(step):
        loss = step['forward_prob']
        return loss

    


