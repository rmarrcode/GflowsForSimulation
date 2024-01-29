import numpy as np
import torch

class Losses():
    def log_trajectory_balance(trajectory):
        loss =  trajectory.log_norm_constant + trajectory.forward_probs.sum() - trajectory.backward_probs.sum() 
        return loss
    
    def trajectory_balance(trajectory):
        loss =  (trajectory.log_norm_constant * torch.prod(trajectory.forward_probs, 0)) / (torch.prod(trajectory.backward_probs, 0) * trajectory.rewards.sum())
        return loss


    


