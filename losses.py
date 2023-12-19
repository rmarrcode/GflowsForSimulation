import numpy as np
import torch

class Losses():
    def trajectory_balance(trajectory):
        # flows???
        loss =  trajectory.log_norm_constant + trajectory.forward_probs.sum() - trajectory.backward_probs.sum() - trajectory.rewards.sum()
        return loss

