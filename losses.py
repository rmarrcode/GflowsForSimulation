class Losses():
    def trajectory_balence(self, trajectory):
        loss =  trajectory.log_norm_constant + trajectory.forward_probs.sum() - trajectory.backward_probs.sum() - trajectory.rewards()
        return loss

