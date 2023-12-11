class Trajectory():
    def __init__(self) -> None:
        self.init_flow = 1
        self.forward_probs = []
        self.backward_probs = []
        self.flows = []
        self.actions = []
        self.rewards = []
    def add_step(self, forward_prob, bacward_prob, flow, action, reward):
        self.forward_probs.append(forward_prob)
        self.backward_probs.append(bacward_prob)
        self.flows.append(flow)
        self.actions.append(action)
        self.rewards(reward)
