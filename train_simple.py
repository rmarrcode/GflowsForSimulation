import torch
import numpy as np
import random
import wandb
import torch.optim as optim
import torch.nn as nn
from  model.samplers import SamplerFCN, SimpleNetwork

WANDB = True
SEED = 0
LEARNING_RATE = .01
NUM_EPOCHS = 100000
BATCH_SIZE = 150

if WANDB:
    wandb.init(
        project="graph-training-simulation",        
        config={
            "learning_rate": LEARNING_RATE,
        }
    )


def working():

    sampler_fcn = SamplerFCN(
                self_size=27,
                num_hiddens=512,
                num_outputs=1
            )
    sampler_fcn.train()


    criterion = nn.MSELoss()
    optimizer = optim.Adam(sampler_fcn.parameters(), lr=0.01)

    for epoch in range(1000): 
        output = sampler_fcn([torch.ones(27)])
        target = torch.rand_like(output) + 1.0
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    sampler_fcn.eval()

# the loss function is the difference
def notworking():
    criterion = nn.MSELoss()
    sampler_fcn = SamplerFCN(
                self_size=27,
                num_hiddens=512,
                num_outputs=1
            )

    sampler_fcn.train()

    optimizer = optim.AdamW(sampler_fcn.parameters(), lr=LEARNING_RATE)
    batch_loss = 0
    batch_num = 0
    batch_reward = 0

    for _ in range(NUM_EPOCHS):

        action = sampler_fcn.forward([torch.ones(27)])
        reward = torch.tensor(10.0, requires_grad=True)
        if action > 0:
            reward = torch.tensor(99.0, requires_grad=True)

        batch_reward += reward

        batch_num = batch_num + 1

        target = torch.rand_like(action) + 1.0
        episode_loss = criterion(action, target)

        batch_loss += episode_loss

        if batch_num % BATCH_SIZE == 0:
            if WANDB:
                wandb.log({"loss": batch_loss/BATCH_SIZE, "reward": batch_reward/BATCH_SIZE})
                for name, param in sampler_fcn.named_parameters():
                    wandb.log({f"{name}_mean": param.data.mean().item(), f"{name}_std": param.data.std().item()})
                batch_loss = 0
                batch_reward = 0

        optimizer.zero_grad()
        episode_loss.backward()
        optimizer.step()

    



if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    notworking()
    