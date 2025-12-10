import gymnasium as gym
import pygame
from gymnasium.utils.play import play
import torch.nn as nn
import torch

class DefModel(nn.Module):
    def __init__(self):
        self.lin = nn.Linear(4, 1)
    def forward(self, x):
        return self.lin(x)

mapping = {
    (pygame.K_LEFT,): 0,
    (pygame.K_RIGHT,): 1
}
def run_episode(env, model):
    state, info = env.reset()
    total_reward = 0
    terminated = False
    truncated = False
    while not (terminated or truncated):
        tensor = torch.tensor(state)
        
def fitness_(input):
    nul = torch.empty((4,1))
    return input-nul
    

env = gym.make("CartPole-v1", render_mode="rgb_array")
env.reset()
model = DefModel()
while True:
    state = env.state()
    output = model(state)
    mapping_t = mapping[output]
    env.step(mapping_t)



