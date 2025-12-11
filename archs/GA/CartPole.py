import gymnasium as gym
import pygame
from gymnasium.utils.play import play
import torch.nn as nn
import torch
import random
import copy

MUTATION_RATE = 0.01
MUATATION_STREIGHT = 0.001


class DefModel(nn.Module):
    def __init__(self):
        super().__init__()
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
        tensor = torch.tensor(state, dtype=torch.float32)
        output = model(tensor)
        output = output.item()
        mapping_action = 0
        if output < 0:
            mapping_action = 0
        else:
            mapping_action = 1
        state, reward, terminated, truncated, _ = env.step(mapping_action)
        total_reward += reward
    return total_reward

def build_pop(pop_size = 100):
    population = []
    for i in range(pop_size):
        population.append(DefModel())
    return population
def mutate(individual):
    individual_ = copy.deepcopy(individual)
    with torch.no_grad():
        
    if random.rand() < MUTATION_RATE:
        rand_noize = torch.rand_like(individual_.parameters.data) * MUATATION_STREIGHT
        individual_.model.parameters.data += rand_noize
    return individual_
def selection(population, k = 3):
    new_pop = []
    max_rew = population.max(lambda x : x.total_reward)
    population = population.sort(lambda x : x.total_reward)
    new_pop.append(max_rew)
    for i in len(population) - 1:
        #samples = random.sample(population, k)
        #samples = samples.sort(lambda x : x.total_reward)
        parent1, parent2 = population.choise(k)[1], population.choise(k)[1]
        child = crossover(parent1, parent2)
        child = mutate(child)
        new_pop.append(child)
    return new_pop
        

    

env = gym.make("CartPole-v1", render_mode="rgb_array")
env.reset()




