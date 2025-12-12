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
    max_steps = 500
    steps = 0
    while not (terminated or truncated) and steps < max_steps:
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
        steps += 1
    return total_reward

def build_pop(pop_size = 100):
    population = []
    for i in range(pop_size):
        population.append(DefModel())
    return population
def mutate(individual):
    individual_ = copy.deepcopy(individual)
    with torch.no_grad():
        for param in individual_.parameters():
            
            if random.random() < MUTATION_RATE:
                rand_noize = torch.rand_like(param) * MUATATION_STREIGHT
                param.add_(rand_noize)
    return individual_
def selection(population, k = 3):
    new_pop = []
    
    population.sort(key= lambda x : x.total_reward)
    top_3 = population[-3:]
    new_pop.extend([copy.deepcopy(ind) for ind in top_3])
    top_half = population[len(population)//2:]
    while len(new_pop) < len(population):
        #samples = random.sample(population, k)
        #samples = samples.sort(lambda x : x.total_reward)
        
        parent1 = random.choice(top_half)
        #child = crossover(parent1, parent2)
        child = mutate(parent1)
        new_pop.append(child)
    return new_pop
        

    

env = gym.make("CartPole-v1", render_mode="rgb_array", max_episode_steps=10000)
env.reset()

population = build_pop()
for i in range(50):
    for j in range(len(population)):
        reward = run_episode(env ,population[j])
        population[j].total_reward = reward
    new_pop = selection(population=population)
    best_agent = max(new_pop,key = lambda x : x.total_reward)
    print("Max reward :", best_agent.total_reward, "Gen :", i+1)
    if best_agent.total_reward >= 499:
        print("Solved!")
        torch.save(best_agent.state_dict(), "cartpole_model.pth")
        break
    population = new_pop

env.close()



best_model = max(population, key=lambda x: x.total_reward)

print("Запуск демонстрации лучшей модели...")
# Создаем среду с режимом human для отрисовки окна
show_env = gym.make("CartPole-v1", render_mode="human", max_episode_steps=2000)

state, _ = show_env.reset()
done = False

while not done:
    # Обязательно рендерим
    show_env.render()
    
    # Логика выбора действия (как в run_episode)
    tensor = torch.tensor(state, dtype=torch.float32)
    with torch.no_grad():
        output = best_model(tensor)
    action = 0 if output.item() < 0 else 1
    
    state, _, terminated, truncated, _ = show_env.step(action)
    done = terminated or truncated

show_env.close()