import torch
import torch.nn as nn
import random

# --- НАСТРОЙКИ ---
POPULATION_SIZE = 50
MUTATION_RATE = 0.2  # Чуть поднял рейт для динамики
MUTATION_STRENGTH = 0.5 

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

class NeuroEvolutionAgent:
    def __init__(self, model):
        self.model = model
    
    def get_genes(self):
        genes = []
        with torch.no_grad():
            for param in self.model.parameters():
                genes.append(param.data.view(-1))
            return torch.cat(genes)

    # FIX: Убрали аргумент model, используем self.model
    def set_genes(self, genes_vector):
        current_index = 0
        with torch.no_grad():
            for param in self.model.parameters():
                num_weights = param.numel()
                layer_weights = genes_vector[current_index : current_index + num_weights]
                param.data = layer_weights.view(param.shape)
                current_index += num_weights

def crossover(parent1, parent2):
    # Среднее арифметическое родителей
    child = (parent1 + parent2) / 2.0
    return child

def mutate(individual, mutation_rate=0.1, mutation_strength=0.1):
    clone = individual.clone()
    # FIX: Добавляем шум
    if random.random() < mutation_rate:
        noise = torch.randn_like(individual) * mutation_strength
        clone += noise
    return clone

# --- ИНИЦИАЛИЗАЦИЯ ---
simpleNet = SimpleNet()
criterion = torch.nn.MSELoss()
ag = NeuroEvolutionAgent(model=simpleNet)


dummy_vector = ag.get_genes()
print(f"Genome length: {len(dummy_vector)}")


population = [torch.randn(dummy_vector.shape) for _ in range(POPULATION_SIZE)]


inputs = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
targets = torch.tensor([[0.],[1.],[1.],[0.]])

def get_fitness(genes_vector):
    
    ag.set_genes(genes_vector) 
    
    with torch.no_grad():
        output = simpleNet(inputs)
        loss = criterion(output, targets)
        
    return -loss.item()

# --- ЦИКЛ ЭВОЛЮЦИИ ---
for i in range(100):
    scored_pop = []
    
    
    for individual in population:
        fitness = get_fitness(individual)
        scored_pop.append((fitness, individual))

    
    
    scored_pop.sort(key=lambda x: x[0], reverse=True)
    
    best_fitness, best_genes = scored_pop[0]
    
    
    if i % 10 == 0:
        print(f'Gen {i} | Best Fitness: {best_fitness:.4f}')
        if best_fitness > -0.01:
            print("Solved!")
            break

    
    new_pop = []
    
    
    for k in range(4):
        new_pop.append(scored_pop[k][1])

    
    top_25 = scored_pop[:25] 
    
    while len(new_pop) < POPULATION_SIZE:
        parent1 = random.choice(top_25)[1]
        parent2 = random.choice(top_25)[1]
        
        child = crossover(parent1, parent2)
        child = mutate(child, mutation_rate=MUTATION_RATE, mutation_strength=MUTATION_STRENGTH)
        
        new_pop.append(child)
        
    population = new_pop
print("\n--- FINAL TEST ---")
best_genes = scored_pop[0][1]
ag.set_genes(best_genes)
with torch.no_grad():
    print(simpleNet(inputs))
    print("Targets:\n", targets)