import random
import math

# --- НАСТРОЙКИ ---
POPULATION_SIZE = 200
MUTATION_RATE = 0.01      
MUTATION_STRENGTH = 0.05  
GENERATIONS = 100


X_MIN = 0
X_MAX = 30

def fitness_function(x):
    
    return x * math.sin(x)

def create_individual():
    
    return random.uniform(X_MIN, X_MAX)

def create_population(size):
    return [create_individual() for _ in range(size)]

def crossover(parent1, parent2):
    
    return (parent1 + parent2) / 2.0

def mutate(individual):
    
    if random.random() < MUTATION_RATE:

        noise = random.uniform(-MUTATION_STRENGTH, MUTATION_STRENGTH)
        individual += noise
        individual = max(X_MIN, min(individual, X_MAX))
        
    return individual

def selection_tournament(population, k=3):
    samples = random.sample(population, k)
    return max(samples, key=fitness_function)

# --- ЗАПУСК ---

population = create_population(POPULATION_SIZE)

for gen in range(GENERATIONS):
    best_ind = max(population, key=fitness_function)
    best_fit = fitness_function(best_ind)
    
    if gen % 5 == 0:
        print(f"Gen {gen}: Best X = {best_ind:.4f} | Fitness = {best_fit:.4f}")
        
    new_population = [best_ind]
    
    while len(new_population) < POPULATION_SIZE:
        p1 = selection_tournament(population)
        p2 = selection_tournament(population)
        
        child = crossover(p1, p2)
        child = mutate(child)
        new_population.append(child)
        
    population = new_population

print(f"\nFINAL RESULT: X = {best_ind:.5f}")
print(f"Math check: {best_ind} * sin({best_ind}) = {best_ind * math.sin(best_ind)}")    