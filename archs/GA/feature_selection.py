import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import random
import copy

# 1. Подготовка данных
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Параметры GA
POPULATION_SIZE = 20    # Сколько индивидов в поколении
GENERATIONS = 15        # Сколько эпох эволюции
MUTATION_RATE = 0.1     # Шанс мутации гена
PENALTY_FACTOR = 0.002  # Штраф за каждый лишний признак (чтобы искать минимальный набор)


def create_individual(n_features):
    return [random.randint(0, 1) for _ in range(n_features)]


def calculate_fitness(mask):
    
    selected_indices = [i for i, bit in enumerate(mask) if bit == 1]
    
    
    if len(selected_indices) == 0:
        return 0.0
    
    
    X_train_sel = X_train[:, selected_indices]
    X_test_sel = X_test[:, selected_indices]
    
    
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train_sel, y_train)
    
    
    predictions = clf.predict(X_test_sel)
    accuracy = accuracy_score(y_test, predictions)
    
    
    fitness = accuracy - (PENALTY_FACTOR * len(selected_indices))
    
    return fitness

# 4. Скрещивание (Crossover) - Одноточечное
def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

# 5. Мутация
def mutate(individual):
    for i in range(len(individual)):
        if random.random() < MUTATION_RATE:
            individual[i] = 1 - individual[i] # Инверсия бита (0->1, 1->0)
    return individual




population = [create_individual(X.shape[1]) for _ in range(POPULATION_SIZE)]

for gen in range(GENERATIONS):

    ranked_population = sorted(population, key=lambda ind: calculate_fitness(ind), reverse=True)
    
    best_ind = ranked_population[0]
    best_fitness = calculate_fitness(best_ind)
    num_features = sum(best_ind)
    
    print(f"Gen {gen+1}: Best Fitness = {best_fitness:.4f} | Features: {num_features}/{X.shape[1]}")
    
    
    next_generation = ranked_population[:4] 
    

    while len(next_generation) < POPULATION_SIZE:

        parent1 = random.choice(ranked_population[:10])
        parent2 = random.choice(ranked_population[:10])
        
        c1, c2 = crossover(parent1, parent2)
        next_generation.append(mutate(c1))
        if len(next_generation) < POPULATION_SIZE:
            next_generation.append(mutate(c2))
            
    population = next_generation

best_mask = sorted(population, key=calculate_fitness, reverse=True)[0]
selected_features = [i for i, bit in enumerate(best_mask) if bit == 1]
print("\nBest features indices:", selected_features)
print(f"Original Accuracy (all features): {calculate_fitness([1]*30):.4f}")
print(f"GA Selected Accuracy ({len(selected_features)} features): {calculate_fitness(best_mask):.4f}")