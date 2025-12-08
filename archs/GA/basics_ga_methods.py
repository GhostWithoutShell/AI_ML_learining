import string
import random
import math
TARGET_PHRASE = "AI is Amazing"

GENRES = string.ascii_letters + " "
#random.seed(23234)
def create_individual(target_len):
    
    new_string = ""
    for _ in range(target_len):
        max_index = len(GENRES)
        symbol = random.randrange(0, max_index)
        new_string +=(GENRES[symbol])
    return new_string
def create_population(pop_size, target_len):
    population = []
    for _ in range(pop_size):
        population.append(create_individual(target_len=target_len))

    return population
def calculate_fitness(individual):
    fitness = 0
    for i in range(len(individual)):
        if individual[i] == TARGET_PHRASE[i]:
            fitness += 1
    return fitness
    

def selection_tournament(population, tournament_size=3):
    """
    Выбирает случайных `tournament_size` индивидов из популяции
    и возвращает того, у кого самый высокий fitness.
    """
    tournament = random.sample(population, tournament_size)
    winner = max(tournament, key=calculate_fitness)
    return winner
def crossover(parent1, parent2):
    """
    Создает ребенка, беря половину генов от parent1 и половину от parent2.
    """
    child = parent1[:len(parent1)//2]
    child += parent2[len(parent2)//2 :]
    return child

def mutate(individual, mutation_rate=0.01):
    """
    Проходит по каждому символу индивида и с вероятностью mutation_rate
    заменяет его на случайный новый символ из GENES.
    """
    individual_list = list(individual) # Строки в Python неизменяемы, лучше работать со списком
    for i in range(len(TARGET_PHRASE)):
        if random.random() < mutation_rate:
            index_new_sy = random.randrange(0, len(GENRES))
            new_sy = GENRES[index_new_sy]
            individual_list[i] = new_sy
    return "".join(individual_list)

def evolve(population, target_len, mutation_rate=0.01):
    """
    Создает новое поколение
    """
    new_population = []
    
    
    for _ in range(len(population)):
        parent1 = selection_tournament(population=population)
        parent2 = selection_tournament(population=population)
        child = crossover(parent1=parent1, parent2=parent2)
        child = mutate(child, mutation_rate=mutation_rate)
        new_population.append(child)
    
    
    return new_population
POPULATION_SIZE = 200
population = create_population(POPULATION_SIZE, len(TARGET_PHRASE))
generation = 1

while True:
    
    best_individual = max(population, key=calculate_fitness)
    fitness = calculate_fitness(best_individual)
    
    print(f"Gen {generation} | Best: '{best_individual}' | Fitness: {fitness}/{len(TARGET_PHRASE)}")
    
    
    if best_individual == TARGET_PHRASE:
        break
        
    
    population = evolve(population, POPULATION_SIZE, mutation_rate=0.1)
    generation += 1

print("\nУра! Фраза восстановлена!")


