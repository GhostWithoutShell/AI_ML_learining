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
    samples = random.sample(population, k = tournament_size)
    result = []
    for i in range(len(samples)-1):
        first_fit, second_fit = calculate_fitness(samples[i]), calculate_fitness(samples[i+1])
        result.append(math.max(first_fit, second_fit))

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
    for _ in range(len(TARGET_PHRASE)):
        if random() < mutation_rate:
            index_new_sy = random.randomrange(0, len(GENRES))
            new_sy = GENRES[index_new_sy]
            new_index_sy = random.randomrange(0, len(TARGET_PHRASE))
            individual[new_index_sy] = new_sy
    return "".join(individual_list)

def evolve(population, target_len, mutation_rate=0.01):
    """
    Создает новое поколение
    """
    new_population = []
    
    # Элитизм: можно (но не обязательно) сразу перенести самого лучшего из старого поколения в новое без изменений
    # Это гарантирует, что fitness никогда не упадет.
    
    # Пока размер новой популяции не станет равен размеру старой:
    # 1. parent1 = selection_tournament(...)
    # 2. parent2 = selection_tournament(...)
    # 3. child = crossover(parent1, parent2)
    # 4. child = mutate(child, mutation_rate)
    # 5. добавить child в new_population
    
    return new_population
print(create_individual(len(TARGET_PHRASE)))
print(create_individual(len(TARGET_PHRASE)))
print(create_individual(len(TARGET_PHRASE)))


