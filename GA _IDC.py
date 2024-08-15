import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
import Energy_calculation
url = 'http://localhost:5399/kspice/measurement'
# Send a GET request to connect to k-spice
response = requests.get(url)
if response.status_code == 200:
    data = response.json()
    try:
        T_s = data['air_supply_temperature']
        T_r = data['air_return_temperature']
    except:
        print(f"Request failed")
else:
    # If the request is unsuccessful, print the status code
    print(f"Request failed with status code: {response.status_code}")

# objective function
def objective_function():
    E = Energy_calculation.func()
    return E


# Constraint Function 1
def constraint_function1(T_s,):

    return T_s - 27

# Constraint Function 2
def constraint_function2(T_r):
    return T_r - 35

# penalty function
def penalty_function(x, k1, k2):
    p1 = max(0, constraint_function1(x))
    p2 = max(0, constraint_function2(x))
    return k1 * p1 + k2 * p2

# Initializing populations
def initialize_population(pop_size, gene_length):
    return np.random.uniform(lower_bound, upper_bound, (pop_size, gene_length))

# fitness function
def fitness_function(x, penalty_coefficient):
    return objective_function() - penalty_function(x, penalty_coefficient)


# Initializing populations
def initialize_population(size, lower_bound, upper_bound):
    return np.random.uniform(lower_bound, upper_bound, size)


# Select operation
def selection(population, fitness):
    indices = np.argsort(fitness)
    return population[indices[:len(population) // 2]]


# crossover operation
def crossover(parents):
    offspring = np.empty_like(parents)
    for i in range(len(parents)):
        parent1 = parents[i]
        parent2 = parents[(i + 1) % len(parents)]
        offspring[i] = (parent1 + parent2) / 2
    return offspring


# Mutation operations
def mutation(offspring, mutation_rate, lower_bound, upper_bound):
    for i in range(len(offspring)):
        if np.random.rand() < mutation_rate:
            offspring[i] = np.random.uniform(lower_bound, upper_bound)
    return offspring


# Adaptive Penalty Coefficient
def adaptive_penalty_coefficient(population, initial_penalty_coefficient, increase_rate, decrease_rate):
    constraint_violations1 = sum([1 for x in population if constraint_function1(T_s) > 0])
    constraint_violations2 = sum([1 for x in population if constraint_function2(T_r) > 0])
    violation_ratio1 = constraint_violations1 / len(population)
    violation_ratio2 = constraint_violations2 / len(population)
    if violation_ratio1 > 0.2:
        return initial_penalty_coefficient1 * decrease_rate
    else:
        return initial_penalty_coefficient1 * increase_rate
    if violation_ratio2 > 0.2:
        return initial_penalty_coefficient2 * decrease_rate
    else:
        return initial_penalty_coefficient2 * increase_rate

# Parameter settings
population_size = 20
generations = 50
lower_bound = 0
upper_bound = 7
initial_penalty_coefficient1 = 10
initial_penalty_coefficient2 = 10
increase_rate = 1.1
decrease_rate = 0.9
mutation_rate = 0.1
elite_size = 1
# initialization
population = initialize_population(population_size, lower_bound, upper_bound)
penalty_coefficient1 = initial_penalty_coefficient1
penalty_coefficient2 = initial_penalty_coefficient2
# Genetic Algorithm Main Loop
for generation in range(generations):
    fitness = np.array([fitness_function(x, penalty_coefficient) for x in population])
    parents = selection(population, fitness)
    offspring = crossover(parents)
    offspring = mutation(offspring, mutation_rate, lower_bound, upper_bound)
    population = np.concatenate((parents, offspring))

    penalty_coefficient = adaptive_penalty_coefficient(population, initial_penalty_coefficient1, increase_rate,
                                                       decrease_rate)

    best_fitness = np.min(fitness)
    best_individual = population[np.argmin(fitness)]

    print(
        f'Generation {generation}: Best Fitness = {best_fitness}, Best Individual = {best_individual}')




