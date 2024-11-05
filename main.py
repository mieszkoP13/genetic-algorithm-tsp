from genetic_algorithm import initialize_population, one_point_crossover, mutate
from fitness import calculate_route_distance

# Example distance matrix for 4 cities
distance_matrix = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]

# Initialize population
population = initialize_population(population_size=5, num_cities=len(distance_matrix))

# Select two parents for testing crossover
parent1 = population[0]
parent2 = population[1]
print("Parent 1:", parent1)
print("Parent 2:", parent2)

# Perform one-point crossover
offspring = one_point_crossover(parent1, parent2)
print("Offspring:", offspring)

# Apply mutation to the offspring
mutate(offspring, mutation_rate=0.5)
print("Offspring after mutation:", offspring)

# Calculate distance for each individual in the population
for route in population:
    print("Route:", route, "Distance:", calculate_route_distance(route, distance_matrix))
