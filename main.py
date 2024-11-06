from genetic_algorithm import run_genetic_algorithm

# Example distance matrix for 4 cities
distance_matrix = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]

# Parameters for the genetic algorithm
population_size = 10
generations = 50
mutation_rate = 0.1

# Run the genetic algorithm
best_route, best_distance = run_genetic_algorithm(distance_matrix, population_size, generations, mutation_rate)

print("Best Route Found:", best_route)
print("Best Distance:", best_distance)
