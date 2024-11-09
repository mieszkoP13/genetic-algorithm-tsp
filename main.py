from genetic_algorithm import run_genetic_algorithm, generate_random_coordinates, generate_distance_matrix
from visualization import plot_route

# Example number of cities
num_cities = 6

# Generate random city coordinates
coordinates = generate_random_coordinates(num_cities)

# Generate the distance matrix from coordinates
distance_matrix = generate_distance_matrix(coordinates)

# Parameters for the genetic algorithm
population_size = 50
generations = 100
mutation_rate = 0.1  # Example mutation rate
crossover_rate = 0.8  # Example crossover rate (80% chance of crossover)

# Run the genetic algorithm
best_route, best_distance = run_genetic_algorithm(distance_matrix, population_size, generations, mutation_rate, crossover_rate)

# Display results
print("Best Route Found:", best_route)
print("Best Distance:", best_distance)

# Visualize the best route
plot_route(best_route, coordinates)
