from genetic_algorithm import initialize_population
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

# Calculate fitness (route distance) for each individual in the population
for route in population:
    print("Route:", route, "Distance:", calculate_route_distance(route, distance_matrix))
