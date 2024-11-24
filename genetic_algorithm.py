import random
import math

def calculate_distance(city1, city2):
    """Calculate the Euclidean distance between two cities."""
    return math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

def generate_distance_matrix(coordinates):
    """Generate the distance matrix from the coordinates of the cities."""
    num_cities = len(coordinates)
    distance_matrix = [[0] * num_cities for _ in range(num_cities)]
    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            distance = calculate_distance(coordinates[i], coordinates[j])
            distance_matrix[i][j] = distance_matrix[j][i] = distance
    return distance_matrix

def generate_random_coordinates(num_cities):
    return [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(num_cities)]

def initialize_population(population_size, num_cities):
    """Initialize the population with random routes."""
    population = []
    for _ in range(population_size):
        route = list(range(num_cities))
        random.shuffle(route)  # Shuffle to create a diverse set of routes
        population.append(route)
    return population

def calculate_fitness(population, distance_matrix):
    """Calculate the fitness (distance) of each individual in the population."""
    fitness = []
    for route in population:
        total_distance = 0
        for i in range(len(route) - 1):
            total_distance += distance_matrix[route[i]][route[i + 1]]
        total_distance += distance_matrix[route[-1]][route[0]]  # To close the loop
        fitness.append(total_distance)
    return fitness

def mutate(route, mutation_rate):
    """Perform mutation by swapping two cities in the route."""
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(route)), 2)
        route[idx1], route[idx2] = route[idx2], route[idx1]

def tournament_selection(population, fitness, tournament_size=3):
    """Select parents using tournament selection."""
    selected_parents = []
    for _ in range(len(population) // 2):
        # Pick random individuals for the tournament
        tournament = random.sample(list(zip(population, fitness)), tournament_size)
        tournament.sort(key=lambda x: x[1])  # Sort by fitness (ascending)
        selected_parents.append(tournament[0][0])  # Select the fittest
    return selected_parents

def one_point_crossover(parent1, parent2):
    """Perform one-point crossover between two parents."""
    crossover_point = random.randint(1, len(parent1) - 1)
    offspring = parent1[:crossover_point] + [gene for gene in parent2 if gene not in parent1[:crossover_point]]
    return offspring

def run_genetic_algorithm(distance_matrix, population_size, generations, mutation_rate, crossover_rate):
    """Run the genetic algorithm for a specified number of generations to solve TSP."""
    # Initialize population
    population = initialize_population(population_size, len(distance_matrix))
    
    # Print initial population and its fitness
    #print("Initial Population:")
    for idx, route in enumerate(population):
        fitness = calculate_fitness([route], distance_matrix)[0]
        #print(f"Route {idx + 1}: {route} | Distance: {fitness}")

    for generation in range(generations):
        fitness = calculate_fitness(population, distance_matrix)
        
        if not fitness:
            print("Error: Fitness list is empty!")
            break

        # Use tournament selection to select parents
        parents = tournament_selection(population, fitness)

        # Generate new population with crossover and mutation
        next_generation = []
        while len(next_generation) < population_size:
            for i in range(0, len(parents), 2):
                if len(next_generation) >= population_size:
                    break
                parent1 = parents[i]
                parent2 = parents[i + 1] if i + 1 < len(parents) else parents[0]
                
                # Perform crossover based on crossover_rate
                if random.random() < crossover_rate:
                    offspring = one_point_crossover(parent1, parent2)
                else:
                    # If no crossover, offspring is just a copy of one parent
                    offspring = parent1.copy()

                # Apply mutation
                mutate(offspring, mutation_rate)
                next_generation.append(offspring)

        population = next_generation

        # Optional: Display progress if desired
        best_distance = min(fitness)
        print(f"Generation {generation + 1}, Best Distance: {best_distance}")

    # Return the best solution from the final population
    fitness = calculate_fitness(population, distance_matrix)
    best_index = fitness.index(min(fitness))
    return population[best_index], min(fitness)
