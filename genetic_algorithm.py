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

def mutate_swap(route, mutation_rate):
    """Perform swap mutation by swapping two cities in the route."""
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(route)), 2)
        route[idx1], route[idx2] = route[idx2], route[idx1]

def mutate_adjacent_swap(route, mutation_rate):
    """
    Perform mutation by swapping two adjacent cities in the route.
    Args:
        route: The current route (list of cities).
        mutation_rate: Probability of applying the mutation.
    """
    if random.random() < mutation_rate:
        # Randomly select an index for adjacent swapping
        idx = random.randint(0, len(route) - 2)
        # Swap the selected city with its adjacent neighbor
        route[idx], route[idx + 1] = route[idx + 1], route[idx]

def mutate_inverse(route, mutation_rate):
    """Perform inversion mutation by reversing a segment of the route."""
    if random.random() < mutation_rate:
        idx1, idx2 = sorted(random.sample(range(len(route)), 2))
        route[idx1:idx2] = reversed(route[idx1:idx2])

def mutate_insertion(route, mutation_rate):
    """
    Perform mutation by moving a city to a new position in the route.
    Args:
        route: The current route (list of cities).
        mutation_rate: Probability of applying the mutation.
    """
    if random.random() < mutation_rate:
        # Select a random city and a new position
        city_idx = random.randint(0, len(route) - 1)
        new_pos = random.randint(0, len(route) - 1)
        
        # Remove the city from its original position
        city = route.pop(city_idx)
        
        # Insert the city at the new position
        route.insert(new_pos, city)

def tournament_selection(population, fitness, tournament_size=6):
    """Select parents using tournament selection."""
    selected_parents = []
    for _ in range(len(population) // 2):
        # Pick random individuals for the tournament
        tournament = random.sample(list(zip(population, fitness)), tournament_size)
        tournament.sort(key=lambda x: x[1])  # Sort by fitness (ascending)
        selected_parents.append(tournament[0][0])  # Select the fittest
    return selected_parents

def elitism_selection(population, fitness, elite_size=2):
    sorted_indices = sorted(range(len(fitness)), key=lambda i: fitness[i])
    return [population[i] for i in sorted_indices[:elite_size]]

def steady_state_selection(population, fitness, num_survivors=None):
    """
    Perform steady-state selection by keeping the best individuals.
    Args:
        population: List of individuals (routes).
        fitness: List of fitness values (lower is better for TSP).
        num_survivors: Number of individuals to keep in the next generation.
                       If None, defaults to keeping half the population.
    Returns:
        Selected survivors (list of individuals).
    """
    # Set default number of survivors to half the population size
    if num_survivors is None:
        num_survivors = len(population) // 2

    # Sort population by fitness (ascending order)
    sorted_population = [x for _, x in sorted(zip(fitness, population))]

    # Keep the best `num_survivors` individuals
    return sorted_population[:num_survivors]


def one_point_crossover(parent1, parent2):
    """Perform one-point crossover between two parents."""
    crossover_point = random.randint(1, len(parent1) - 1)
    offspring = parent1[:crossover_point] + [gene for gene in parent2 if gene not in parent1[:crossover_point]]
    return offspring

def cycle_crossover(parent1, parent2):
    """Perform cycle crossover between two parents."""
    size = len(parent1)
    offspring = [None] * size
    index = 0

    while None in offspring:
        start = parent1[index]
        while True:
            offspring[index] = parent1[index]
            index = parent1.index(parent2[index])
            if parent1[index] == start:
                break

        index = offspring.index(None) if None in offspring else -1
    return offspring

def order_crossover(parent1, parent2):
    """Perform order crossover."""
    size = len(parent1)
    offspring = [None] * size

    start, end = sorted(random.sample(range(size), 2))
    offspring[start:end] = parent1[start:end]

    current_idx = end
    for gene in parent2:
        if gene not in offspring:
            if current_idx >= size:
                current_idx = 0
            offspring[current_idx] = gene
            current_idx += 1

    return offspring

def run_genetic_algorithm(distance_matrix, population_size, generations, mutation_rate, crossover_rate, 
                          selection_method="tournament", crossover_method="one_point", mutation_method="swap"):
    """
    Run the genetic algorithm for a specified number of generations to solve TSP.
    Supports dynamic selection of selection, crossover, and mutation methods.
    """

    # Mapping selection crossover and mutation
    selection_functions = {
        "tournament": tournament_selection,
        "elitism": elitism_selection,
        "steady_state": steady_state_selection
    }
    crossover_functions = {
        "one_point": one_point_crossover,
        "cycle": cycle_crossover,
        "order": order_crossover
    }
    mutation_functions = {
        "swap": mutate_swap,
        "adjacent_swap": mutate_adjacent_swap,
        "inverse": mutate_inverse,
        "insertion": mutate_insertion,
    }
    
    select_func = selection_functions[selection_method]
    crossover_func = crossover_functions[crossover_method]
    mutate_func = mutation_functions[mutation_method]

    # Initialize population
    population = initialize_population(population_size, len(distance_matrix))
    
    best_results = []
    for generation in range(generations):
        fitness = calculate_fitness(population, distance_matrix)
        
        if not fitness:
            print("Error: Fitness list is empty!")
            break

        # Use tournament selection to select parents
        parents = select_func(population, fitness)

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
                    offspring = crossover_func(parent1, parent2)
                else:
                    # If no crossover, offspring is just a copy of one parent
                    offspring = parent1.copy()

                # Apply mutation
                mutate_func(offspring, mutation_rate)
                next_generation.append(offspring)

        population = next_generation

        # Optional: Display progress if desired
        best_distance = min(fitness)
        print(f"Generation {generation + 1}, Best Distance: {best_distance}")
        best_results.append(best_distance)

    # Return the best solution from the final population
    fitness = calculate_fitness(population, distance_matrix)
    best_index = fitness.index(min(fitness))
    return population[best_index], min(fitness), best_results
