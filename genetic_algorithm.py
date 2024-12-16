import random
import math
from typing import List, Tuple, Optional

def calculate_distance(city1: Tuple[float, float], city2: Tuple[float, float]) -> float:
    """Calculate the Euclidean distance between two cities."""
    return math.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)

def generate_distance_matrix(coordinates: List[Tuple[float, float]]) -> List[List[float]]:
    """Generate the distance matrix from the coordinates of the cities."""
    num_cities = len(coordinates)
    distance_matrix = [[0.0] * num_cities for _ in range(num_cities)]
    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            distance = calculate_distance(coordinates[i], coordinates[j])
            distance_matrix[i][j] = distance_matrix[j][i] = distance
    return distance_matrix

def generate_random_coordinates(num_cities: int) -> List[Tuple[float, float]]:
    """Generate random coordinates for a given number of cities."""
    return [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(num_cities)]

def initialize_population(population_size: int, num_cities: int) -> List[List[int]]:
    """Initialize the population with random routes."""
    population = []
    for _ in range(population_size):
        route = list(range(num_cities))
        random.shuffle(route)
        population.append(route)
    return population

def calculate_fitness(population: List[List[int]], distance_matrix: List[List[float]]) -> List[float]:
    """Calculate the fitness (distance) of each individual in the population."""
    fitness = []
    for route in population:
        total_distance = 0.0
        for i in range(len(route) - 1):
            total_distance += distance_matrix[route[i]][route[i + 1]]
        total_distance += distance_matrix[route[-1]][route[0]]  # To close the loop
        fitness.append(total_distance)
    return fitness

def mutate_swap(route: List[int], mutation_rate: float) -> None:
    """Perform swap mutation by swapping two cities in the route."""
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(route)), 2)
        route[idx1], route[idx2] = route[idx2], route[idx1]

def mutate_adjacent_swap(route: List[int], mutation_rate: float) -> None:
    """Perform mutation by swapping two adjacent cities in the route."""
    if random.random() < mutation_rate:
        idx = random.randint(0, len(route) - 2)
        route[idx], route[idx + 1] = route[idx + 1], route[idx]

def mutate_inverse(route: List[int], mutation_rate: float) -> None:
    """Perform inversion mutation by reversing a segment of the route."""
    if random.random() < mutation_rate:
        idx1, idx2 = sorted(random.sample(range(len(route)), 2))
        route[idx1:idx2] = reversed(route[idx1:idx2])

def mutate_insertion(route: List[int], mutation_rate: float) -> None:
    """Perform mutation by moving a city to a new position in the route."""
    if random.random() < mutation_rate:
        city_idx = random.randint(0, len(route) - 1)
        new_pos = random.randint(0, len(route) - 1)
        city = route.pop(city_idx)
        route.insert(new_pos, city)

def tournament_selection(population: List[List[int]], fitness: List[float], tournament_size: int = 6) -> List[List[int]]:
    """Select parents using tournament selection."""
    selected_parents = []
    for _ in range(len(population) // 2):
        tournament = random.sample(list(zip(population, fitness)), tournament_size)
        tournament.sort(key=lambda x: x[1])
        selected_parents.append(tournament[0][0])
    return selected_parents

def elitism_selection(population: List[List[int]], fitness: List[float], elite_size: int = 2) -> List[List[int]]:
    """Select the best individuals using elitism."""
    sorted_indices = sorted(range(len(fitness)), key=lambda i: fitness[i])
    return [population[i] for i in sorted_indices[:elite_size]]

def steady_state_selection(
    population: List[List[int]],
    fitness: List[float],
    num_survivors: Optional[int] = None
) -> List[List[int]]:
    """Perform steady-state selection by keeping the best individuals."""
    if num_survivors is None:
        num_survivors = len(population) // 2
    sorted_population = [x for _, x in sorted(zip(fitness, population))]
    return sorted_population[:num_survivors]

def one_point_crossover(parent1: List[int], parent2: List[int]) -> List[int]:
    """Perform one-point crossover between two parents."""
    crossover_point = random.randint(1, len(parent1) - 1)
    offspring = parent1[:crossover_point] + [
        gene for gene in parent2 if gene not in parent1[:crossover_point]
    ]
    return offspring

def cycle_crossover(parent1: List[int], parent2: List[int]) -> List[int]:
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
    return offspring  # type: ignore

def order_crossover(parent1: List[int], parent2: List[int]) -> List[int]:
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
    return offspring  # type: ignore

def run_genetic_algorithm(
    distance_matrix: List[List[float]],
    population_size: int,
    generations: int,
    mutation_rate: float,
    crossover_rate: float,
    selection_method: str = "tournament",
    crossover_method: str = "one_point",
    mutation_method: str = "swap",
) -> Tuple[List[int], float, List[float]]:
    """
    Run the genetic algorithm for a specified number of generations to solve TSP.
    Supports dynamic selection of selection, crossover, and mutation methods.
    
    Args:
        distance_matrix: A matrix containing distances between cities.
        population_size: Number of individuals in the population.
        generations: Number of generations to simulate.
        mutation_rate: Probability of mutation.
        crossover_rate: Probability of crossover.
        selection_method: Selection strategy to use.
        crossover_method: Crossover strategy to use.
        mutation_method: Mutation strategy to us.

    Returns:
        A tuple containing the best route, its distance, and the fitness progression.
    """
    # Mapping selection crossover and mutation
    selection_functions = {
        "tournament": tournament_selection,
        "elitism": elitism_selection,
        "steady_state": steady_state_selection,
    }
    crossover_functions = {
        "one_point": one_point_crossover,
        "cycle": cycle_crossover,
        "order": order_crossover,
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
