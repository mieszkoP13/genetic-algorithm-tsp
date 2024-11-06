import random
from fitness import calculate_route_distance

def generate_random_route(num_cities):
    """Generate a random route for a given number of cities."""
    route = list(range(num_cities))
    random.shuffle(route)
    return route

def initialize_population(population_size, num_cities):
    """Initialize a population of a given size with random routes."""
    return [generate_random_route(num_cities) for _ in range(population_size)]

def one_point_crossover(parent1, parent2):
    """Perform one-point crossover on two parent routes to produce an offspring."""
    num_cities = len(parent1)
    crossover_point = random.randint(1, num_cities - 2)  # Avoiding the ends

    # Start with the first part from parent1, then complete with parent2's order
    child = parent1[:crossover_point]
    child += [city for city in parent2 if city not in child]

    return child

def mutate(route, mutation_rate=0.1):
    """Apply mutation by swapping two cities with a certain probability."""
    for i in range(len(route)):
        if random.random() < mutation_rate:
            # Select another city to swap with
            swap_idx = random.randint(0, len(route) - 1)
            # Swap the cities
            route[i], route[swap_idx] = route[swap_idx], route[i]

def calculate_fitness(population, distance_matrix):
    """Calculate the fitness for each individual in the population."""
    return [calculate_route_distance(route, distance_matrix) for route in population]

def select_parents(population, fitness):
    """Select parents based on their fitness using a tournament selection."""
    tournament_size = 3
    selected_parents = []

    for _ in range(len(population) // 2):  # Select pairs of parents
        tournament = random.sample(list(zip(population, fitness)), tournament_size)
        winner = min(tournament, key=lambda x: x[1])  # Select the one with the lowest distance
        selected_parents.append(winner[0])

    return selected_parents

def run_genetic_algorithm(distance_matrix, population_size, generations, mutation_rate):
    """Run the genetic algorithm for a number of generations."""
    population = initialize_population(population_size, len(distance_matrix))

    for generation in range(generations):
        fitness = calculate_fitness(population, distance_matrix)

        # Debug: Print the current population and fitness values
        print(f"Generation {generation + 1}, Fitness values: {fitness}")

        if not fitness:
            print("Error: Fitness list is empty!")
            break

        # Selection
        parents = select_parents(population, fitness)

        # Ensure enough parents are selected to generate the full population
        next_generation = []
        while len(next_generation) < population_size:
            for i in range(0, len(parents), 2):
                if len(next_generation) >= population_size:
                    break
                parent1 = parents[i]
                parent2 = parents[i + 1] if i + 1 < len(parents) else parents[0]
                offspring = one_point_crossover(parent1, parent2)
                mutate(offspring, mutation_rate)
                next_generation.append(offspring)

        population = next_generation

        # Print the best fitness value of the current generation
        print(f"Generation {generation + 1}, Best Distance: {min(fitness)}")

    # Return the best solution found
    best_index = fitness.index(min(fitness))
    return population[best_index], min(fitness)  # Return best route and its distance
