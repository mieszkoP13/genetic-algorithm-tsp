import random

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
