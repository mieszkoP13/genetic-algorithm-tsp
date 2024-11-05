import random

def generate_random_route(num_cities):
    """Generate a random route for a given number of cities."""
    route = list(range(num_cities))
    random.shuffle(route)
    return route

def initialize_population(population_size, num_cities):
    """Initialize a population of a given size with random routes."""
    return [generate_random_route(num_cities) for _ in range(population_size)]
