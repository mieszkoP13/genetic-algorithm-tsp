import random
import math
from typing import List, Tuple, Optional

class GeneticAlgorithm:
    def __init__(
        self,
        distance_matrix: List[List[float]],
        population_size: int,
        generations: int,
        mutation_rate: float,
        crossover_rate: float,
        selection_method: str = "tournament",
        crossover_method: str = "one_point",
        mutation_method: str = "swap",
    ):
        self.distance_matrix = distance_matrix
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method

        # Selection, crossover, and mutation mapping
        self.selection_functions = {
            "tournament": self.tournament_selection,
            "elitism": self.elitism_selection,
            "steady_state": self.steady_state_selection,
        }
        self.crossover_functions = {
            "one_point": self.one_point_crossover,
            "cycle": self.cycle_crossover,
            "order": self.order_crossover,
        }
        self.mutation_functions = {
            "swap": self.mutate_swap,
            "adjacent_swap": self.mutate_adjacent_swap,
            "inverse": self.mutate_inverse,
            "insertion": self.mutate_insertion,
        }

    def initialize_population(self) -> List[List[int]]:
        """Initialize the population with random routes."""
        population = []
        for _ in range(self.population_size):
            route = list(range(len(self.distance_matrix)))
            random.shuffle(route)
            population.append(route)
        return population

    def calculate_fitness(self, population: List[List[int]]) -> List[float]:
        """Calculate the fitness (distance) of each individual in the population."""
        fitness = []
        for route in population:
            total_distance = 0.0
            for i in range(len(route) - 1):
                total_distance += self.distance_matrix[route[i]][route[i + 1]]
            total_distance += self.distance_matrix[route[-1]][route[0]]  # To close the loop
            fitness.append(total_distance)
        return fitness

    def mutate_swap(self, route: List[int], mutation_rate: float) -> None:
        """Perform swap mutation by swapping two cities in the route."""
        if random.random() < mutation_rate:
            idx1, idx2 = random.sample(range(len(route)), 2)
            route[idx1], route[idx2] = route[idx2], route[idx1]

    def mutate_adjacent_swap(self, route: List[int], mutation_rate: float) -> None:
        """Perform mutation by swapping two adjacent cities in the route."""
        if random.random() < mutation_rate:
            idx = random.randint(0, len(route) - 2)
            route[idx], route[idx + 1] = route[idx + 1], route[idx]

    def mutate_inverse(self, route: List[int], mutation_rate: float) -> None:
        """Perform inversion mutation by reversing a segment of the route."""
        if random.random() < mutation_rate:
            idx1, idx2 = sorted(random.sample(range(len(route)), 2))
            route[idx1:idx2] = reversed(route[idx1:idx2])

    def mutate_insertion(self, route: List[int], mutation_rate: float) -> None:
        """Perform mutation by moving a city to a new position in the route."""
        if random.random() < mutation_rate:
            city_idx = random.randint(0, len(route) - 1)
            new_pos = random.randint(0, len(route) - 1)
            city = route.pop(city_idx)
            route.insert(new_pos, city)

    def tournament_selection(self, population: List[List[int]], fitness: List[float], tournament_size: int = 6) -> List[List[int]]:
        """Select parents using tournament selection."""
        selected_parents = []
        for _ in range(len(population) // 2):
            tournament = random.sample(list(zip(population, fitness)), tournament_size)
            tournament.sort(key=lambda x: x[1])
            selected_parents.append(tournament[0][0])
        return selected_parents

    def elitism_selection(self, population: List[List[int]], fitness: List[float], elite_size: int = 2) -> List[List[int]]:
        """Select the best individuals using elitism."""
        sorted_indices = sorted(range(len(fitness)), key=lambda i: fitness[i])
        return [population[i] for i in sorted_indices[:elite_size]]

    def steady_state_selection(self, population: List[List[int]], fitness: List[float], num_survivors: Optional[int] = None) -> List[List[int]]:
        """Perform steady-state selection by keeping the best individuals."""
        if num_survivors is None:
            num_survivors = len(population) // 2
        sorted_population = [x for _, x in sorted(zip(fitness, population))]
        return sorted_population[:num_survivors]

    def one_point_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Perform one-point crossover between two parents."""
        crossover_point = random.randint(1, len(parent1) - 1)
        offspring = parent1[:crossover_point] + [gene for gene in parent2 if gene not in parent1[:crossover_point]]
        return offspring

    def cycle_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
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

    def order_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
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

    def run(self) -> Tuple[List[int], float, List[float]]:
        """Run the genetic algorithm for the specified number of generations."""
        select_func = self.selection_functions[self.selection_method]
        crossover_func = self.crossover_functions[self.crossover_method]
        mutate_func = self.mutation_functions[self.mutation_method]

        population = self.initialize_population()
        best_results = []

        for generation in range(self.generations):
            fitness = self.calculate_fitness(population)
            parents = select_func(population, fitness)

            next_generation = []
            while len(next_generation) < self.population_size:
                for i in range(0, len(parents), 2):
                    if len(next_generation) >= self.population_size:
                        break
                    parent1 = parents[i]
                    parent2 = parents[i + 1] if i + 1 < len(parents) else parents[0]

                    # Perform crossover
                    if random.random() < self.crossover_rate:
                        offspring = crossover_func(parent1, parent2)
                    else:
                        offspring = parent1.copy()

                    # Apply mutation
                    mutate_func(offspring, self.mutation_rate)
                    next_generation.append(offspring)

            population = next_generation
            best_distance = min(fitness)
            best_results.append(best_distance)

        fitness = self.calculate_fitness(population)
        best_index = fitness.index(min(fitness))
        return population[best_index], min(fitness), best_results
