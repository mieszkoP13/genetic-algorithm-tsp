import numpy as np
import random

class SimulatedAnnealing:
    def __init__(self, distance_matrix=None, initial_temperature=1000, cooling_rate=0.995, stop_temperature=1e-3):
        self.distance_matrix = np.array(distance_matrix)
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.stop_temperature = stop_temperature

    def _calculate_distance(self, route):
        """
        Calculate the total distance of a given route.
        """
        return sum(self.distance_matrix[route[i], route[i + 1]] for i in range(len(route) - 1)) + \
               self.distance_matrix[route[-1], route[0]]

    def _swap(self, route):
        """
        Perform a random swap of two cities in the route.
        """
        idx1, idx2 = random.sample(range(len(route)), 2)
        route[idx1], route[idx2] = route[idx2], route[idx1]

    def run(self):
        """
        Run the Simulated Annealing algorithm.
        """
        num_cities = self.distance_matrix.shape[0]
        current_route = list(range(num_cities))
        random.shuffle(current_route)
        current_distance = self._calculate_distance(current_route)

        best_route = current_route[:]
        best_distance = current_distance

        temperature = self.initial_temperature

        while temperature > self.stop_temperature:
            new_route = current_route[:]
            self._swap(new_route)
            new_distance = self._calculate_distance(new_route)

            if new_distance < current_distance or random.random() < np.exp((current_distance - new_distance) / temperature):
                current_route = new_route
                current_distance = new_distance

                if current_distance < best_distance:
                    best_route = current_route[:]
                    best_distance = current_distance

            temperature *= self.cooling_rate

        return best_route, best_distance
