import math
import random
from typing import List, Tuple
import numpy as np


class HelperUtils:
    @staticmethod
    def calculate_distance(city1: Tuple[float, float], city2: Tuple[float, float]) -> float:
        """Calculate the Euclidean distance between two cities."""
        return math.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)

    @staticmethod
    def generate_random_coordinates(num_cities: int) -> List[Tuple[float, float]]:
        """Generate random coordinates for a given number of cities."""
        return [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(num_cities)]

    @staticmethod
    def generate_distance_matrix(coordinates: List[Tuple[float, float]]) -> List[List[float]]:
        """Generate the distance matrix from the coordinates of the cities."""
        num_cities = len(coordinates)
        distance_matrix = [[0.0] * num_cities for _ in range(num_cities)]
        for i in range(num_cities):
            for j in range(i + 1, num_cities):
                distance = HelperUtils.calculate_distance(coordinates[i], coordinates[j])
                distance_matrix[i][j] = distance_matrix[j][i] = distance
        return distance_matrix

    @staticmethod
    def parse_range(values):
        """
        Parses a range argument.
        """
        if len(values) == 3:
            value_type = float if isinstance(values[0], float) or "." in str(values[0]) else int
            start, step, stop = map(value_type, values)
            return list(np.arange(start, stop + step, step))  # Inclusive stop
        else:
            raise ValueError("Error.")

    @staticmethod
    def find_test_param_name(args) -> str:
        """
        Find and validate the test parameter.
        Ensures that exactly one parameter has 3 numeric values (start, step, stop) for testing.
        """
        test_param: str = ""
        test_params_count: int = 0

        for name, value in args.__dict__.items():
            # Skip non-list (not nargs) arguments
            if not isinstance(value, list):
                continue
            
            # Validate that the list contains numbers
            if all(isinstance(x, (float, int)) for x in value):
                if len(value) == 1:
                    continue  # Single value is valid but not a test param
                elif len(value) == 3:
                    test_param = name
                    test_params_count += 1
                else:
                    raise ValueError(
                        f"Error. Invalid number of values for '{name}': Expected 1 or 3, got {len(value)}."
                    )
            else:
                raise ValueError(f"Error. Invalid test parameter type for '{name}', choose int or float.")

        # Ensure only one test parameter is defined
        if test_params_count == 0:
            return None
        elif test_params_count == 1:
            return test_param
        else:
            raise ValueError("Error. Too many test parameters, choose only one for testing.")

    @staticmethod
    def dynamic_round(value, significant_digits=4):
        """
        Dynamically formats a number based on its magnitude.
        Ensures a specified number of significant digits.
        """
        if value == 0:
            return "0"  # Special case for zero
        magnitude = int(np.floor(np.log10(abs(value))))  # Find the order of magnitude
        decimal_places = max(significant_digits - magnitude - 1, 0)
        return f"{value:.{decimal_places}f}"
