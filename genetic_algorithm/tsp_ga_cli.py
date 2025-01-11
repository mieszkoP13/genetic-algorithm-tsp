import argparse
import random
import numpy as np
import pandas as pd
from typing import Any, Optional, List
from genetic_algorithm.visualization import Visualization
from genetic_algorithm.genetic_algorithm import GeneticAlgorithm

SELECTION_METHODS = ["tournament", "elitism", "steady_state"]
CROSSOVER_METHODS = ["one_point", "cycle", "order"]
MUTATION_METHODS = ["swap", "adjacent_swap", "inverse", "insertion"]

PREDEFINED_COLORS = [
    (1.0, 0.0, 0.0),  # Red
    (0.0, 1.0, 0.0),  # Green
    (0.0, 0.0, 1.0),  # Blue
    (1.0, 1.0, 0.0),  # Yellow
    (1.0, 0.0, 1.0),  # Magenta
    (0.0, 1.0, 1.0),  # Cyan
    (0.5, 0.5, 0.5),  # Gray
    (0.8, 0.4, 0.0),  # Orange
    (0.4, 0.0, 0.8),  # Purple
    (0.0, 0.4, 0.8)   # Light Blue
]

class TSP_GA_CLI:
    def __init__(self, args=None, viz=Visualization()):
        self.parser = argparse.ArgumentParser(description="Genetic Algorithm for the Traveling Salesman Problem (TSP)")
        self.config()
        self.args = self.parser.parse_args(args)
        self.viz = viz

    def config(self):
        self.parser.add_argument("-n", "--num-cities", type=int, nargs="+", required=True,
                                 help="Number of cities. Provide one value (e.g., 6) or three values for range testing (start, step, stop).")
        self.parser.add_argument("-p", "--population-size", type=int, nargs="+", required=True,
                                 help="Population size. Provide one value (e.g., 50) or three values for range testing (start, step, stop).")
        self.parser.add_argument("-g", "--generations", type=int, nargs="+", required=True,
                                 help="Number of generations. Provide one value (e.g., 100) or three values for range testing (start, step, stop).")
        self.parser.add_argument("-m", "--mutation-rate", type=float, nargs="+", required=True,
                                 help="Mutation rate. Provide one value (e.g., 0.1) or three values for range testing (start, step, stop).")
        self.parser.add_argument("-c", "--crossover-rate", type=float, nargs="+", required=True,
                                 help="Crossover rate. Provide one value (e.g., 0.8) or three values for range testing (start, step, stop).")
        
        self.parser.add_argument("--selection-method", choices=["tournament", "elitism", "steady_state", "each"], default="tournament",
                                 help="Method of selection: 'tournament' (default), 'elitism', or 'each' to iterate through all.")
        self.parser.add_argument("--crossover-method", choices=["one_point", "cycle", "order", "each"], default="one_point",
                                 help="Method of crossover: 'one_point' (default), 'cycle', 'order', or 'each' to iterate through all.")
        self.parser.add_argument("--mutation-method", choices=["swap", "adjacent_swap", "inverse", "insertion", "each"], default="swap",
                                 help="Method of mutation: 'swap' (default), 'inverse', or 'each' to iterate through all.")


        self.parser.add_argument("-s", "--fixed-seed", action="store_true",
                                 help="Use a fixed seed (42) for random number generation to ensure reproducibility.")
        self.parser.add_argument("-r", "--repeats", type=int, default=1,
                                 help="Number of times to repeat the experiment for averaging. Default is 1.")

    def parse_range(self, values):
        """
        Parses a range argument.
        """
        if len(values) == 3:
            value_type = float if isinstance(values[0], float) or "." in str(values[0]) else int
            start, step, stop = map(value_type, values)
            return list(np.arange(start, stop + step, step))  # Inclusive stop
        else:
            raise ValueError("Error.")

    def find_test_param_name(self) -> str:
        """
        Find and validate the test parameter.
        Ensures that exactly one parameter has 3 numeric values (start, step, stop) for testing.
        """
        test_param: str = ""
        test_params_count: int = 0

        for name, value in self.args.__dict__.items():
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

    def run(self):
        test_param = self.find_test_param_name()

        # Set seed if fixed-seed option is provided
        if self.args.fixed_seed:
            random.seed(42)

        # If no parameter is a range, run the algorithm once
        if not test_param:
            # Prepare methods for iteration
            selection_methods = SELECTION_METHODS if self.args.selection_method == "each" else [self.args.selection_method]
            crossover_methods = CROSSOVER_METHODS if self.args.crossover_method == "each" else [self.args.crossover_method]
            mutation_methods = MUTATION_METHODS if self.args.mutation_method == "each" else [self.args.mutation_method]

            color_index = 0  # Start with the first color

            stats_data = []  # Collect statistics list

            # Generate coordinates and distance matrix
            coordinates = GeneticAlgorithm.generate_random_coordinates(self.args.num_cities[0])
            distance_matrix = GeneticAlgorithm.generate_distance_matrix(coordinates)

            for selection_method in selection_methods:
                for crossover_method in crossover_methods:
                    for mutation_method in mutation_methods:
                        # Select a color from predefined list or generate one if out of predefined colors
                        color = PREDEFINED_COLORS[color_index] if color_index < len(PREDEFINED_COLORS) else np.random.rand(3,)

                        best_results_for_test_value = []
                        for _ in range(self.args.repeats):

                            # Initialize the Genetic Algorithm with current parameters
                            ga = GeneticAlgorithm(
                                distance_matrix=distance_matrix,
                                population_size=self.args.population_size[0],
                                generations=self.args.generations[0],
                                mutation_rate=self.args.mutation_rate[0],
                                crossover_rate=self.args.crossover_rate[0],
                                selection_method=selection_method,
                                crossover_method=crossover_method,
                                mutation_method=mutation_method,
                            )

                            # Run the genetic algorithm
                            best_route, best_distance, best_results = ga.run()

                            best_results_for_test_value.append(best_results)

                            # Display results
                            print(f"Testing Selection: {selection_method}, Crossover: {crossover_method}, Mutation: {mutation_method}")
                            print(f"Best Distance = {best_distance:.2f}")


                        # Collect statistics for the current method combination
                        min_values = [min(results) for results in best_results_for_test_value]
                        stats_data.append({
                            'selectionType': selection_method,
                            'crossoverType': crossover_method,
                            'mutationType': mutation_method,
                            'mean_min': np.mean(min_values),
                            'std_min': np.std(min_values),
                            'p25_min': np.percentile(min_values, 25),
                            'p50_min': np.percentile(min_values, 50),
                            'p75_min': np.percentile(min_values, 75),
                            'max_min': np.max(min_values)
                        })

                        # Update color for the next combination
                        color_index = (color_index + 1) % (len(PREDEFINED_COLORS) + 1)
                        label = f"{selection_method} + {crossover_method} + {mutation_method}"
                        y_mean = np.mean(best_results_for_test_value, axis=0)
                        self.viz.add_results(y_mean, label, color)

            # Convert stats data to a DataFrame
            # only if there is sufficient data
            if self.args.repeats > 1:
                stats_df = pd.DataFrame(stats_data)
                stats_df.set_index(['selectionType', 'crossoverType', 'mutationType'], inplace=True)
                print(stats_df)

            return


        # Parse the range of a test parameter
        setattr(self.args, test_param, self.parse_range(getattr(self.args, test_param)))

        stats_data = []  # Collect statistics list

        # Generate random city coordinates
        coordinates = GeneticAlgorithm.generate_random_coordinates(
            self.args.num_cities[0] if test_param != "num_cities" else int(test_value)
        )

        # Generate the distance matrix from coordinates
        distance_matrix = GeneticAlgorithm.generate_distance_matrix(coordinates)

        # Test loop
        for test_value in getattr(self.args, test_param):
            best_results_for_test_value = []

            color = np.random.rand(3,)  # Random color for plotting, if needed

            for _ in range(self.args.repeats):
                # Initialize the Genetic Algorithm with current test parameters
                ga = GeneticAlgorithm(
                    distance_matrix=distance_matrix,
                    population_size=self.args.population_size[0] if test_param != "population_size" else int(test_value),
                    generations=self.args.generations[0] if test_param != "generations" else int(test_value),
                    mutation_rate=self.args.mutation_rate[0] if test_param != "mutation_rate" else float(test_value),
                    crossover_rate=self.args.crossover_rate[0] if test_param != "crossover_rate" else float(test_value),
                    selection_method=self.args.selection_method,
                    crossover_method=self.args.crossover_method,
                    mutation_method=self.args.mutation_method,
                )

                # Run the genetic algorithm
                best_route, best_distance, best_results = ga.run()

                best_results_for_test_value.append(best_results)

                # Display results
                print(f"Testing {test_param} = {test_value:.2f}")
                print(f"Best Distance = {best_distance:.2f}")

            # Collect statistics for the current method combination
            min_values = [min(results) for results in best_results_for_test_value]
            stats_data.append({
                f'{test_param}': test_value,
                'mean_min': np.mean(min_values),
                'std_min': np.std(min_values),
                'p25_min': np.percentile(min_values, 25),
                'p50_min': np.percentile(min_values, 50),
                'p75_min': np.percentile(min_values, 75),
                'max_min': np.max(min_values)
            })
            
            y_mean = np.mean(best_results_for_test_value, axis=0)
            self.viz.add_results(y_mean, f"\n{test_param}={test_value:.2f}", color)

        # Convert stats data to a DataFrame
        # only if there is sufficient data
        if self.args.repeats > 1:
            stats_df = pd.DataFrame(stats_data)
            stats_df.set_index([f'{test_param}'], inplace=True)
            print(stats_df)

        # # If only one value tested, visualize the route
        # if len(params[test_param]) == 1:
        #     self.viz.plot_route(best_route, coordinates)
