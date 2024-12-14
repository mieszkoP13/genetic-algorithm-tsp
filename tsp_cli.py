import argparse
import random
import numpy as np
import pandas as pd
from genetic_algorithm import run_genetic_algorithm, generate_random_coordinates, generate_distance_matrix
from visualization import Visualization

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

class TSPCLI:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Genetic Algorithm for the Traveling Salesman Problem (TSP)")
        self.config()
        self.viz = Visualization()

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
        Parses a range argument (1 value = fixed, 3 values = range).
        Returns a list of values.
        """
        if len(values) == 1:
            return [values[0]]
        elif len(values) == 3:
            # Determine type based on first value
            value_type = float if isinstance(values[0], float) or "." in str(values[0]) else int
            start, step, stop = map(value_type, values)
            return list(np.arange(start, stop + step, step))  # Inclusive stop
        else:
            raise ValueError("Provide either 1 value (fixed) or 3 values (start, step, stop).")

    def ensure_single_test_param(self, params):
        """
        Ensures that only one parameter is a range.
        """
        range_params = [key for key, val in params.items() if len(val) > 1]
        if len(range_params) > 1:
            raise ValueError(f"Only one parameter can be a range. Multiple ranges provided: {range_params}.")
        return range_params[0] if range_params else None

    def run(self):
        args = self.parser.parse_args()

        # Parse all parameters
        num_cities = self.parse_range(args.num_cities)
        population_size = self.parse_range(args.population_size)
        generations = self.parse_range(args.generations)
        mutation_rate = self.parse_range(args.mutation_rate)
        crossover_rate = self.parse_range(args.crossover_rate)
        repeats = args.repeats

        params = {
            "num_cities": num_cities,
            "population_size": population_size,
            "generations": generations,
            "mutation_rate": mutation_rate,
            "crossover_rate": crossover_rate
        }

        # Ensure only one parameter has a range
        test_param = self.ensure_single_test_param(params)

        # Set seed if fixed-seed option is provided
        if args.fixed_seed:
            random.seed(42)

        # If no parameter is a range, run the algorithm once
        if not test_param:
            # Prepare methods for iteration
            selection_methods = SELECTION_METHODS if args.selection_method == "each" else [args.selection_method]
            crossover_methods = CROSSOVER_METHODS if args.crossover_method == "each" else [args.crossover_method]
            mutation_methods = MUTATION_METHODS if args.mutation_method == "each" else [args.mutation_method]

            color_index = 0  # Start with the first color

            stats_data = []  # Collect statistics list

            for selection_method in selection_methods:
                for crossover_method in crossover_methods:
                    for mutation_method in mutation_methods:
                        # Select a color from predefined list or generate one if out of predefined colors
                        color = PREDEFINED_COLORS[color_index] if color_index < len(PREDEFINED_COLORS) else np.random.rand(3,)
            
                        best_results_for_test_value = []
                        for _ in range(repeats):
                            print(f"\nTesting Selection: {selection_method}, Crossover: {crossover_method}, Mutation: {mutation_method}")

                            # Generate coordinates and distance matrix
                            coordinates = generate_random_coordinates(num_cities[0])
                            distance_matrix = generate_distance_matrix(coordinates)

                            # Run the genetic algorithm
                            best_route, best_distance, best_results = run_genetic_algorithm(
                                distance_matrix,
                                population_size[0],
                                generations[0],
                                mutation_rate[0],
                                crossover_rate[0],
                                selection_method,
                                crossover_method,
                                mutation_method
                            )

                            best_results_for_test_value.append(best_results)

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
            stats_df = pd.DataFrame(stats_data)
            stats_df.set_index(['selectionType', 'crossoverType', 'mutationType'], inplace=True)
            print(stats_df)

            # Visualize results for all combinations
            self.viz.plot_best_results()
            return


        # Fixed parameters (exclude the test parameter)
        fixed_params = {key: val[0] for key, val in params.items() if key != test_param}

        # Test loop
        for test_value in params[test_param]:
            best_results_for_test_value = []

            color = np.random.rand(3,)
            
            for _ in range(repeats):
                #print(f"\nTesting {test_param}={test_value} with fixed parameters: {fixed_params}")

                # Generate random city coordinates
                coordinates = generate_random_coordinates(int(fixed_params["num_cities"]) if test_param != "num_cities" else int(test_value))
                
                # Generate the distance matrix from coordinates
                distance_matrix = generate_distance_matrix(coordinates)

                # Run the genetic algorithm
                best_route, best_distance, best_results = run_genetic_algorithm(
                    distance_matrix,
                    int(fixed_params["population_size"]) if test_param != "population_size" else int(test_value),
                    int(fixed_params["generations"]) if test_param != "generations" else int(test_value),
                    float(fixed_params["mutation_rate"]) if test_param != "mutation_rate" else float(test_value),
                    float(fixed_params["crossover_rate"]) if test_param != "crossover_rate" else float(test_value),
                    args.selection_method,
                    args.crossover_method,
                    args.mutation_method
                )

                best_results_for_test_value.append(best_results)
                

                # Display results
                print(f"Best Distance = {best_distance:.2f}")
            
            y_mean = np.mean(best_results_for_test_value, axis=0)
            self.viz.add_results(y_mean, f"\n{test_param}={test_value:.2f}", color)

        self.viz.plot_best_results()

        # If only one value tested, visualize the route
        if len(params[test_param]) == 1:
            self.viz.plot_route(best_route, coordinates)
