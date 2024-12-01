import argparse
import random
import numpy as np
from genetic_algorithm import run_genetic_algorithm, generate_random_coordinates, generate_distance_matrix
from visualization import Visualization

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
            coordinates = generate_random_coordinates(num_cities[0])
            distance_matrix = generate_distance_matrix(coordinates)

            best_route, best_distance, best_results = run_genetic_algorithm(
                distance_matrix,
                population_size[0],
                generations[0],
                mutation_rate[0],
                crossover_rate[0]
            )

            self.viz.add_results(best_results, "Single Execution")

            # Display results
            print("\nSingle Execution:")
            print("Best Route Found:", best_route)
            print("Best Distance:", best_distance)

            # Visualize the best route
            self.viz.plot_route(best_route, coordinates)
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
                    float(fixed_params["crossover_rate"]) if test_param != "crossover_rate" else float(test_value)
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
