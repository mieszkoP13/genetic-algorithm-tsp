import argparse
from simulated_annealing.simulated_annealing import SimulatedAnnealing
from utils.helpers import HelperUtils
from genetic_algorithm.visualization import Visualization
import pandas as pd
import numpy as np

class TSP_SA_CLI:
    def __init__(self, args=None, viz=Visualization()):
        self.parser = argparse.ArgumentParser(description="Simulated Annealing for the Traveling Salesman Problem (TSP)")
        self.config()
        self.args = self.parser.parse_args(args)
        self.viz = viz

    def config(self):
        self.parser.add_argument("-n", "--num-cities", type=int, required=True,
                                 help="Number of cities.")
        self.parser.add_argument("-t", "--initial-temperature", type=float, nargs="+", required=True,
                                 help="Initial temperature. Provide one value or three values for range testing (start, step, stop).")
        self.parser.add_argument("-x", "--cooling-rate", type=float, nargs="+", required=True,
                                 help="Cooling rate. Provide one value or three values for range testing (start, step, stop).")
        self.parser.add_argument("-e", "--stop-temperature", type=float, nargs="+", required=True,
                                 help="Stopping temperature. Provide one value or three values for range testing (start, step, stop).")
        self.parser.add_argument("-s", "--fixed-seed", action="store_true",
                                 help="Use a fixed seed (42) for random number generation to ensure reproducibility.")
        self.parser.add_argument("-r", "--repeats", type=int, default=1,
                                 help="Number of times to repeat the experiment for averaging. Default is 1.")

    def test_param_ranges(self):
        test_param = self.test_param
        setattr(self.args, test_param, HelperUtils.parse_range(getattr(self.args, test_param)))

        stats_data = []  # Collect statistics list

        for test_value in getattr(self.args, test_param):
            best_results_for_test_value = []

            # Generate random TSP problem
            coordinates = HelperUtils.generate_random_coordinates(self.args.num_cities)
            distance_matrix = HelperUtils.generate_distance_matrix(coordinates)

            for _ in range(self.args.repeats):
                sa = SimulatedAnnealing(
                    distance_matrix=distance_matrix,
                    initial_temperature=self.args.initial_temperature[0] if test_param != "initial_temperature" else float(test_value),
                    cooling_rate=self.args.cooling_rate[0] if test_param != "cooling_rate" else float(test_value),
                    stop_temperature=self.args.stop_temperature[0] if test_param != "stop_temperature" else float(test_value),
                )

                # Run the Simulated Annealing algorithm
                best_route, best_distance, best_results = sa.run()
                best_results_for_test_value.append(best_results)

                # Display results
                print(f"Testing {test_param} = {HelperUtils.dynamic_round(test_value)}")
                print(f"Best Distance = {HelperUtils.dynamic_round(best_distance)}")

            # Collect statistics for the current test value
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

            color = np.random.rand(3,)
            y_mean = np.mean(best_results_for_test_value, axis=0)
            self.viz.add_results(y_mean, f"\n{test_param}={HelperUtils.dynamic_round(test_value)}", color)

        # Convert stats data to a DataFrame if enough data
        if self.args.repeats > 1:
            stats_df = pd.DataFrame(stats_data)
            stats_df.set_index([f'{test_param}'], inplace=True)
            print(stats_df)

    def single_execution(self):
        # Generate random TSP problem
        coordinates = HelperUtils.generate_random_coordinates(self.args.num_cities)
        distance_matrix = HelperUtils.generate_distance_matrix(coordinates)

        # Initialize variables for statistics collection
        stats_data = []
        best_results_for_repeats = []

        for repeat in range(self.args.repeats):
            print(f"Run {repeat + 1}/{self.args.repeats}")

            # Initialize Simulated Annealing with generated distance matrix
            sa = SimulatedAnnealing(
                distance_matrix=distance_matrix,
                initial_temperature=self.args.initial_temperature[0],
                cooling_rate=self.args.cooling_rate[0],
                stop_temperature=self.args.stop_temperature[0],
            )

            # Run Simulated Annealing
            best_route, best_distance, best_results = sa.run()

            # Store results for current repeat
            best_results_for_repeats.append(best_results)

            # Display results
            print(f"Best Distance = {HelperUtils.dynamic_round(best_distance)}")

        # Collect statistics
        min_values = [min(results) for results in best_results_for_repeats]
        stats_data.append({
            'initial_temperature': self.args.initial_temperature[0],
            'cooling_rate': self.args.cooling_rate[0],
            'stop_temperature': self.args.stop_temperature[0],
            'mean_min': np.mean(min_values),
            'std_min': np.std(min_values),
            'p25_min': np.percentile(min_values, 25),
            'p50_min': np.percentile(min_values, 50),
            'p75_min': np.percentile(min_values, 75),
            'max_min': np.max(min_values)
        })

        y_mean = np.mean(best_results_for_repeats, axis=0)
        self.viz.add_results(y_mean, f"t = {self.args.initial_temperature[0]}, cr = {self.args.cooling_rate[0]}, e = {self.args.stop_temperature[0]}", "red")

        # Convert stats data to a DataFrame
        # only if there is sufficient data
        if self.args.repeats > 1:
            stats_df = pd.DataFrame(stats_data)
            stats_df.set_index(['initial_temperature', 'cooling_rate', 'stop_temperature'], inplace=True)
            print(stats_df)

    def run(self):
        self.test_param = HelperUtils.find_test_param_name(self.args)

        if self.args.fixed_seed:
            import random
            random.seed(42)

        if not self.test_param:
            self.single_execution()
        else:
            self.test_param_ranges()
