import argparse
from simulated_annealing.simulated_annealing import SimulatedAnnealing
from genetic_algorithm.genetic_algorithm import GeneticAlgorithm
from genetic_algorithm.visualization import Visualization
import pandas as pd
import numpy as np

class TSP_SA_CLI:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Simulated Annealing for the Traveling Salesman Problem (TSP)")
        self.config()
        self.viz = Visualization()

    def config(self):
        self.parser.add_argument("-n", "--num-cities", type=int, required=True, help="Number of cities.")
        self.parser.add_argument("-t", "--initial-temperature", type=float, default=1000, help="Initial temperature.")
        self.parser.add_argument("-x", "--cooling-rate", type=float, default=0.995, help="Cooling rate.")
        self.parser.add_argument("-e", "--stop-temperature", type=float, default=1e-3, help="Stopping temperature.")
        self.parser.add_argument("-s", "--fixed-seed", action="store_true",
                                 help="Use a fixed seed (42) for random number generation to ensure reproducibility.")
        self.parser.add_argument("-r", "--repeats", type=int, default=1,
                                 help="Number of times to repeat the experiment for averaging. Default is 1.")

    def run(self, args=None):
        if args:
            args = self.parser.parse_args(args)
        else:
            args = self.parser.parse_args()

        if args.fixed_seed:
            import random
            random.seed(42)

        # Generate random TSP problem
        coordinates = GeneticAlgorithm.generate_random_coordinates(args.num_cities)
        distance_matrix = GeneticAlgorithm.generate_distance_matrix(coordinates)

        # Initialize variables for statistics collection
        stats_data = []
        best_results_for_repeats = []

        for repeat in range(args.repeats):
            print(f"Run {repeat + 1}/{args.repeats}")

            # Initialize Simulated Annealing with generated distance matrix
            sa = SimulatedAnnealing(
                distance_matrix=distance_matrix,
                initial_temperature=args.initial_temperature,
                cooling_rate=args.cooling_rate,
                stop_temperature=args.stop_temperature,
            )

            # Run Simulated Annealing
            best_route, best_distance, best_results = sa.run()

            # Store results for current repeat
            best_results_for_repeats.append(best_results)

            # Display results
            print(f"Best Distance = {best_distance:.2f}")

        # Collect statistics
        min_values = [min(results) for results in best_results_for_repeats]
        stats_data.append({
            'initial_temperature': args.initial_temperature,
            'cooling_rate': args.cooling_rate,
            'stop_temperature': args.stop_temperature,
            'mean_min': np.mean(min_values),
            'std_min': np.std(min_values),
            'p25_min': np.percentile(min_values, 25),
            'p50_min': np.percentile(min_values, 50),
            'p75_min': np.percentile(min_values, 75),
            'max_min': np.max(min_values)
        })

        y_mean = np.mean(best_results_for_repeats, axis=0)
        self.viz.add_results(y_mean, "label", "red")

        # Visualize results for all combinations
        self.viz.plot_best_results()

        # Convert stats data to a DataFrame
        if args.repeats > 1:
            stats_df = pd.DataFrame(stats_data)
            stats_df.set_index(['initial_temperature', 'cooling_rate', 'stop_temperature'], inplace=True)
            print("\nStatistics:")
            print(stats_df)
