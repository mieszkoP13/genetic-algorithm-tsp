import argparse
from simulated_annealing.simulated_annealing import SimulatedAnnealing
from genetic_algorithm.genetic_algorithm import GeneticAlgorithm

class TSP_SA_CLI:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Simulated Annealing for the Traveling Salesman Problem (TSP)")
        self.config()

    def config(self):
        self.parser.add_argument("-n", "--num-cities", type=int, required=True, help="Number of cities.")
        self.parser.add_argument("-t", "--initial-temperature", type=float, default=1000, help="Initial temperature.")
        self.parser.add_argument("-x", "--cooling-rate", type=float, default=0.995, help="Cooling rate.")
        self.parser.add_argument("-e", "--stop-temperature", type=float, default=1e-3, help="Stopping temperature.")
        self.parser.add_argument("-s", "--fixed-seed", action="store_true",
                                 help="Use a fixed seed (42) for random number generation to ensure reproducibility.")

    def run(self, args=None):
        if args:
            args = self.parser.parse_args(args)
        else:
            args = self.parser.parse_args()

        # Generate random TSP problem
        sa = SimulatedAnnealing()
        coordinates = GeneticAlgorithm.generate_random_coordinates(args.num_cities)
        distance_matrix = GeneticAlgorithm.generate_distance_matrix(coordinates)

        # Initialize Simulated Annealing with generated distance matrix
        sa = SimulatedAnnealing(
            distance_matrix=distance_matrix,
            initial_temperature=args.initial_temperature,
            cooling_rate=args.cooling_rate,
            stop_temperature=args.stop_temperature,
        )

        # Run Simulated Annealing
        best_route, best_distance = sa.run()

        print(f"Best Route: {best_route}")
        print(f"Best Distance: {best_distance:.2f}")
