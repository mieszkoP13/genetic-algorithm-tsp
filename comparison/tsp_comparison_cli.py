import argparse
from genetic_algorithm.tsp_cli import TSPCLI
from simulated_annealing.tsp_sa_cli import TSP_SA_CLI

class TSPComparisonCLI:
    def __init__(self):
        # Create main parser for `compare`
        self.parser = argparse.ArgumentParser(description="Compare Genetic Algorithm and Simulated Annealing for TSP")
        self.config()

    def config(self):
        # Args shared for both Algorithms
        self.parser.add_argument("-n", "--num-cities", type=int, required=True, help="Number of cities.")

        # Args for GA
        self.parser.add_argument("-p", "--population-size", type=int, required=True, help="Population size for GA.")
        self.parser.add_argument("-g", "--generations", type=int, required=True, help="Number of generations for GA.")
        self.parser.add_argument("-m", "--mutation-rate", type=float, required=True, help="Mutation rate for GA.")
        self.parser.add_argument("-c", "--crossover-rate", type=float, required=True, help="Crossover rate for GA.")

        # Args for SA
        self.parser.add_argument("-t", "--initial-temperature", type=float, required=True, help="Initial temperature for SA.")
        self.parser.add_argument("-x", "--cooling-rate", type=float, required=True, help="Cooling rate for SA.")
        self.parser.add_argument("-e", "--stop-temperature", type=float, required=True, help="Stop temperature for SA.")

    def run(self):
        args = self.parser.parse_args()

        # Pass args to GA
        ga_args = [
            "-n", str(args.num_cities),
            "-p", str(args.population_size),
            "-g", str(args.generations),
            "-m", str(args.mutation_rate),
            "-c", str(args.crossover_rate),
        ]

        # Pass args to SA
        sa_args = [
            "-n", str(args.num_cities),
            "-t", str(args.initial_temperature),
            "-x", str(args.cooling_rate),
            "-e", str(args.stop_temperature),
        ]

        # Run GA
        print("\nRunning Genetic Algorithm...")
        TSPCLI().run(args=ga_args)

        # Run SA
        print("\nRunning Simulated Annealing...")
        TSP_SA_CLI().run(args=sa_args)

