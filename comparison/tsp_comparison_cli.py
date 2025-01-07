import argparse
from genetic_algorithm.tsp_ga_cli import TSP_GA_CLI
from simulated_annealing.tsp_sa_cli import TSP_SA_CLI
from genetic_algorithm.visualization import Visualization

class TSPComparisonCLI:
    def __init__(self):
        # Create main parser for `compare`
        self.parser = argparse.ArgumentParser(description="Compare Genetic Algorithm and Simulated Annealing for TSP")
        self.config()
        self.viz = Visualization()

    def config(self):
        # Args shared for both Algorithms
        self.parser.add_argument("-n", "--num-cities", type=int, required=True, help="Number of cities.")
        self.parser.add_argument("-r", "--repeats", type=int, default=1,
                            help="Number of times to repeat the experiment for averaging. Default is 1.")

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
            "-r", str(args.repeats),
        ]

        # Pass args to SA
        sa_args = [
            "-n", str(args.num_cities),
            "-t", str(args.initial_temperature),
            "-x", str(args.cooling_rate),
            "-e", str(args.stop_temperature),
            "-r", str(args.repeats),
        ]

        # Run GA
        print("\nRunning Genetic Algorithm...")
        TSP_GA_CLI(ga_args, self.viz).run()

        # Run SA
        print("\nRunning Simulated Annealing...")
        TSP_SA_CLI(sa_args, self.viz).run()
