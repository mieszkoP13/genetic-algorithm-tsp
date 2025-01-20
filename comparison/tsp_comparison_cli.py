import argparse
from genetic_algorithm.tsp_ga_cli import TSP_GA_CLI
from simulated_annealing.tsp_sa_cli import TSP_SA_CLI
from genetic_algorithm.visualization import Visualization

class TSPComparisonCLI:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Compare Genetic Algorithm and Simulated Annealing for TSP")
        self.config()
        self.viz = Visualization()

    def config(self):
        # Args shared for both Algorithms
        self.parser.add_argument("-n", "--num-cities", type=int, help="Number of cities.")
        self.parser.add_argument("-r", "--repeats", type=int, default=1,
                                 help="Number of times to repeat the experiment for averaging. Default is 1.")

        # Args for GA
        self.parser.add_argument("-p", "--population-size", type=int, help="Population size for GA.")
        self.parser.add_argument("-g", "--generations", type=int, help="Number of generations for GA.")
        self.parser.add_argument("-m", "--mutation-rate", type=float, help="Mutation rate for GA.")
        self.parser.add_argument("-c", "--crossover-rate", type=float, help="Crossover rate for GA.")

        # Args for SA
        self.parser.add_argument("-t", "--initial-temperature", type=float, help="Initial temperature for SA.")
        self.parser.add_argument("-x", "--cooling-rate", type=float, help="Cooling rate for SA.")
        self.parser.add_argument("-e", "--stop-temperature", type=float, help="Stop temperature for SA.")

        # Optional: Predefined parameter tests
        self.parser.add_argument("--preset-tests", action="store_true", 
                                 help="Run predefined parameter tests.")

    def run(self):
        args = self.parser.parse_args()

        if args.preset_tests:
            self.run_preset_tests(args.num_cities, args.repeats)
        else:
            self.run_single_comparison(args)

    def run_single_comparison(self, args):
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

        # Run GA and SA
        print("\nRunning Genetic Algorithm...")
        TSP_GA_CLI(ga_args, self.viz).run()

        print("\nRunning Simulated Annealing...")
        TSP_SA_CLI(sa_args, self.viz).run()

    def run_preset_tests(self, num_cities, repeats):
        # Predefined parameter sets for testing
        ga_tests = [
            {"population_size": 50, "generations": 3000, "mutation_rate": 0.1, "crossover_rate": 0.8},
            {"population_size": 100, "generations": 3000, "mutation_rate": 0.2, "crossover_rate": 0.9},
            {"population_size": 200, "generations": 3000, "mutation_rate": 0.05, "crossover_rate": 0.85},
        ]
        sa_tests = [
            {"initial_temperature": 1000, "cooling_rate": 0.995, "stop_temperature": 0.001},
            {"initial_temperature": 500, "cooling_rate": 0.99, "stop_temperature": 0.01},
            {"initial_temperature": 2000, "cooling_rate": 0.997, "stop_temperature": 0.0005},
        ]

        for ga_params in ga_tests:
            print(f"\nRunning GA with params: {ga_params}")

            ga_args = [
                "-n", str(num_cities),
                "-p", str(ga_params["population_size"]),
                "-g", str(ga_params["generations"]),
                "-m", str(ga_params["mutation_rate"]),
                "-c", str(ga_params["crossover_rate"]),
                "-r", str(repeats),
            ]

            # Run GA and SA
            TSP_GA_CLI(ga_args, self.viz).run()

        for sa_params in sa_tests:
            print(f"Running SA with params: {sa_params}")

            sa_args = [
                "-n", str(num_cities),
                "-t", str(sa_params["initial_temperature"]),
                "-x", str(sa_params["cooling_rate"]),
                "-e", str(sa_params["stop_temperature"]),
                "-r", str(repeats),
            ]

            # Run SA
            TSP_SA_CLI(sa_args, self.viz).run()
