import argparse
import random
from genetic_algorithm import run_genetic_algorithm, generate_random_coordinates, generate_distance_matrix
from visualization import plot_route

class TSPCLI:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Genetic Algorithm for the Traveling Salesman Problem (TSP)")
        self.config()

    def config(self):
        self.parser.add_argument("-n", "--num-cities", type=int, required=True,
                                 help="Number of cities in the TSP problem.")
        self.parser.add_argument("-p", "--population-size", type=int, required=True,
                                 help="Size of the population for the genetic algorithm.")
        self.parser.add_argument("-g", "--generations", type=int, required=True,
                                 help="Number of generations to run the genetic algorithm.")
        self.parser.add_argument("-m", "--mutation-rate", type=float, required=True,
                                 help="Mutation rate for the genetic algorithm (e.g., 0.1).")
        self.parser.add_argument("-c", "--crossover-rate", type=float, required=True,
                                 help="Crossover rate for the genetic algorithm (e.g., 0.8).")
        self.parser.add_argument("-s", "--fixed-seed", action="store_true",
                                 help="Use a fixed seed (42) for random number generation to ensure reproducibility.")

    def run(self):
        args = self.parser.parse_args()

        # Set seed if fixed-seed option is provided
        if args.fixed_seed:
            random.seed(42)

        # Generate random city coordinates
        coordinates = generate_random_coordinates(args.num_cities)
        
        # Generate the distance matrix from coordinates
        distance_matrix = generate_distance_matrix(coordinates)

        # Run the genetic algorithm
        best_route, best_distance = run_genetic_algorithm(
            distance_matrix, 
            args.population_size, 
            args.generations, 
            args.mutation_rate, 
            args.crossover_rate
        )

        # Display results
        print("Best Route Found:", best_route)
        print("Best Distance:", best_distance)

        # Visualize the best route
        plot_route(best_route, coordinates)
