import unittest
from genetic_algorithm import GeneticAlgorithm

class TestGeneticAlgorithm(unittest.TestCase):
    """
    Unit test suite for the GeneticAlgorithm class.
    This includes tests for methods related to distance calculation,
    mutation, crossover, selection, and the overall algorithm run.
    """
    
    def setUp(self):
        """
        Set up a basic test environment for the GeneticAlgorithm instance.
        This is run before each test case to ensure consistent setup.
        """
        # Sample distance matrix for testing
        self.distance_matrix = [[0, 10, 15], [10, 0, 20], [15, 20, 0]]
        self.population_size = 10
        self.generations = 100
        self.mutation_rate = 0.01
        self.crossover_rate = 0.7
        
        # Initialize the GeneticAlgorithm instance
        self.ga = GeneticAlgorithm(
            distance_matrix=self.distance_matrix,
            population_size=self.population_size,
            generations=self.generations,
            mutation_rate=self.mutation_rate,
            crossover_rate=self.crossover_rate
        )

    def test_calculate_distance(self):
        """
        Test the calculate_distance method.
        Ensure that it correctly computes the Euclidean distance
        between two points.
        """
        distance = self.ga.calculate_distance((0, 0), (3, 4))
        self.assertEqual(distance, 5.0)  # Expect distance to be 5 (3-4-5)

    def test_mutate_swap(self):
        """
        Test the mutate_swap method.
        Ensure that the mutation correctly swaps two cities in the route.
        """
        route = [0, 1, 2]
        # Temporarily increase mutation rate for testing
        self.ga.mutate_swap(route, 1.0)  # Force mutation
        self.assertTrue(route != [0, 1, 2])  # Route should change

    def test_one_point_crossover(self):
        """
        Test the one_point_crossover method.
        Ensure that the crossover creates a valid child route.
        """
        parent1 = [0, 1, 2]
        parent2 = [2, 0, 1]
        child = self.ga.one_point_crossover(parent1, parent2)
        self.assertEqual(len(child), len(parent1))  # Child route should have same length as parents

    def test_elitism_selection(self):
        """
        Test the elitism_selection method.
        Ensure that the best individuals are selected for the next generation.
        """
        # Define a population and fitness values
        population = [[0, 1, 2], [2, 0, 1], [1, 2, 0]]
        fitness = [10, 20, 15]  # Lower values are better

        # Select 1 elite individual
        elite = self.ga.elitism_selection(population, fitness, elite_size=1)
        self.assertEqual(elite, [[0, 1, 2]])  # Best individual should be [0, 1, 2]

        # Select 2 elite individuals
        elite = self.ga.elitism_selection(population, fitness, elite_size=2)
        self.assertEqual(elite, [[0, 1, 2], [1, 2, 0]])  # Best individuals in order


    def test_run(self):
        """
        Test the run method.
        Ensure that the genetic algorithm executes correctly and returns
        a valid route and distance after the specified number of generations.
        """
        best_route, best_distance, fitness = self.ga.run()
        self.assertIsInstance(best_route, list)  # Best route should be a list
        self.assertTrue(best_distance >= 0)  # Best distance should be non-negative

if __name__ == "__main__":
    unittest.main()
