import matplotlib.pyplot as plt

class Visualization:
    def __init__(self):
        """Initialize the Visualization class with storage for result sets."""
        self.results_sets = []

    def plot_route(self, route, distance_matrix):
        """Visualize the best route."""
        # Get coordinates for cities from the distance matrix
        x_coords = [distance_matrix[i][0] for i in route]
        y_coords = [distance_matrix[i][1] for i in route]

        # Add the first city at the end to close the loop
        x_coords.append(x_coords[0])
        y_coords.append(y_coords[0])

        plt.figure(figsize=(8, 6))
        plt.plot(x_coords, y_coords, 'bo-', label="Route")
        
        # Annotating cities
        for i, city in enumerate(route):
            plt.text(x_coords[i], y_coords[i], f"City {city}", fontsize=12, ha='right')

        plt.title("TSP Route Visualization")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()
        plt.grid(True)
        plt.show()

    def add_results(self, best_results, label=None):
        """
        Add a new list of best results for later visualization.
        
        Parameters:
        - best_results (list): List of the best results for one series.
        - label (str): Optional label for this series on the plot.
        """
        if label is None:
            label = f"Series {len(self.results_sets) + 1}"
        self.results_sets.append((best_results, label))

    def plot_best_results(self):
        """
        Generate the plot of all added result sets.
        """
        plt.figure(figsize=(10, 6))
        for results, label in self.results_sets:
            plt.plot(range(1, len(results) + 1), results, label=label)

        plt.title("Best Results Over Iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Best Result")
        plt.legend()
        plt.grid(True)
        plt.show()
