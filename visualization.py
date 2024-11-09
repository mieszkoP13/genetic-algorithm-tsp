import matplotlib.pyplot as plt

def plot_route(route, distance_matrix):
    """Visualize the best route."""
    # Get coordinates for cities from the distance matrix
    x_coords = [distance_matrix[i][0] for i in route]
    y_coords = [distance_matrix[i][1] for i in route]

    # Add the first city at the end to close the loop
    x_coords.append(x_coords[0])
    y_coords.append(y_coords[0])

    plt.figure(figsize=(8, 6))
    plt.plot(x_coords, y_coords, 'bo-', label="Route")
    plt.scatter(x_coords, y_coords, c='red', marker='o')
    
    # Annotating cities
    for i, city in enumerate(route):
        plt.text(x_coords[i], y_coords[i], f"City {city}", fontsize=12, ha='right')

    plt.title("TSP Route Visualization")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)
    plt.show()
