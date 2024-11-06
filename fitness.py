def calculate_route_distance(route, distance_matrix):
    """Calculate the total distance of the given route using the distance matrix."""
    total_distance = 0
    num_cities = len(route)
    
    for i in range(num_cities):
        from_city = route[i]
        to_city = route[(i + 1) % num_cities]  # Wrap around to the first city
        total_distance += distance_matrix[from_city][to_city]
    
    return total_distance
