# Genetic Algorithm for the Traveling Salesman Problem (TSP)

This project implements a Genetic Algorithm to solve the Traveling Salesman Problem (TSP). The algorithm allows for testing different combinations of selection, crossover, and mutation methods and provides visualizations of results and routes. Additionally, it supports statistical analysis of performance.

## Features

- **Selection Methods**:
  - Tournament
  - Elitism
  - Steady State

- **Crossover Methods**:
  - One-point
  - Cycle
  - Order

- **Mutation Methods**:
  - Swap
  - Adjacent Swap
  - Inverse
  - Insertion

- **Visualization**:
  - Route plotting for the best solution.
  - Comparison of results for different methods.

- **Statistical Analysis**:
  - Outputs statistical summaries (mean, standard deviation, quartiles, max) for minimum results across repeated runs.

## Installation

1. Clone the repository:
   ```bash
   git clone git@github.com:mieszkoP13/genetic-algorithm-tsp.git
   cd tsp-genetic-algorithm
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the program using the CLI.

### Example: Testing mutation rates
```bash
python main.py -n 30 -p 20 -g 800 -m 0 0.1 1 -c 0.9 -s
```

### Example: Testing different selection, crossover, and mutation methods
```bash
python main.py -n 20 -p 20 -g 800 -m 0.1 -c 0.8 --selection-method each --crossover-method each --mutation-method insertion -r 10
```

### Help Page
```bash
python main.py -h
```
**Output:**
```
usage: main.py [-h] -n NUM_CITIES [NUM_CITIES ...] -p POPULATION_SIZE [POPULATION_SIZE ...] -g GENERATIONS [GENERATIONS ...] -m MUTATION_RATE [MUTATION_RATE ...] -c CROSSOVER_RATE [CROSSOVER_RATE ...]
               [--selection-method {tournament,elitism,steady_state,each}] [--crossover-method {one_point,cycle,order,each}] [--mutation-method {swap,adjacent_swap,inverse,insertion,each}] [-s] [-r REPEATS]

Genetic Algorithm for the Traveling Salesman Problem (TSP)

options:
  -h, --help            show this help message and exit
  -n NUM_CITIES [NUM_CITIES ...], --num-cities NUM_CITIES [NUM_CITIES ...]
                        Number of cities. Provide one value (e.g., 6) or three values for range testing (start, step, stop).
  -p POPULATION_SIZE [POPULATION_SIZE ...], --population-size POPULATION_SIZE [POPULATION_SIZE ...]
                        Population size. Provide one value (e.g., 50) or three values for range testing (start, step, stop).
  -g GENERATIONS [GENERATIONS ...], --generations GENERATIONS [GENERATIONS ...]
                        Number of generations. Provide one value (e.g., 100) or three values for range testing (start, step, stop).
  -m MUTATION_RATE [MUTATION_RATE ...], --mutation-rate MUTATION_RATE [MUTATION_RATE ...]
                        Mutation rate. Provide one value (e.g., 0.1) or three values for range testing (start, step, stop).
  -c CROSSOVER_RATE [CROSSOVER_RATE ...], --crossover-rate CROSSOVER_RATE [CROSSOVER_RATE ...]
                        Crossover rate. Provide one value (e.g., 0.8) or three values for range testing (start, step, stop).
  --selection-method {tournament,elitism,steady_state,each}
                        Method of selection: 'tournament' (default), 'elitism', or 'each' to iterate through all.
  --crossover-method {one_point,cycle,order,each}
                        Method of crossover: 'one_point' (default), 'cycle', 'order', or 'each' to iterate through all.
  --mutation-method {swap,adjacent_swap,inverse,insertion,each}
                        Method of mutation: 'swap' (default), 'inverse', or 'each' to iterate through all.
  -s, --fixed-seed      Use a fixed seed (42) for random number generation to ensure reproducibility.
  -r REPEATS, --repeats REPEATS
                        Number of times to repeat the experiment for averaging. Default is 1.
```
