import sys
from genetic_algorithm.tsp_cli import TSPCLI
from simulated_annealing.tsp_sa_cli import TSP_SA_CLI

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py [ga|sa]")
        sys.exit(1)

    mode = sys.argv[1]
    sys.argv = sys.argv[1:] # Pass left args to cli

    if mode == "ga":
        cli = TSPCLI()
    elif mode == "sa":
        cli = TSP_SA_CLI()
    else:
        print("Invalid mode. Use 'ga' or 'sa'.")
        sys.exit(1)

    cli.run()
