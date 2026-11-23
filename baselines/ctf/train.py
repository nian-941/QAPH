"""Run CTF through its public command-line interface."""
import sys
from baselines.train import main

if __name__ == "__main__":
    sys.argv[1:1]=["--method", "ctf"]
    main()
