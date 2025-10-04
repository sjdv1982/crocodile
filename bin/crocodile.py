import sys, os

# Remove current path from imports
currdir = os.path.realpath(os.path.dirname(sys.argv[0]))
sys.path = [d for d in sys.path if os.path.realpath(d) != currdir]

from crocodile.main import main
main()