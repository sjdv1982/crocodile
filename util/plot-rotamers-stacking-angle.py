import numpy as np
from math import pi
import sys
from matplotlib import pyplot as plt
data = np.loadtxt(sys.argv[1])
outfile = sys.argv[2]
plt.hist(data/pi * 180, bins=100, cumulative=True)
plt.savefig(outfile)