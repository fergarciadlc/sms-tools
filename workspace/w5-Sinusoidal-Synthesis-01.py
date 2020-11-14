import numpy as np
import sys, os
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../software/models'))
import utilFunctions as UF

bins = np.array([-4, -3, -2, -1, 0, 1, 2, 3]) + .5
X = UF.genBhLobe(bins)

plt.figure()
plt.plot(X)

plt.figure()
plt.plot(20*np.log10(X))

plt.show()
