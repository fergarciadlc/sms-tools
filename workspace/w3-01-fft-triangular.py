import numpy as np
from scipy.signal import triang
from scipy.fftpack import fft

x = triang(15) # 15 samples
X = fft(x)  # FFT algorith size of x, 15
mX = abs(X)
pX = np.angle(X)

"""
run ipython --pylab to run terminal with matplotlib loaded
and type:
    >> run script
"""
