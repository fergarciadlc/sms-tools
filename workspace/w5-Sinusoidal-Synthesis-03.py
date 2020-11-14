import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../software/models'))
import utilFunctions as UF
from scipy.signal import blackmanharris, triang
from scipy.fftpack import ifft
"""
Generate 4kHz sine spectrum, and calculate its inverse fourier transform
"""
fs = 44100
Ns = 512
hNs = Ns/2
H = Ns/4
ipfreq = np.array([4000.0])
ipmag = np.array([0.0])
ipphase = np.array([0.0])
Y = UF.genSpecSines_p(ipfreq, ipmag, ipphase, Ns, fs)
y = np.real(ifft(Y))

# Undo the windowing
sw = np.zeros(Ns)
ow = triang(Ns/2)
sw[hNs-H : hNs+H] = ow
bh = blackmanharris(Ns)
bh = bh / sum(bh)
sw[hNs-H : hNs+H] = sw[hNs-H : hNs+H] / bh[hNs-H : hNs+H]

yw = np.zeros(Ns)
yw[:hNs-1] = y[hNs+1:]
yw[hNs-1:] = y[:hNs+1]
yw *= sw

plt.figure()
plt.plot(abs(Y))
plt.title("abs of spectrum")

plt.figure()
plt.plot(y)
plt.title("blackmanharris window sinusoid in time domain")

plt.figure()
plt.plot(yw)
plt.title("Undo windowing")

plt.show()