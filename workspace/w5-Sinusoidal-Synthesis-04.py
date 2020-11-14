import numpy as np
import matplotlib.pyplot as plt
import sys, os
from scipy.signal import get_window
from scipy.signal import blackmanharris, triang
from scipy.fftpack import ifft
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../software/models'))
import utilFunctions as UF
import dftModel as DFT

"""
Real-world sound applied to this analysis
"""
fs, x = UF.wavread("../sounds/oboe-A4.wav")
Ns = 512 # FFT size
hNs = Ns/2
H = Ns/4
M = 511
t = -70
w = get_window("hamming", M)

x1 = x[int(.8*fs) : int(.8*fs)+M]
mX, pX = DFT.dftAnal(x1, w, Ns)
ploc = UF.peakDetection(mX, t)
iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc) #interpolated
ipfreq = fs*iploc / float(Ns)

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

freqaxis = fs * np.arange(Ns/2+1)/float(Ns)

plt.figure()
plt.plot(freqaxis, mX)
plt.plot(fs * iploc / float(Ns), ipmag, marker='x', linestyle='')
plt.title("Magnitude spectrum")


plt.figure()
plt.plot(y)
plt.title("Zero centered Inverse spectrum signal")


plt.figure()
plt.plot(yw, label="Undo windowing and triangulared signal", lw=2)
plt.plot(x[int(.8*fs) : int(.8*fs)+M], label="Original signal", lw=1)
plt.legend()
plt.tight_layout()

plt.show()
