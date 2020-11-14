import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../software/models'))
import dftModel as DFT
import utilFunctions as UF

(fs, x) = UF.wavread('../sounds/sine-440.wav')
M = 501
N = 2048 # FFT Size Try different sizes of FFT, the resolution of the spectrum can be calculated by: Fs / N (FFT size)
t = -20
w = get_window('hamming', M)
x1 = x[int(.8*fs) : int(.8*fs)+M]
mX, pX = DFT.dftAnal(x1, w, N)
ploc = UF.peakDetection(mX, t)
iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc) #interpolated
pmag = mX[ploc]

freqaxis = fs * np.arange(N/2+1)/float(N)
plt.figure()
plt.plot(freqaxis, mX)
plt.plot(fs * ploc / float(N), pmag, marker='x', linestyle='')


plt.figure()
plt.plot(freqaxis, mX)
plt.plot(fs * iploc / float(N), ipmag, marker='x', linestyle='')


plt.show()
