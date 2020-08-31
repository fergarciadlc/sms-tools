import numpy as np
from scipy.signal import get_window
from scipy.fftpack import fft
import math
import matplotlib.pyplot as plt

windows = ['hanning', 'hamming', 'blackman', 'blackmanharris']

for win in windows:	

	M = 62 # Window Size
	window = get_window(win, M)
	# Middle of the window
	hM1 = int(math.floor((M+1)/2)) 
	hM2 = int(math.floor(M/2))

	# Preparing the window for the fft
	N = 512
	hN = N/2 #+ 1 #,,, why +1 ?
	fftbuffer = np.zeros(N)
	# Place window arround zero:
	# second half of the window at the beggining and the fist half at the end
	fftbuffer[:hM1] = window[hM2:]
	fftbuffer[N-hM2:] = window[:hM2]

	# Compute Spectrum of Buffer
	X = fft(fftbuffer)
	absX = abs(X)
	absX[absX < np.finfo(float).eps] = np.finfo(float).eps # ensure not zeros (-inf)
	mX = 20 * np.log10(absX)
	pX = np.angle(X)

	"""
	in order to show/display better the magnitude phase spectrum and
	see it in the middle of the array we undo the zero phase windowing thing,
	so that we place back the data in the middle of the array, easier to visualize
	"""
	mX1 = np.zeros(N)
	pX1 = np.zeros(N)
	mX1[:hN] = mX[hN:]
	mX1[N-hN:] = mX[:hN]
	pX1[:hN] = pX[hN:]
	pX1[N-hN:] = pX[:hN]

	plt.plot(np.arange(-hN, hN) / float(N)*M, mX1 - max(mX1))
	#plt.axis([-20, 20, -80, 0])

plt.axis([-20, 20, -120, 0])
plt.title('Windows spectrum')
plt.legend(windows)
plt.xlabel('bins')
plt.ylabel('Magnitude (dB)')

plt.show()