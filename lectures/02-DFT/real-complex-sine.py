"""
This script computes a real sine wave of the form:
x[n] = Acos(2 pi f n T + phi)
"""
import matplotlib.pyplot as plt
import numpy as np

A = 0.8
f0 = 1000
phi = np.pi / 2
fs = 44100

t = np.arange(-0.002, 0.002, 1.0 / fs)

x = A * np.cos(2 * np.pi * f0 * t + phi)

plt.plot(t, x)
plt.axis([-.002, .002, -.8, .8])
plt.title('Real Sine Wave')
plt.xlabel('time')
plt.ylabel('amplitude')

plt.show()

"""
This script computes a complex sine wave of the form:
s_k[n] = exp[j 2 pi k n / N] = cos(2 pi k n / N) +j sin(2 pi k n / N)
"""
N = 500
k = 3
n = np.arange(-N / 2, N / 2)
s = np.exp(1j * 2 * np.pi * k * n / N)

plt.plot(n, np.real(s))
plt.axis([-N / 2, N / 2 - 1, -1, 1])
plt.title('Complex Sine Wave')
plt.xlabel('n')
plt.ylabel('amplitude')

plt.show()