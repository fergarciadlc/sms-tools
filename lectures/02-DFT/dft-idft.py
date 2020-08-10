"""
This script computes the discrete Fourier Transform:
X[k] = \sum_{n=0}^{N-1} x[n] exp[-j 2 pi k n / N] ; k = 0, ..., N-1
"""
import matplotlib.pyplot as plt
import numpy as np

"""
Complex signal:
    N = 64
    k0 = 7
    x = np.exp(1j * 2 * np.pi * k0 / N * np.arange(N))
    ...
    plt.plot(np.arange(N), abs(X))
    plt.axis([0, N-1, 0, N])
Real signal:
    N = 64
    k0 = 7
    x = np.cos(2 * np.pi * k0 / N * np.arange(N))
    ...
        
"""
N = 32
k0 = 7  # Frequency
x = np.cos(2 * np.pi * k0 / N * np.arange(N))  # REAL
# x = np.exp(1j * 2 * np.pi * k0 / N * np.arange(N)) # COMPLEX

X = np.array([])
nv = np.arange(-N / 2, N / 2)  # Time index for real signals
kv = np.arange(-N / 2, N / 2)  # Frequency indexes

# For complex signals:
# for k in range(N):
#     s = np.exp(1j * 2 * np.pi * k / N * np.arange(N))
#     X = np.append(X, sum(x * np.conjugate(s)))

for k in kv:
    s = np.exp(1j * 2 * np.pi * k / N * nv)
    X = np.append(X, sum(x * np.conjugate(s)))

plt.plot(kv, abs(X))
plt.axis([-N / 2, N / 2 - 1, 0, N])
plt.title('DFT')
plt.xlabel('N')
plt.ylabel('magnitude')

plt.show()

y = np.array([])
for n in nv:
    s = np.exp(1j * 2 * np.pi * n / N * kv)
    y = np.append(y, 1.0/N * sum(X*s))
plt.plot(kv, y)
plt.axis([-N/2, N/2 - 1, -1, 1])
plt.title('IDFT')
plt.xlabel('N')
plt.ylabel('magnitude')

plt.show()
