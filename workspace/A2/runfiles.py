import numpy as np

import A2Part1, A2Part2, A2Part3, A2Part4, A2Part5

A = 1.0
f = 10.0
phi = 1.0
fs = 50.0
t = 0.1
print("genSine")
print(A2Part1.genSine(A, f, phi, fs, t))
print("")

N = 5
k = 1
print("genComplexSine")
print(A2Part2.genComplexSine(k, N))
print("")

x = np.array([1, 2, 3, 4])
print("DFT")
print(A2Part3.DFT(x))
print("")

X = np.array([1, 1, 1, 1])
print("IDFT")
print(A2Part4.IDFT(X))
print("")

print("genMagSpec")
print(A2Part5.genMagSpec(x))
print("")