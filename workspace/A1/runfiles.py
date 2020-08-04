import numpy as np
import A1Part1, A1Part2, A1Part3, A1Part4

print("A1Part1:")
print(A1Part1.readAudio('../../sounds/piano.wav'))

print("A1Part2:")
print(A1Part2.minMaxAudio('../../sounds/oboe-A4.wav'))

print("A1Part3:")
x = np.arange(10)
M = 2
print(A1Part3.hopSamples(x,M))
