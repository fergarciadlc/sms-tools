import numpy as np
"""
This script generates a test signal of two signusoids of different frequency sampled
by a given fs
"""
fs = 48000.0
f1 = 300.0
f2 = 800.0

# 1 second of signal
t = np.linspace(0, 1, int(fs))

x = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)
