import numpy as np
import matplotlib.pyplot as plt
import scipy
import seaborn as sns

sns.set(style="darkgrid")

# Generate a chirp signal
np.random.seed(0)
Fs = 120
time_step = 1 / Fs
time_vec = np.arange(0, 70, time_step)

sig = np.sin(0.5 * np.pi * time_vec * (1 + 0.1 * time_vec))

plt.figure(figsize=(14, 4))
plt.plot(time_vec, sig)
plt.show()

# Plotting the spectrum
n = len(sig)
k = np.arange(n)
T = n / Fs
freq = k / T
freq = freq[range(n // 2)]
Y = np.fft.fft(sig) / n
Y = Y[range(n // 2)]
plt.plot(freq, abs(Y), "r")
plt.xlabel("FREQ(HZ)")
plt.ylabel("Y(FREQ)")
plt.show()

# Compute and plot the spectogram
from scipy import signal

freqs, times, spectogram = signal.spectrogram(sig)
plt.imshow(spectogram, aspect="auto", cmap="hot_r", origin="lower")
plt.title("spectogram")
plt.xlabel("time window")
plt.ylabel("frequency band")
plt.tight_layout()
plt.show()

# Compute and plot the power spectral density
freqs, psd = signal.welch(sig)
plt.semilogx(freqs, psd)
plt.xlabel("Frequency")
plt.ylabel("Power")
plt.tight_layout()
plt.show()
