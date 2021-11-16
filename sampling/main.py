from numpy import linspace, cos, pi, ceil, floor, arange
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

plt.figure(figsize=(18, 6))
f = 40
tmin = -0.3
tmax = 0.3
t = linspace(tmin, tmax, 400)
x = cos(2 * pi) + cos(2 * pi * f * t)

plt.plot(t, x, label="Signal sampling 40 Hz")
plt.show()

T = 1 / 80
nmin = ceil(tmin / T)
nmax = floor(tmax / T)
n = arange(nmin, nmax)
x1 = cos(2 * pi * n * T) + cos(2 * pi * f * T)

plt.plot(n * T, x1, "bo", label="Signal sampling 80 Hz")
plt.show()
