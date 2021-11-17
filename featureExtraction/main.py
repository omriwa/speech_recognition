from librosa.core.audio import zero_crossings
from librosa.feature.spectral import spectral_rolloff
import numba
import librosa
import numpy as np
import sklearn
import matplotlib.pyplot as plt

audio_path = "./featureExtraction/example_music.wav"
x, sr = librosa.load(audio_path)

from IPython.display import Audio
from IPython.core.display import display
import librosa.display

plt.figure(figsize=(12, 4))
# librosa.display.waveplot(x, sr)
# plt.show()

X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
# librosa.display.specshow(Xdb, sr=sr)
# plt.show()

n0 = 9000
n1 = 9100
# plt.plot(x[n0:n1])
# plt.grid()
# plt.show()

zero_crossings = librosa.zero_crossings(x[n0:n1], pad=False)
# print("zero_crossings ", sum(zero_crossings))

spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)


def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)


# librosa.display.waveplot(x, sr=sr, alpha=0.4)
# plt.plot(t, normalize(spectral_centroids), color="r")
# plt.show()

spectral_rolloff = librosa.feature.spectral_rolloff(x, sr=sr)[0]
# librosa.display.waveplot(x, sr=sr, alpha=0.4)
# plt.plot(t, normalize(spectral_rolloff), color="r")
# plt.show()

mfcc = librosa.feature.mfcc(x, sr=sr)
print(mfcc.shape)
librosa.display.specshow(mfcc, sr=sr, x_axis="time")
plt.show()
