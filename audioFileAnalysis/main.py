import librosa

audio_path = "./T08-violin.wav"

x, sr = librosa.load(audio_path)

import matplotlib.pyplot as plt
import librosa.display

plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)
# plt.show()

X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis="time", y_axis="log")
plt.colorbar()
# plt.show()

import soundfile as sf
sf.write("./example.wav", x, sr)
