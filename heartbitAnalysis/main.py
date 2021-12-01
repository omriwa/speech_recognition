import librosa

audio_path = "./heart_beat_sound.wav"
x, sr = librosa.load(audio_path)

import matplotlib.pyplot as plt
import librosa.display

plt.figure(figsize=(12, 5))
librosa.display.waveplot(x, sr=sr)
# plt.show()

n0 = 9000
n1 = 9100
plt.plot(x[n0:n1])
plt.grid()
# plt.show()
