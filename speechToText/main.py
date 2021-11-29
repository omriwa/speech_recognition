import os
import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import shape
from scipy.io import wavfile
import warnings

warnings.filterwarnings("ignore")

train_audio_path = "./data"
samples, sample_rate = librosa.load(
    train_audio_path + "/yes/00f0204f_nohash_0.wav", sr=16000
)
fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_subplot(211)

ax1.set_title("Raw wave of " + train_audio_path + "/yes/0a7c2a8d_nohash_0.wav")
ax1.set_xlabel("time")
ax1.set_ylabel("Amplitude")
ax1.plot(np.linspace(0, sample_rate / len(samples), sample_rate), samples)

plt.show()

# Sampling rate
ipd.Audio(samples, rate=sample_rate)

# Resampling
samples = librosa.resample(samples, sample_rate, 8000)
ipd.Audio(samples, rate=8000)

# Number of recording of each voices
labels = [
    dir
    for dir in os.listdir(train_audio_path)
    if os.path.isdir(train_audio_path + "/" + dir)
]
no_of_recordings = []

for label in labels:
    waves = [
        f for f in os.listdir(train_audio_path + "/" + label) if f.endswith(".wav")
    ]
    no_of_recordings.append(len(waves))

# Plot
index = np.arange(len(labels))
plt.figure(figsize=(15, 7))
plt.bar(index, no_of_recordings)
plt.xlabel("Commands", fontsize=12)
plt.ylabel("Recordings", fontsize=12)
plt.xticks(index, labels, fontsize=15, rotation=60)
plt.title("No. of recordings for each command")

plt.show()

# Preprocessing
all_wave = []
all_label = []

for label in labels:
    waves = [
        f for f in os.listdir(train_audio_path + "/" + label) if f.endswith(".wav")
    ]

    for wav in waves:
        samples, sample_rate = librosa.load(
            train_audio_path + "/" + label + "/" + wav, sr=16000
        )
        samples = librosa.resample(samples, sample_rate, 8000)
        if len(samples) == 8000:
            all_wave.append(samples)
            all_label.append(label)

# Split into train and validation set
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(all_label)
classes = list(le.classes_)

from keras.utils import np_utils

y = np_utils.to_categorical(y, num_classes=len(labels))
all_waves = np.array(all_wave).reshape(-1, 8000, 1)

from sklearn.model_selection import train_test_split

x_tr, x_val, y_tr, y_val = train_test_split(
    np.array(all_wave), np.array(y), stratify=y, test_size=0.2, random_state=7
)

# Building the model
from keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPool1D
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

K.clear_session()
inputs = Input(shape(8000, 1))

# First layer
conv = Conv1D(8, 13, padding="valid", activation="relu", strides=1)(inputs)
conv = MaxPool1D(3)(conv)
conv = Dropout(0.3)(conv)

# Second layer
conv = Conv1D(16, 11, padding="valid", activation="relu", strides=1)(inputs)
conv = MaxPool1D(3)(conv)
conv = Dropout(0.3)(conv)

# Third layer
conv = Conv1D(32, 9, padding="valid", activation="relu", strides=1)(inputs)
conv = MaxPool1D(3)(conv)
conv = Dropout(0.3)(conv)

# Fourth layer
conv = Conv1D(64, 7, padding="valid", activation="relu", strides=1)(inputs)
conv = MaxPool1D(3)(conv)
conv = Dropout(0.3)(conv)

# Flatten layer
conv = Flatten()(conv)

# Desnse layer 1
conv = Dense(256, activation="relu")(conv)
conv = Dropout(0.3)(conv)

# Desnse layer 2
conv = Dense(128, activation="relu")(conv)
conv = Dropout(0.3)(conv)

outputs = Dense(len(labels), activation="softmax")(conv)

model = Model(inputs, outputs)

model.summery()
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
es = EarlyStopping(
    monitor="val_loss", mode="min", verbose=1, patience=10, min_delta=0.0001
)
mc = ModelCheckpoint(
    "best_model.hdf5", monitor="val_acc", verbose=1, save_best_only=True, mode="max"
)
history = model.fit(
    x_tr,
    y_tr,
    epochs=100,
    callbacks=[es, mc],
    batch_size=32,
    validation_data=(x_val, y_val),
)

# Plotting model prediction
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="test")
plt.legend()
plt.show()
