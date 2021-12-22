import os 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib
import librosa.display
from tensorflow.python.client import session
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import librosa

data_dir = "./data"
if(os.listdir(data_dir) == 0):
    raise Exception("Wrong data input")

def get_wav_paths(speaker):
    speaker_path = data_dir + "/" + speaker
    all_paths = [item for item in os.listdir(speaker_path)]

    return all_paths

benjamin_netanyau = get_wav_paths("Benjamin_Netanyau")


def load_wav(wav_path,speaker):
    with tf.compat.v1.Session(graph=tf.compat.v1.Graph()) as sess:
        wav_path = data_dir  + "/" +  speaker + "/" + wav_path
        wav_filename_placeholder = tf.compat.v1.placeholder(tf.compat.v1.string,{})
        wav_loader = tf.io.read_file(wav_filename_placeholder)
        wav_decoder = tf.audio.decode_wav(wav_loader, desired_channels=1)
        wav_data = sess.run(wav_decoder, feed_dict={
            wav_filename_placeholder: wav_path
        }).audio.flatten().reshape(1,16000)
        sess.close

    return wav_data

def generate_training_data(speaker_paths, speaker, label):
    wavs, labels = [],[]

    for i in tqdm(speaker_paths):
        wav = load_wav(i,speaker)
        wavs.append(wav)
        labels.append(label)

    return wavs,labels

benjamin_netanyau_wavs, benjamin_netanyau_labels = generate_training_data(benjamin_netanyau,"Benjamin_Netanyau",0)

train_wavs,test_wavs, train_labels, test_labels = train_test_split(benjamin_netanyau_wavs,benjamin_netanyau_labels, test_size=0.1)

train_x,train_y = np.array(train_wavs),np.array(train_labels)
test_x,test_y = np.array(test_wavs),np.array(test_labels)
train_y = tf.keras.utils.to_categorical(train_y)
test_y = tf.keras.utils.to_categorical(test_y)

# MFCC feature extraction
train_x_new = []
test_x_new = []
INPUT_SHAPE = (126,40)

train_x_new = np.zeros((train_x.shape[0], INPUT_SHAPE[0], INPUT_SHAPE[1]), dtype=np.float64)

count = 0
for sample in train_x:
    mfcc = librosa.feature.mfcc(y=sample, sr=16000, hop_length=128, n_fft=256, n_mfcc=20)
    mfcc_delta = librosa.feature.delta(mfcc)[:10, :]
    mfcc_double_delta = librosa.feature.delta(mfcc, order=2)[:10, :]
    train_x_new[count, :, :20] = mfcc.T
    train_x_new[count, :, 20:30] = mfcc_delta.T
    train_x_new[count, :, 30:] = mfcc_double_delta.T
    count += 1
    if count%500 == 0:
        print('Train', count)
        
test_x_new = np.zeros((test_x.shape[0], INPUT_SHAPE[0], INPUT_SHAPE[1]), dtype=np.float64)

count = 0
for sample in test_x:
    mfcc = librosa.feature.mfcc(y=sample, sr=16000, hop_length=128, n_fft=256, n_mfcc=20)
    mfcc_delta = librosa.feature.delta(mfcc)[:10, :]
    mfcc_double_delta = librosa.feature.delta(mfcc, order=2)[:10, :]
    test_x_new[count, :, :20] = mfcc.T
    test_x_new[count, :, 20:30] = mfcc_delta.T
    test_x_new[count, :, 30:] = mfcc_double_delta.T
    count += 1
    if count%500 == 0:
        print('Test', count)

test_x_new = np.zeros(test_x.shape[0],INPUT_SHAPE[0],INPUT_SHAPE[1],dtype=np.float64)

count = 0

for sample in test_x:
    mfcc = librosa.feature.mfcc(y = sample,sr = 1600, hop_length = 128,n_fft = 256, n_mfcc=20)
    mfcc_delta = librosa.feature.delta(mfcc)[:10,:]
    mfcc_double_delta = librosa.feature.delta(mfcc,order = 2)[:10,:]
    test_x_new[count :, : 20] = mfcc.T
    test_x_new[count :, 20: 30] = mfcc_delta.T
    test_x_new[count :, 30:] = mfcc_double_delta.T
    count += 1
    
    if count % 500 == 0:
        print("Test ", count)

train_x_new = np.expand_dims(train_x_new,axis = 3)
test_x_new = np.expand_dims(test_x_new,axis = 3)

# create a model
def create_model(speech_feature):
    model = tf.keras.Sequential()
    if speech_feature == "spectrogram":
        model.add(Spectrogram(n_dft=512, n_hop=256, input_shape=(1, 16000),
                            return_decibel_spectrogram=True, power_spectrogram=2.0,
                            trainable_kernel=False, name='static_stft'))
    elif speech_feature == "melspectrogram":
        model.add(Melspectrogram(sr=16000, n_mels=128,n_dft=512, n_hop=256,
                            input_shape=(1 , 16000),return_decibel_melgram=True,
                            trainable_kernel=False, name='melgram'))
        
    elif speech_feature == "mfcc":
        model.add(tf.keras.layers.Conv2D(128, (3, 3), activation="relu", input_shape=(126,40,1)))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
#         model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.Flatten())        
        model.add(tf.keras.layers.Dense(5, activation="softmax"))
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=3e-4)
                , loss = "categorical_crossentropy"
                , metrics = ["accuracy"])
        return model

    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(5, activation="softmax"))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=3e-4)
            , loss = "categorical_crossentropy"
            , metrics = ["accuracy"])
    return model

model = create_model("mfcc")
model.summary()

model.fit(x = train_x_new,y = train_y,epochs=5,validation_data=(test_x_new,test_y))

y_pred = model.predict(test_x_new)