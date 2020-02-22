#Important Classes
from dataloader import DataLoader
from vae_model import AutoEncoder
from blackboxmodel import BlackBox
from helpers import *

#Normie stuff
import sys
import random
import pandas as pd
import numpy as np
import json
import argparse

#Pytorch
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable

#Seed for repoduability
torch.manual_seed(10000000)

# Tensorflow libraries
import tensorflow as tf
from tensorflow import keras
#Keras
from tensorflow.keras.layers import Conv2D, Input, Dense, Lambda, Layer, Add, Multiply, Dropout, Flatten, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras import backend as K

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print('x_train shape:', x_train.shape, 'y_train shape:', y_train.shape)

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = np.reshape(x_train, x_train.shape + (1,))
x_test = np.reshape(x_test, x_test.shape + (1,))
print('x_train shape:', x_train.shape, 'x_test shape:', x_test.shape)
#y_train = to_categorical(y_train)
#y_test = to_categorical(y_test)
print('y_train shape:', y_train.shape, 'y_test shape:', y_test.shape)

def ae_model():
    # encoder
    x_in = Input(shape=(28, 28, 1))
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x_in)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    encoded = Conv2D(1, (3, 3), activation=None, padding='same')(x)
    encoder = Model(x_in, encoded)
    
    # decoder
    dec_in = Input(shape=(14, 14, 1))
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(dec_in)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    decoded = Conv2D(1, (3, 3), activation=None, padding='same')(x)
    decoder = Model(dec_in, decoded)
    
    # autoencoder = encoder + decoder
    x_out = decoder(encoder(x_in))
    autoencoder = Model(x_in, x_out)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder, encoder, decoder

# Training Target Auto Encoder for all the classes
for t_c in range(10):
    ae, enc, dec = ae_model()
    
    y_train_tc= y_train[y_train==t_c]
    x_train_tc= x_train[ y_train == t_c ]
    print('Training for Target Class: ', t_c)
    print('Data Size: ', x_train_tc.shape)
    print('Sanity Check: ', np.unique(y_train_tc, return_counts=True))
    
    ae.fit(x_train, x_train, batch_size=128, epochs=4, validation_data=(x_test, x_test), verbose=0)
    
    path= '../models/mnist_ae_' + 'target_class_' + str(t_c)
    model_json = ae.to_json()
    with open(path+".json", "w") as json_file:
        json_file.write(model_json)
    ae.save_weights(path+".h5")        
    
# For the whole data trained AE    
ae, enc, dec = ae_model()

y_train_tc= y_train
x_train_tc= x_train
print('Data Size: ', x_train_tc.shape)
print('Sanity Check: ', np.unique(y_train_tc, return_counts=True))

ae.fit(x_train, x_train, batch_size=128, epochs=4, validation_data=(x_test, x_test), verbose=0)

path= '../models/mnist_ae_' + 'target_class_' + str(-1)
model_json = ae.to_json()
with open(path+".json", "w") as json_file:
    json_file.write(model_json)
ae.save_weights(path+".h5")            