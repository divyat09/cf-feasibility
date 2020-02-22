#Important Classes
from dataloader import DataLoader

#Normie stuff
import sys
import pandas as pd
import numpy as np
import json

#Torch 
from torchvision import datasets, transforms

base_dir='../data/'
dataset_name=sys.argv[1]

if dataset_name == 'mnist':
   
    #Loading MNIST Data
#     dataset_size= 75000 #Consistent with the training size of CEM paper
# Proof of concept with small dataset
    dataset_size= 1000
    mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)
    dataset = (mnist.data[:dataset_size].numpy(), mnist.targets[:dataset_size].numpy())
    print(type(dataset[0]), dataset[0].shape, dataset[1].shape)
    print(np.unique(dataset[1], return_counts=True))
    images= dataset[0]
    labels= dataset[1]
    
    #Reshape images from the matrix to vector for concatenation with the labels
    images_old= images
    images= np.reshape(images, (images.shape[0], images.shape[1]*images.shape[2]))
    labels= np.reshape(labels, (labels.shape[0],1))
    
    #Normalization of pixels
    images= images/255.
    
    #Feature, Label concatenated dataset
    dataset= np.concatenate((images, labels), axis=1)
    print(dataset.shape)
    
    #Sanity Check
#     images_recon = np.reshape(images, (images.shape[0], 28, 28))
#     print(images_recon.shape, images_old.shape, images_recon==images_old)

#Train, Val, Test Splits
np.random.shuffle(dataset)
test_size= int(0.1*dataset.shape[0])
vae_test_dataset= dataset[:test_size]
dataset= dataset[test_size:]
vae_val_dataset= dataset[:test_size]
vae_train_dataset= dataset[test_size:]

# Saving datasets 
np.save(base_dir+dataset_name+'-'+'train-set', vae_train_dataset )
np.save(base_dir+dataset_name+'-'+'val-set', vae_val_dataset )
np.save(base_dir+dataset_name+'-'+'test-set', vae_test_dataset )