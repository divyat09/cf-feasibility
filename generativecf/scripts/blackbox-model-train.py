#Important Classes
from dataloader import DataLoader
from blackboxmodel import BlackBox
from helpers import *

#Normie stuff
import sys
import pandas as pd
import numpy as np
import json

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

base_data_dir='../data/'
base_model_dir='../models/'
dataset_name=sys.argv[1]

#Dataset
if dataset_name=='bn1':
    dataset= pd.read_csv(base_data_dir+dataset_name+'.csv')
    dataset.drop(['Unnamed: 0'], axis=1, inplace=True)	

    params= {'dataframe':dataset.copy(), 'continuous_features':['x1','x2','x3'], 'outcome_name':'y'}
    d = DataLoader(params)

    inp_shape= len(d.encoded_feature_names)
    pred_model= BlackBox(inp_shape)
    learning_rate = 0.01
    # Default Batch Size of Keras
    batch_size = 32
    optimizer = optim.Adam([
        {'params': filter(lambda p: p.requires_grad, pred_model.predict_net.parameters()) }
    ], lr=learning_rate)
    crieterion= nn.CrossEntropyLoss()
    

elif dataset_name=='adult':
    dataset = load_adult_income_dataset()

    params= {'dataframe':dataset.copy(), 'continuous_features':['age','hours_per_week'], 'outcome_name':'income'}
    d = DataLoader(params)

    inp_shape= len(d.encoded_feature_names)
    pred_model= BlackBox(inp_shape)
    learning_rate = 0.01
    # Default Batch Size of Keras
    batch_size = 32
    optimizer = optim.Adam([
        {'params': filter(lambda p: p.requires_grad, pred_model.predict_net.parameters()) }
    ], lr=learning_rate)
    crieterion= nn.CrossEntropyLoss()


elif dataset_name=='sangiovese':
    dataset = pd.read_csv(  base_data_dir + dataset_name + '.csv', index_col=None )
    dataset= dataset.drop(columns= ['Unnamed: 0'])
    outcome=[]
    for i in range(dataset.shape[0]):
        if dataset['GrapeW'][i] > 0: 
            outcome.append( 1 )
        else:
            outcome.append( 0 )
    dataset['outcome'] = pd.Series(outcome)
    dataset.drop(columns=['GrapeW'], axis=1, inplace=True)

    # Continuous Features
    l=list(dataset.columns)
    l.remove('outcome')
    params= {'dataframe':dataset.copy(), 'continuous_features':l, 'outcome_name':'outcome'}
    d = DataLoader(params)    

    inp_shape= len(d.encoded_feature_names)
    pred_model= BlackBox(inp_shape)
    learning_rate = 0.01
    # Default Batch Size of Keras
    batch_size = 32
    optimizer = optim.Adam([
        {'params': filter(lambda p: p.requires_grad, pred_model.predict_net.parameters()) }
    ], lr=learning_rate)
    crieterion= nn.CrossEntropyLoss()


#Train/Val dataset
train_dataset= np.load(base_data_dir+dataset_name+'-train-set.npy')
print(train_dataset.shape)
np.random.shuffle(train_dataset)

validation_dataset= np.load(base_data_dir+dataset_name+'-val-set.npy')
print(validation_dataset.shape)

#Training
for epoch in range(100):
    np.random.shuffle(train_dataset)
    train_batches= np.array_split( train_dataset, train_dataset.shape[0]//batch_size ,axis=0 )    
    print('Epoch: ', epoch)
    train_acc=0.0
    for i in range(len(train_batches)):    
        optimizer.zero_grad()
        train_x= torch.tensor( train_batches[i][:,:-1] ).float() 
        train_y= torch.tensor( train_batches[i][:,-1], dtype=torch.int64 )
        out= pred_model(train_x)
        train_acc += torch.sum( torch.argmax(out, axis=1) == train_y )
        # Cross Entropy
        loss= crieterion(out, train_y) 
        loss.backward()
        optimizer.step()
    print(train_acc, len(train_dataset))

# Validation        
np.random.shuffle(validation_dataset)
train_batches= np.array_split( validation_dataset, validation_dataset.shape[0]//batch_size ,axis=0 )    
val_acc=0.0
for i in range(len(train_batches)):    
    optimizer.zero_grad()
    train_x= torch.tensor( train_batches[i][:,:-1] ).float() 
    train_y= torch.tensor( train_batches[i][:,-1], dtype=torch.int64 )
    out= pred_model(train_x)
    val_acc += torch.sum( torch.argmax(out, axis=1) == train_y )
print(val_acc, len(validation_dataset))	

#Saving the Black Box Model
path=base_model_dir+dataset_name+'.pth'
torch.save(pred_model.state_dict(), path)        