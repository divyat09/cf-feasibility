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


def traverse(train_dataset, epochs=1, batch_size=2048):
    batch_num=0
    loss=0.0
    train_size=0
    train_dataset= np.array_split( train_dataset, train_dataset.shape[0]//batch_size ,axis=0 )
    for i in range(len(train_dataset)):
        train_x = train_dataset[i]
        train_x= torch.tensor( train_x ).float() 
        train_y = torch.argmax( pred_model(train_x), dim=1 )
        train_size += train_x.shape[0]
        print(np.unique(train_y, return_counts=True))
    print(train_size)

def ae_compute_loss( model, model_out, x, normalise_weights ): 
    
    em = model_out['em']
    ev = model_out['ev']
    z  = model_out['z']
    dm = model_out['x_pred']
    mc_samples = model_out['mc_samples']
            
    #KL Divergence
    kl_divergence = 0.5*torch.mean( em**2 +ev - torch.log(ev) - 1, axis=1 ) 
    
    #Reconstruction Term
    #Proximity: L1 Loss
    x_pred = dm[0] 
    s= model.encoded_start_cat
    recon_err = -torch.sum( torch.abs(x[:,s:-1] - x_pred[:,s:-1]), axis=1 )
    for key in normalise_weights.keys():
        # recon_err+= -(1/mad_feature_weights[d.encoded_feature_names[int(key)]])*(normalise_weights[key][1] - normalise_weights[key][0])*torch.abs(x[:,key] - x_pred[:,key]) 
        recon_err+= -(normalise_weights[key][1] - normalise_weights[key][0])*torch.abs(x[:,key] - x_pred[:,key]) 

    # Sum to 1 over the categorical indexes of a feature
    for v in model.encoded_categorical_feature_indexes:
        temp = -torch.abs(  1.0-torch.sum( x_pred[:, v[0]:v[-1]+1], axis=1) )
        recon_err += temp
    
    count=0
    count+= torch.sum(x_pred[:,:s]<0,axis=1).float()
    count+= torch.sum(x_pred[:,:s]>1,axis=1).float()    
            
    for i in range(1,mc_samples):
        x_pred = dm[i]        

        recon_err += -torch.sum( torch.abs(x[:,s:-1] - x_pred[:,s:-1]), axis=1 )
        for key in normalise_weights.keys():
            # recon_err+= -(1/mad_feature_weights[d.encoded_feature_names[int(key)]])*(normalise_weights[key][1] - normalise_weights[key][0])*torch.abs( (x[:,key] - x_pred[:,key]))
            recon_err+= -(normalise_weights[key][1] - normalise_weights[key][0])*torch.abs( (x[:,key] - x_pred[:,key]))
            
        # Sum to 1 over the categorical indexes of a feature
        for v in model.encoded_categorical_feature_indexes:
            temp = -torch.abs(  1.0-torch.sum( x_pred[:, v[0]:v[-1]+1], axis=1) )
            recon_err += temp

        count+= torch.sum(x_pred[:,:s]<0,axis=1).float()
        count+= torch.sum(x_pred[:,:s]>1,axis=1).float()        
            
    recon_err = recon_err / mc_samples

    print('Avg wrong cont dim: ', torch.mean(count)/mc_samples)
    print('recon: ',-torch.mean(recon_err), ' KL: ', torch.mean(kl_divergence))
    return -torch.mean(recon_err - kl_divergence) 


def ae_compute_elbo(model, x):
    em, ev = model.encoder(x)
    kl_divergence = 0.5*torch.mean( em**2 +ev - torch.log(ev) - 1, axis=1 ) 

    z = model.sample_latent_code(em, ev)
    dm = model.decoder(z)
    x_pred = dm  
    log_px_z = torch.tensor(0.0)

    return torch.mean(log_px_z), torch.mean(kl_divergence), x, x_pred, torch.argmax( pred_model(x_pred), dim=1 )


def ae_train(model, train_dataset, optimizer, normalise_weights, epochs=1000, batch_size=1024):
    batch_num=0
    train_loss=0.0
    train_size=0
    train_dataset= np.array_split( train_dataset, train_dataset.shape[0]//batch_size ,axis=0 )
    for i in range(len(train_dataset)):
        optimizer.zero_grad()
        train_x = train_dataset[i]
        train_x= torch.tensor( train_x ).float() 
        train_size += train_x.shape[0]

        train_x = torch.Tensor(train_x)
        out= model(train_x)
        loss = ae_compute_loss(model, out, train_x, normalise_weights)            
        loss.backward()
        train_loss += loss.item()
        optimizer.step()                               

        batch_num+=1

    ret= loss/batch_num
    print('Train Avg Loss: ', ret, train_size)
    return ret


def ae_test(model, train_dataset, epochs=1, batch_size=2048):
    batch_num=0
    likelihood=0.0
    valid_cf_count=0
    train_size=0
    train_dataset= np.array_split( train_dataset, train_dataset.shape[0]//batch_size ,axis=0 )
    index=random.randrange(0,len(train_dataset),1)

    for i in range(len(train_dataset)):

        train_x = train_dataset[i]
        train_x= torch.tensor( train_x ).float() 
        train_size += train_x.shape[0]        
        
        recon_err, kl_err, x_true, x_pred, cf_label = ae_compute_elbo( model, train_x )
        likelihood += recon_err-kl_err            
                       
        x_pred= d.de_normalize_data( d.get_decoded_data(x_pred.detach().numpy()) )
        x_true= d.de_normalize_data( d.get_decoded_data(x_true.detach().numpy()) )                
        if batch_num == index:
            rand_idx= random.randrange(0, batch_size/2-1, 1)
            print('Likelihood: ', recon_err, kl_err, recon_err-kl_err)
            print('X: ', x_true.iloc[rand_idx,:])
            print('Xpred: ', x_pred.iloc[rand_idx,:])
        batch_num+=1

    ret= likelihood
    print('ELBO Avg: ', ret, train_size )

    return ret

#Argparsing
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='bn1')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--target_class', type=int, default=0)
args = parser.parse_args()

#Main Code
base_data_dir='../data/'
base_model_dir='../models/'
dataset_name=args.dataset_name

#Dataset
if dataset_name=='bn1':
    dataset= pd.read_csv(base_data_dir+dataset_name+'.csv')
    dataset.drop(['Unnamed: 0'], axis=1, inplace=True)	
    params= {'dataframe':dataset.copy(), 'continuous_features':['x1','x2','x3'], 'outcome_name':'y'}
    d = DataLoader(params)

elif dataset_name=='adult':
    dataset = load_adult_income_dataset()
    params= {'dataframe':dataset.copy(), 'continuous_features':['age','hours_per_week'], 'outcome_name':'income'}
    d = DataLoader(params)
    
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

#Load Train, Val, Test Dataset
vae_train_dataset= np.load(base_data_dir+dataset_name+'-train-set.npy')
vae_val_dataset= np.load(base_data_dir+dataset_name+'-val-set.npy')
if args.target_class !=-1:
    # Sampling data corresponding for training Auto Encoder for a particular class only
    vae_train_dataset= vae_train_dataset[vae_train_dataset[:,-1] == args.target_class, :]
    vae_val_dataset= vae_val_dataset[vae_val_dataset[:,-1] == args.target_class, :]
    print('Sanity check for target class auto encoder: ', np.unique( vae_train_dataset[:,-1], return_counts=True ), np.unique( vae_val_dataset[:,-1], return_counts=True ) )
    
vae_train_dataset= vae_train_dataset[:,:-1]
vae_val_dataset= vae_val_dataset[:,:-1]
print(vae_train_dataset.shape, vae_val_dataset.shape)

with open(base_data_dir+dataset_name+'-normalise_weights.json') as f:
    normalise_weights= json.load(f)
normalise_weights = {int(k):v for k,v in normalise_weights.items()}

with open(base_data_dir+dataset_name+'-mad.json') as f:
    mad_feature_weights= json.load(f)

print(normalise_weights)
print(mad_feature_weights)

#Load Black Box Prediction Model
data_size= len(d.encoded_feature_names)
pred_model= BlackBox(data_size)
path= base_model_dir + dataset_name +'.pth'
pred_model.load_state_dict(torch.load(path))
pred_model.eval()

# Initiliase new model
wm1=1e-2
wm2=1e-2
wm3=1e-2

encoded_size=10
ae_vae = AutoEncoder(data_size, encoded_size, d)
learning_rate = 1e-2
batch_size= args.batch_size
ae_vae_optimizer = optim.Adam([
    {'params': filter(lambda p: p.requires_grad, ae_vae.encoder_mean.parameters()),'weight_decay': wm1},
    {'params': filter(lambda p: p.requires_grad, ae_vae.encoder_var.parameters()),'weight_decay': wm2},
    {'params': filter(lambda p: p.requires_grad, ae_vae.decoder_mean.parameters()),'weight_decay': wm3},
], lr=learning_rate)

# Train AE
loss_val = []
likelihood_val = []

traverse(vae_train_dataset, 1, len(vae_train_dataset))

for epoch in range(args.epoch):
    np.random.shuffle(vae_train_dataset)
    loss_val.append( ae_train( ae_vae, vae_train_dataset, ae_vae_optimizer, normalise_weights, 1, batch_size) )
    print('Done training for epoch: ', epoch)
    if epoch%2==0:
        likelihood= ae_test( ae_vae, vae_train_dataset, 1, batch_size)
        likelihood_val.append( likelihood )
        
print(loss_val)
print(likelihood_val)

#Saving the final model
torch.save(ae_vae.state_dict(),  base_model_dir + dataset_name + '-'+ str(args.batch_size) + '-' + str(args.epoch) + '-' +  'target-class-' + str(args.target_class) + '-' + 'auto-encoder' + '.pth')
