#Important Classes
from dataloader import DataLoader
from vae_model import CF_VAE
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

education_change= {'HS-grad': ['Some-college', 'Bachelors', 'Assoc', 'Prof-school', 'Masters', 'Doctorate'],
        'School': ['Some-college', 'Bachelors', 'Assoc', 'Prof-school', 'Masters', 'Doctorate'],
        'Bachelors': ['Prof-school', 'Masters', 'Doctorate'],
        'Assoc': ['Prof-school', 'Masters', 'Doctorate'] ,
        'Some-college': ['Prof-school', 'Masters', 'Doctorate'],
        'Masters': ['Doctorate'],
        'Prof-school': ['Doctorate'],
        'Doctorate': [] }

education_score= {'HS-grad': 0,
        'School': 0,
        'Bachelors': 1,
        'Assoc': 1,
        'Some-college': 1,
        'Masters': 2,
        'Prof-school': 2,
        'Doctorate': 3 }

def traverse(train_dataset, epochs=1, batch_size=128):
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

    
def gen_cf_set_adult(model, pred_mode, train_dataset, fine_tune_size, d, upper_limit, case):

    train_dataset= np.array_split( train_dataset, train_dataset.shape[0] ,axis=0 )    
    good_cf_dataset_save={}

    temp= d.data_df.copy()
    temp.drop('income', axis=1, inplace=True)
    columns=temp.columns 
    gen_cf_pd=pd.DataFrame(columns=columns)
    gen_cf_inv= []
    gen_cf_label= []
    
    counter=0
    for i in range(len(train_dataset)):

        # Oracle Number of FineTune CF Examples 
        if counter>= fine_tune_size:
            break        
        
        train_x = train_dataset[i]
        train_y = torch.argmax( pred_model(torch.tensor( train_x ).float()), dim=1 )                
    
        valid_change= 0
        invalid_change= 0
        low_change= 0
        no_change= 0
        break_cond= valid_change+invalid_change+low_change+no_change
        
        valid_oracle_cf=1
        gen_cf_list=[] 
        loop_iters= 0       
        
        # Oracle CF's per train example: Upper Limit
        while (valid_change+invalid_change+low_change+no_change)<upper_limit:
            # Unable to find requested number of Feasible CFs for the current train example
            if loop_iters>50:
                valid_oracle_cf=0
                break
                
            loop_iters +=1            
            recon_err, kl_err, x_true, x_pred, cf_label = model.compute_elbo( torch.tensor( train_x ).float(), 1-train_y, pred_model )            
            x_pred= d.de_normalize_data( d.get_decoded_data(x_pred.detach().numpy()) )
            x_true= d.de_normalize_data( d.get_decoded_data(x_true.detach().numpy()) )                
            cf_label = cf_label.numpy()
                
            #Should be Valid CF and Low to High CF considered only
            if train_y[0] == cf_label[0] or cf_label[0]==0:
                continue

            # Age should not decrease constraint
            if case==0:
                label=-1
                age_idx = x_true.columns.get_loc('age')            
                for idx in range(x_true.shape[0]): 
                    if x_pred.iloc[idx, age_idx] >= x_true.iloc[idx, age_idx]:
                        label=1
                        valid_change+=1
                    else:
                        invalid_change+=1
                        label=0

            # Education Age Relationship
            elif case==1:
                ed_idx = x_true.columns.get_loc('education')            
                age_idx = x_true.columns.get_loc('age')       
                label=-1
                for idx in range(x_true.shape[0]): 
                    if education_score[ x_pred.iloc[idx,ed_idx] ] < education_score[ x_true.iloc[idx,ed_idx] ]:
                        label=0
                        invalid_change+=1
                    elif education_score[ x_pred.iloc[idx,ed_idx] ] > education_score[ x_true.iloc[idx,ed_idx] ]:
                        if x_pred.iloc[idx, age_idx] > x_true.iloc[idx, age_idx]:
                            valid_change += 1
                            label=1
                        else:
                            invalid_change += 1
                            label=0
                    elif education_score[ x_pred.iloc[idx,ed_idx] ] == education_score[ x_true.iloc[idx,ed_idx] ]:
                        no_change += 1                    
                        if x_pred.iloc[idx, age_idx] >= x_true.iloc[idx, age_idx]:
                            valid_change += 1
                            label=1
                        else:
                            invalid_change += 1
                            label=0
             
            if label!=-1:
                temp=[]
                temp.append( x_pred )
                temp.append( train_x )
                temp.append( label )                
                gen_cf_list.append(temp)
                
        # Saving the triplet (CF, X, Label)
        if valid_oracle_cf==1:
            for item in gen_cf_list:                
                x_pred=item[0]
                train_x=item[1]
                label=item[2]                
                good_cf_dataset_save[counter]=[ x_pred.to_json(),  train_x.tolist(), label]                        
                gen_cf_pd.loc[counter]= x_pred.loc[0]
                gen_cf_inv.append( train_x[0] )
                gen_cf_label.append(label)
            counter+=1
    
            print(no_change, valid_change, invalid_change, low_change)
            print('Done for: ', i, loop_iters, len(gen_cf_list), counter)

    return good_cf_dataset_save  

def gen_cf_set_bn1(model, pred_model, train_dataset, fine_tune_size, d, upper_limit):

    train_dataset= np.array_split( train_dataset, train_dataset.shape[0] ,axis=0 )    
    good_cf_dataset_save={}

    temp= d.data_df.copy()
    temp.drop('y', axis=1, inplace=True)
    columns=temp.columns 
    gen_cf_pd=pd.DataFrame(columns=columns)
    gen_cf_inv= []
    gen_cf_label= []
    
    counter=0
    for i in range(len(train_dataset)):

        # Oracle Number of FineTune CF Examples 
        if counter>= fine_tune_size:
            break        
        
        train_x = train_dataset[i]
        train_y = torch.argmax( pred_model(torch.tensor( train_x ).float()), dim=1 )                
    
        valid_change= 0
        invalid_change= 0
        low_change= 0
        no_change= 0
        break_cond= valid_change+invalid_change+low_change+no_change
        
        valid_oracle_cf=1
        gen_cf_list=[] 
        loop_iters= 0       
        
        # Oracle CF's per train example: Upper Limit
        while (valid_change+invalid_change+low_change+no_change)<upper_limit:
            # Unable to find requested number of Feasible CFs for the current train example
            if loop_iters>50:
                valid_oracle_cf=0
                break
                
            loop_iters +=1            
            recon_err, kl_err, x_true, x_pred, cf_label = model.compute_elbo( torch.tensor( train_x ).float(), 1-train_y, pred_model )            
            x_pred= d.de_normalize_data( d.get_decoded_data(x_pred.detach().numpy()) )
            x_true= d.de_normalize_data( d.get_decoded_data(x_true.detach().numpy()) )                
            cf_label = cf_label.numpy()
                
            if train_y[0] == cf_label[0]:
                continue

            # No gender and race change
            label=-1
            x1_idx = x_true.columns.get_loc('x1')            
            x2_idx = x_true.columns.get_loc('x2')            
            x3_idx = x_true.columns.get_loc('x3')                            
            for idx in range(x_true.shape[0]):
                if x_pred.iloc[idx, x1_idx] < x_true.iloc[idx, x1_idx] and x_pred.iloc[idx,x2_idx] < x_true.iloc[idx,x2_idx] :
                    if x_pred.iloc[idx, x3_idx] < x_true.iloc[idx, x3_idx]:
                        valid_change+=1
                        label=1
                    else:
                        invalid_change += 1
                        label=0
                if x_pred.iloc[idx, x1_idx] > x_true.iloc[idx, x1_idx] and x_pred.iloc[idx,x2_idx] > x_true.iloc[idx,x2_idx] :
                    if x_pred.iloc[idx, x3_idx] > x_true.iloc[idx, x3_idx]:
                        valid_change+=1
                        label=1
                    else:
                        invalid_change += 1                        
                        label=0                                        
             
            if label!=-1:
                temp=[]
                temp.append( x_pred )
                temp.append( train_x )
                temp.append( label )                
                gen_cf_list.append(temp)
                
        # Saving the triplet (CF, X, Label)
        if valid_oracle_cf==1:
            for item in gen_cf_list:                
                x_pred=item[0]
                train_x=item[1]
                label=item[2]                
                good_cf_dataset_save[counter]=[ x_pred.to_json(),  train_x.tolist(), label]                        
                gen_cf_pd.loc[counter]= x_pred.loc[0]
                gen_cf_inv.append( train_x[0] )
                gen_cf_label.append(label)
            counter+=1
    
            print(no_change, valid_change, invalid_change, low_change)
            print('Done for: ', i, loop_iters, len(gen_cf_list), counter)

    return good_cf_dataset_save         

#Argparsing
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='bn1')
parser.add_argument('--fine_tune_size', type=int, default=100)
parser.add_argument('--upper_limit', type=int, default=10)
parser.add_argument('--cf_path', type=str, default='')

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

#Load Black Box Prediction Model
data_size= len(d.encoded_feature_names)
pred_model= BlackBox(data_size)
path= base_model_dir + dataset_name +'.pth'
pred_model.load_state_dict(torch.load(path))
pred_model.eval()

#Load Train, Val, Test Dataset
vae_train_dataset= np.load(base_data_dir+dataset_name+'-train-set.npy')
if dataset_name=='adult':
    vae_train_dataset= vae_train_dataset[vae_train_dataset[:,-1]==0,:]
print(vae_train_dataset.shape)
vae_train_dataset= vae_train_dataset[:,:-1]
np.random.shuffle( vae_train_dataset )
fine_tune_size= args.fine_tune_size
upper_limit= args.upper_limit
#fine_tune_dataset= vae_train_dataset[:fine_tune_size]

#Load CF Generator Model
encoded_size=10
path= base_model_dir + args.cf_path
cf_vae = CF_VAE(data_size, encoded_size, d)
cf_vae.load_state_dict(torch.load(path))
cf_vae.eval()

#traverse(fine_tune_dataset, 1, len(fine_tune_dataset))

if dataset_name=='bn1':
    good_cf_dataset_save=  gen_cf_set_bn1( cf_vae, pred_model, vae_train_dataset, fine_tune_size, d, upper_limit )
    f=open( base_data_dir+ dataset_name + '-fine-tune-size-' + str(args.fine_tune_size) + '-upper-lim-' + str(args.upper_limit)  + '-good-cf-set.json','w')
    f.write(json.dumps(good_cf_dataset_save))
    f.close()

elif dataset_name=='adult':

    good_cf_dataset_save=  gen_cf_set_adult( cf_vae, pred_model, vae_train_dataset, fine_tune_size, d, upper_limit, 0 )
    f=open( base_data_dir+ dataset_name + '-fine-tune-size-' + str(args.fine_tune_size) + '-upper-lim-' + str(args.upper_limit)  +  '-age-good-cf-set.json','w')
    f.write(json.dumps(good_cf_dataset_save))
    f.close()
    print('Done for the case: Age ')

    good_cf_dataset_save=  gen_cf_set_adult( cf_vae, pred_model, vae_train_dataset, fine_tune_size, d, upper_limit, 1)
    f=open( base_data_dir+ dataset_name + '-fine-tune-size-' + str(args.fine_tune_size) + '-upper-lim-' + str(args.upper_limit)  + '-age-ed-good-cf-set.json','w')
    f.write(json.dumps(good_cf_dataset_save))
    f.close()
    print('Done for the case: Age Ed')