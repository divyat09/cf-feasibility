#Important Classes
from scripts.dataloader import DataLoader
from scripts.vae_model import CF_VAE, AutoEncoder
from scripts.blackboxmodel import BlackBox
from scripts.evaluation_functions import *

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
#GPU
cuda= torch.device('cuda:0')

def de_normalise( x, normalise_weights ):
    return (normalise_weights[1] - normalise_weights[0])*x + normalise_weights[0]

def traverse(train_dataset, epochs=1, batch_size=128):
    batch_num=0
    loss=0.0
    train_size=0
    train_dataset= np.array_split( train_dataset, train_dataset.shape[0]//batch_size ,axis=0 )
    for i in range(len(train_dataset)):
        train_x = train_dataset[i]
        train_x= torch.tensor( train_x ).float().to(cuda)
        train_y = torch.argmax( pred_model(train_x), dim=1 )
        train_size += train_x.shape[0]
        print(np.unique(train_y.cpu(), return_counts=True))
    print(train_size)    

def ae_reconstruct_loss( model, model_out, x, normalise_weights ): 
    
    em = model_out['em']
    ev = model_out['ev']
    z  = model_out['z']
    dm = model_out['x_pred']
    mc_samples = model_out['mc_samples']
            
    #Reconstruction Term
    x_pred = dm[0]        
    s= model.encoded_start_cat
    recon_err = -torch.sum( torch.abs(x[:,s:-1] - x_pred[:,s:-1]), axis=1 )
    for key in normalise_weights.keys():
        recon_err+= -(normalise_weights[key][1] - normalise_weights[key][0])*torch.abs(x[:,key] - x_pred[:,key]) 
            
#     for i in range(1,mc_samples):
#         x_pred = model.sample_latent_code(dm[i], dv[i])        
#         recon_err += -torch.sum( torch.abs(x[:,s:-1] - x_pred[:,s:-1]), axis=1 )
#         for key in normalise_weights.keys():
#             recon_err+= -(1/mad_feature_weights[d.encoded_feature_names[int(key)]])*(normalise_weights[key][1] - normalise_weights[key][0])*torch.abs( (x[:,key] - x_pred[:,key]))
                
#     recon_err = recon_err / mc_samples
    return -torch.mean(recon_err) 

def model_constraint_score(x, xpred, constrained_feat_indices, model_param, delta_case ):    
    idx=constrained_feat_indices[0]

    if delta_case:
        score= de_normalise(xpred[:,idx], normalise_weights[0]) - de_normalise(x[:,idx], normalise_weights[0] ) 
        for i in range(1, len(constrained_feat_indices)):
            idx=constrained_feat_indices[i]
            score+= -1*model_param[i]*( de_normalise(xpred[:,idx], normalise_weights[idx]) - de_normalise(x[:,idx], normalise_weights[idx]) )
    else:
        score= de_normalise(xpred[:,idx], normalise_weights[0]) 
        for i in range(1, len(constrained_feat_indices)):
            idx=constrained_feat_indices[i]
            score+= -1*model_param[i]*( de_normalise(xpred[:,idx], normalise_weights[idx]) )
        
    return torch.mean( score**2 )

#     constrained_feature_indices=[2,0,1]
def linear_model_feature_approx(train_dataset, constrained_feature_indices, param_tensor):    
    #Define the linear model parameters: [ effect, cause_1, cause_2, cause_3, ... ]
    num_params=len(constrained_feature_indices)
    wm= 1e-2
    learning_rate=1e-3
    optimizer= optim.Adam([
        {'params': filter(lambda p: p.requires_grad, param_tensor),'weight_decay': wm}
        ], lr=learning_rate)
    
    batch_num=0
    train_loss=0.0
    train_size=0
    train_dataset= np.array_split( train_dataset, train_dataset.shape[0]//batch_size ,axis=0 )
    
    for i in range(len(train_dataset)):
        optimizer.zero_grad()
        train_x= torch.tensor( train_dataset[i] ).float().to(cuda) 
        train_size += train_x.shape[0]
        
        # Forward Pass of the model
        model_loss= torch.zeros(train_x.shape[0]).to(cuda) + param_tensor[0]
        for i in range(0, len(constrained_feature_indices)):
            idx=constrained_feature_indices[i]
            if i==0:
                model_loss+= de_normalise( train_x[:,idx], normalise_weights[idx])
            else:
                model_loss+= -1*param_tensor[i]*de_normalise( train_x[:,idx], normalise_weights[idx])
            
        model_loss= torch.sum( model_loss**2, axis=0 )
        model_loss= model_loss.view(1)
        #print('Model Loss: ', model_loss)

        # Constraint implications on model parameters
        for i in range(1, len(constrained_feature_indices)):
            idx= constrained_feature_indices[i]
            reg= 1

            hinge_loss= F.hinge_embedding_loss( param_tensor[i], torch.tensor(-1).to(cuda), 0 ).to(cuda)
            #hinge_loss.data[hinge_loss>0.1]=0
            #print('Hinge Loss: ', param_tensor[idx], hinge_loss)
            model_loss+= reg*hinge_loss

        # Backward Pass
        train_loss += model_loss
        model_loss.backward()
        batch_num+=1
        optimizer.step()

    ret= train_loss
    print('Train Avg Loss: ', ret, train_size)
    print('Param: ', param_tensor)
    return param_tensor


def compute_root_node_loss( model, model_out, x, target_label, normalise_weights, validity_reg, constraint_nodes, margin ): 
    
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
        if int(key) not in constraint_nodes:
            # recon_err+= -(1/mad_feature_weights[d.encoded_feature_names[int(key)]])*(normalise_weights[key][1] - normalise_weights[key][0])*torch.abs(x[:,key] - x_pred[:,key])         
            recon_err+= -(normalise_weights[key][1] - normalise_weights[key][0])*torch.abs(x[:,key] - x_pred[:,key]) 

    # Sum to 1 over the categorical indexes of a feature
    for v in model.encoded_categorical_feature_indexes:
        temp = -torch.abs(  1.0-torch.sum( x_pred[:, v[0]:v[-1]+1], axis=1) )
        recon_err += temp
    
    count=0
    count+= torch.sum(x_pred[:,:s]<0,axis=1).float()
    count+= torch.sum(x_pred[:,:s]>1,axis=1).float()    
    
    #Validity         
    temp_logits = pred_model(x_pred)
    #validity_loss = -F.cross_entropy(temp_logits, target_label)
    validity_loss= torch.zeros(1).to(cuda)
    temp_1= temp_logits[target_label==1,:]
    temp_0= temp_logits[target_label==0,:]
    validity_loss += F.hinge_embedding_loss( F.sigmoid(temp_1[:,1]).to(cuda) - F.sigmoid(temp_1[:,0]).to(cuda), torch.tensor(-1).to(cuda), margin, reduction='mean')
    validity_loss += F.hinge_embedding_loss( F.sigmoid(temp_0[:,0]).to(cuda) - F.sigmoid(temp_0[:,1]).to(cuda), torch.tensor(-1).to(cuda), margin, reduction='mean')
    
    for i in range(1,mc_samples):
        x_pred = dm[i]        

        recon_err += -torch.sum( torch.abs(x[:,s:-1] - x_pred[:,s:-1]), axis=1 )
        for key in normalise_weights.keys():
            if int(key) not in constraint_nodes:
                # recon_err+= -(1/mad_feature_weights[d.encoded_feature_names[int(key)]])*(normalise_weights[key][1] - normalise_weights[key][0])*torch.abs(x[:,key] - x_pred[:,key]) 
                recon_err+= -(normalise_weights[key][1] - normalise_weights[key][0])*torch.abs(x[:,key] - x_pred[:,key]) 
            
        # Sum to 1 over the categorical indexes of a feature
        for v in model.encoded_categorical_feature_indexes:
            temp = -torch.abs(  1.0-torch.sum( x_pred[:, v[0]:v[-1]+1], axis=1) )
            recon_err += temp

        count+= torch.sum(x_pred[:,:s]<0,axis=1).float()
        count+= torch.sum(x_pred[:,:s]>1,axis=1).float()        
            
        #Validity
        temp_logits = pred_model(x_pred)
        #validity_loss += -F.cross_entropy(temp_logits, target_label)
        temp_1= temp_logits[target_label==1,:]
        temp_0= temp_logits[target_label==0,:]
        validity_loss += F.hinge_embedding_loss( F.sigmoid(temp_1[:,1]).to(cuda) - F.sigmoid(temp_1[:,0]).to(cuda), torch.tensor(-1).to(cuda), margin, reduction='mean')
        validity_loss += F.hinge_embedding_loss( F.sigmoid(temp_0[:,0]).to(cuda) - F.sigmoid(temp_0[:,1]).to(cuda), torch.tensor(-1).to(cuda), margin, reduction='mean')
        
    recon_err = recon_err / mc_samples
    validity_loss = -1*validity_reg*validity_loss/mc_samples

    print('Avg wrong cont dim: ', torch.mean(count)/mc_samples)
    print('recon: ',-torch.mean(recon_err), ' KL: ', torch.mean(kl_divergence), ' Validity: ', -validity_loss)
    return -torch.mean(recon_err - kl_divergence) - validity_loss 

def train_constraint_loss(model, ae_model, feature_dist, constrained_feat_indices, train_dataset, optimizer, validity_reg, constraint_reg, ae_reg, margin, normalise_weights, delta_case, epochs=1000, batch_size=1024):
    batch_num=0 
    train_loss=0.0
    train_size=0
    #train_dataset= np.array_split( train_dataset, train_dataset.shape[0]//batch_size ,axis=0 )
    train_dataset= torch.tensor( train_dataset ).float().to(cuda)
    train_dataset= torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    good_cf_count=0
#     for i in range(len(train_dataset)):
    for train_x in enumerate(train_dataset):
        optimizer.zero_grad()
#         train_x = train_dataset[i]
#         train_x= torch.tensor( train_x ).float()
        train_x= train_x[1]
        train_y = 1.0-torch.argmax( pred_model(train_x), dim=1 )
        train_size += train_x.shape[0]

        out= model(train_x, train_y)
#         constraint_nodes=[constrained_feat_indices[0]]
        constraint_nodes=[2]
        loss = compute_root_node_loss(model, out, train_x, train_y, normalise_weights, validity_reg, constraint_nodes, margin )           
        
        dm = out['x_pred']
        mc_samples = out['mc_samples']
        x_pred = dm[0]
        constraint_loss = model_constraint_score( train_x, x_pred, constrained_feat_indices, feature_dist, delta_case )                                   
        for j in range(1, mc_samples):
            x_pred = dm[j]            
            constraint_loss += model_constraint_score( train_x, x_pred, constrained_feat_indices, feature_dist, delta_case )           
        constraint_loss= constraint_reg*constraint_loss/mc_samples
        
        print('Constraint: ', constraint_loss )

        ae_out= ae_model(train_x)
        ae_loss = ae_reconstruct_loss(ae_model, ae_out, x_pred, normalise_weights)         
        ae_loss= ae_reg*ae_loss
        print('Auto Encoder: ', ae_loss)

        loss+= ae_loss        
        loss+= constraint_loss
        train_loss += loss.item()
        batch_num+=1
        
        loss.backward()
        optimizer.step()        

    ret= train_loss
    print('Train Avg Loss: ', ret, train_size)
    return ret

def test(model, train_dataset, epochs=1, batch_size=2048):
    batch_num=0
    likelihood=0.0
    valid_cf_count=0
    train_size=0
    train_dataset= np.array_split( train_dataset, train_dataset.shape[0]//batch_size ,axis=0 )
    index=random.randrange(0,len(train_dataset),1)

    for i in range(len(train_dataset)):

        train_x = train_dataset[i]
        train_x= torch.tensor( train_x ).float().to(cuda)
        train_y = torch.argmax( pred_model(train_x), dim=1 )                
        train_size += train_x.shape[0]        
        
        recon_err, kl_err, x_true, x_pred, cf_label= model.compute_elbo( train_x, 1.0-train_y, pred_model )
        likelihood += recon_err-kl_err            
        
        train_y= train_y.cpu().numpy()
        cf_label = cf_label.cpu().numpy()
        valid_cf_count += np.sum( train_y != cf_label )
               
        x_pred= d.de_normalize_data( d.get_decoded_data(x_pred.detach().cpu().numpy()) )
        x_true= d.de_normalize_data( d.get_decoded_data(x_true.detach().cpu().numpy()) )                
        if batch_num == index:
            rand_idx= random.randrange(0, batch_size/2-1, 1)
            print('Likelihood: ', recon_err, kl_err, recon_err-kl_err)
            print('X: ', x_true.iloc[rand_idx,:])
            print('Xpred: ', x_pred.iloc[rand_idx,:])
        batch_num+=1

    ret= likelihood/batch_num
    print('ELBO Avg: ', ret, train_size )
    print('Valid CF Percentage: ', valid_cf_count, valid_cf_count/train_size)

    return ret, valid_cf_count/train_size

#Argparsing
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='bn1')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--validity_reg', type=float, default=20)
parser.add_argument('--constraint_reg', type=float, default=1)
parser.add_argument('--ae_reg', type=float, default=0)
parser.add_argument('--ae_path', type=str, default='bn1-64-50-target-class--1-auto-encoder.pth')
parser.add_argument('--learning_rate', type=float, default=1e-2)
parser.add_argument('--delta_case', type=int, default=0)
parser.add_argument('--htune', type=int, default=0)
parser.add_argument('--constrained_nodes', type=str, default='2,0,1')
parser.add_argument('--margin', type=float, default=0.1)
args = parser.parse_args()

#Main Code
base_data_dir='data/'
if args.htune:
    base_model_dir='htune/model-approx/'
else:
    base_model_dir='models/'
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

#Load Train, Val, Test Dataset
vae_train_dataset= np.load(base_data_dir+dataset_name+'-train-set.npy')
vae_val_dataset= np.load(base_data_dir+dataset_name+'-val-set.npy')

# CF Generation for only low to high income data points
if dataset_name == 'adult':
    vae_train_dataset= vae_train_dataset[vae_train_dataset[:,-1]==0,:]
    vae_val_dataset= vae_val_dataset[vae_val_dataset[:,-1]==0,:]

vae_train_dataset= vae_train_dataset[:,:-1]
vae_val_dataset= vae_val_dataset[:,:-1]

with open(base_data_dir+dataset_name+'-normalise_weights.json') as f:
    normalise_weights= json.load(f)
normalise_weights = {int(k):v for k,v in normalise_weights.items()}

with open(base_data_dir+dataset_name+'-mad.json') as f:
    mad_feature_weights= json.load(f)

print(normalise_weights)
print(mad_feature_weights)

#Load Black Box Prediction Model
data_size= len(d.encoded_feature_names)
pred_model= BlackBox(data_size).to(cuda)
path= base_model_dir + dataset_name +'.pth'
pred_model.load_state_dict(torch.load(path))
pred_model.eval()

encoded_size=10
ae_vae = AutoEncoder(data_size, encoded_size, d).to(cuda)
path= base_model_dir + args.ae_path
ae_vae.load_state_dict(torch.load(path))
ae_vae.eval()

# Initiliase new model
wm1=1e-2
wm2=1e-2
wm3=1e-2

encoded_size=10
cf_vae = CF_VAE(data_size, encoded_size, d).to(cuda)
learning_rate = args.learning_rate
batch_size= args.batch_size
cf_vae_optimizer = optim.Adam([
    {'params': filter(lambda p: p.requires_grad, cf_vae.encoder_mean.parameters()),'weight_decay': wm1},
    {'params': filter(lambda p: p.requires_grad, cf_vae.encoder_var.parameters()),'weight_decay': wm2},
    {'params': filter(lambda p: p.requires_grad, cf_vae.decoder_mean.parameters()),'weight_decay': wm3}
], lr=learning_rate)

#Train CFVAE
loss_val = []
likelihood_val = []
valid_cf_count = []

validity_reg=args.validity_reg
constraint_reg=args.constraint_reg
ae_reg=args.ae_reg
delta_case=args.delta_case
margin= args.margin

#Model Approximation
constrained_feat_indices=[]
for idx in args.constrained_nodes.split(','):
    constrained_feat_indices.append(int(idx))

num_params=len(constrained_feat_indices)
model_param= []
for idx in range(0, num_params):
    model_param.append( torch.rand(1, requires_grad=True).to(cuda).detach().requires_grad_(True) )
print(model_param)

for epoch in range(200):
    model_param= linear_model_feature_approx(vae_train_dataset, constrained_feat_indices, model_param)

#traverse(vae_train_dataset, 1, len(vae_train_dataset))

for epoch in range(args.epoch):
    np.random.shuffle(vae_train_dataset)
    loss_val.append( train_constraint_loss( cf_vae, ae_vae, model_param, constrained_feat_indices, vae_train_dataset, cf_vae_optimizer, validity_reg, constraint_reg, ae_reg, margin, normalise_weights, delta_case, 1, batch_size) )
    print('Done training for epoch: ', epoch)
        
#Saving the final model
torch.save(cf_vae.state_dict(),  base_model_dir + dataset_name + '-delta-case-' + str(args.delta_case) + '-marign-' + str(args.margin) +  '-ae-reg-' + str(args.ae_reg) + '-constraint_reg-'+ str(args.constraint_reg) +  '-validity_reg-'+ str(args.validity_reg) + '-epoch-' + str(args.epoch) + '-' + 'model-approx-gen' + '.pth')
