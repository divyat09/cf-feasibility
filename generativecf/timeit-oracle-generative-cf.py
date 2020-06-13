#Important Classes
from scripts.dataloader import DataLoader
from scripts.vae_model import CF_VAE
from scripts.blackboxmodel import BlackBox
from scripts.evaluation_functions import *
from scripts.helpers import *

#Normie stuff
import sys
import random
import pandas as pd
import numpy as np
import json
import argparse
import os
import timeit

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
cuda=torch.device('cuda:0')

# To time the function gen-explanation
def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped

def compute_loss( model, model_out, x, target_label, normalise_weights, validity_reg, margin ): 

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

    #Validity         
    temp_logits = pred_model(x_pred)
    #validity_loss = -F.cross_entropy(temp_logits, target_label)
    validity_loss= torch.zeros(1).to(cuda)
    temp_1= temp_logits[target_label==1,:]
    temp_0= temp_logits[target_label==0,:]
    if torch.sum( target_label==1 ):
        validity_loss += F.hinge_embedding_loss( F.sigmoid(temp_1[:,1]).to(cuda) - F.sigmoid(temp_1[:,0]).to(cuda), torch.tensor(-1).to(cuda), margin, reduction='mean')
    if torch.sum( target_label==0 ):
        validity_loss += F.hinge_embedding_loss( F.sigmoid(temp_0[:,0]).to(cuda) - F.sigmoid(temp_0[:,1]).to(cuda), torch.tensor(-1).to(cuda), margin, reduction='mean')   
        
    for i in range(1,mc_samples):
        x_pred = dm[i]        

        recon_err += -torch.sum( torch.abs(x[:,s:-1] - x_pred[:,s:-1]), axis=1 )
        for key in normalise_weights.keys():
                # recon_err+= -(1/mad_feature_weights[d.encoded_feature_names[int(key)]])*(normalise_weights[key][1] - normalise_weights[key][0])*torch.abs( (x[:,key] - x_pred[:,key]))
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
        
        if torch.sum(target_label==1):
            validity_loss += F.hinge_embedding_loss( F.sigmoid(temp_1[:,1]).to(cuda) - F.sigmoid(temp_1[:,0]).to(cuda), torch.tensor(-1).to(cuda), margin, reduction='mean')
        if torch.sum(target_label==0):
            validity_loss += F.hinge_embedding_loss( F.sigmoid(temp_0[:,0]).to(cuda) - F.sigmoid(temp_0[:,1]).to(cuda), torch.tensor(-1).to(cuda), margin, reduction='mean')   
        
    recon_err = recon_err / mc_samples
    validity_loss = -1*validity_reg*validity_loss/mc_samples

    print('Avg wrong cont dim: ', torch.mean(count)/mc_samples)
    print('recon: ',-torch.mean(recon_err), ' KL: ', torch.mean(kl_divergence), ' Validity: ', -validity_loss)
    return -torch.mean(recon_err - kl_divergence) - validity_loss

def likelihood_score(x, label, mean):
        good_score=[]
        bad_score=[]
        for idx in range(0, len(x)):
                if label[idx] == 1:
                        # good_score.append( (x[idx] - mean[idx])*(1./logvar[idx])*(x[idx]-mean[idx] )) 
                        # good_score.append( F.sigmoid( torch.exp( -.5 * ((x[idx] - mean[idx])*(1./logvar[idx])*(x[idx]-mean[idx]) )) )) 
                        good_score.append( torch.exp( -.5 * ((x[idx] - mean[idx])*(x[idx]-mean[idx]) )) ) 
                        # good_score.append( torch.exp( -.5 * ((x[idx] - mean[idx])*(1./logvar[idx])*(x[idx]-mean[idx]) ))/torch.sqrt(torch.prod(logvar[idx])) ) 
                else:
                        # bad_score.append( (x[idx] - mean[idx])*(1./logvar[idx])*(x[idx]-mean[idx]) )
                        # bad_score.append( F.sigmoid( torch.exp( -.5 * ((x[idx] - mean[idx])*(1./logvar[idx])*(x[idx]-mean[idx]) ))  ))
                        bad_score.append( torch.exp( -.5 * ((x[idx] - mean[idx])*(x[idx]-mean[idx]) )  ))
                        # bad_score.append( torch.exp( -.5 * ((x[idx] - mean[idx])*(1./logvar[idx])*(x[idx]-mean[idx]) )  )/torch.sqrt(torch.prod(logvar[idx]))  )

    #     print( torch.mean(torch.stack(good_score)), torch.mean(torch.stack(bad_score)), (1-torch.mean(torch.stack(good_score))) + torch.mean(torch.stack(bad_score)))
        if len(good_score) and len(bad_score):
            # print(torch.mean(torch.stack(good_score)), torch.mean(torch.stack(bad_score)))
            return (1-torch.mean(torch.stack(good_score))) + torch.mean(torch.stack(bad_score))
#             return torch.mean(torch.stack(bad_score))
        elif len(good_score):
            return (1-torch.mean(torch.stack(good_score)))
        elif len(bad_score):
            return torch.mean(torch.stack(bad_score))
        else:
            print('Nope')
            return torch.tensor(0.0)
    #     return torch.mean(torch.stack(bad_score))

def train_cflabel_likelihood_loss(model, optimizer, normalise_weights, validity_reg, oracle_reg, margin, case, epochs=1000, batch_size=1024):
    batch_num=0
    train_loss=0.0
    train_size=0

    rand=np.array(range(len(gen_cf_dataset)))
    np.random.shuffle(rand)
    train_dataset_batches= np.array_split( gen_cf_dataset[rand], gen_cf_dataset.shape[0]//batch_size, axis=0 )
    train_inv_batches= np.array_split( gen_cf_inv[rand], gen_cf_inv.shape[0]//batch_size ,axis=0 )
    train_label_batches= np.array_split( gen_cf_label[rand], gen_cf_label.shape[0]//batch_size ,axis=0 )

    for i in range(len(train_dataset_batches)):
            optimizer.zero_grad()

            train_x = torch.tensor( train_inv_batches[i] ).float().to(cuda)
            train_y = 1.0-torch.argmax( pred_model(train_x), dim=1 )
            train_cf_label = torch.tensor( train_label_batches[i], dtype=torch.long ).to(cuda)
            gen_cf_x = torch.tensor( train_dataset_batches[i] ).float().to(cuda)
            train_size += train_x.shape[0] 

            out= model(train_x, train_y)
            loss = compute_loss(model, out, train_x, train_y, normalise_weights, validity_reg, margin)                   

            dm = out['x_pred']
            mc_samples = out['mc_samples']
            likelihood_loss= likelihood_score( gen_cf_x, train_cf_label, dm[0] )        
            for j in range(1, mc_samples):
                    likelihood_loss+= likelihood_score( gen_cf_x, train_cf_label, dm[j] )        

            likelihood_loss= oracle_reg*likelihood_loss/mc_samples
            print('Likelihood Score: ', likelihood_loss)

            loss+= likelihood_loss
            train_loss += loss.item()
            batch_num+=1

            if case:
                    likelihood_loss.backward()
                    optimizer.step()
            else:
                    loss.backward()
                    optimizer.step()

    ret= train_loss/batch_num
    print('Train Avg Loss: ', ret, train_size)

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
        
        recon_err, kl_err, x_true, x_pred, cf_label = model.compute_elbo( train_x, 1.0-train_y, pred_model )
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
parser.add_argument('--oracle_reg', type=float, default=0.5)
parser.add_argument('--cf_path', type=str, default='..')
parser.add_argument('--const_case', type=int, default=0)
parser.add_argument('--learning_rate',type=float,default=1e-2)
parser.add_argument('--eval_case',type=int,default=0)
parser.add_argument('--supervision_limit',type=int,default=100)
parser.add_argument('--oracle_data',type=str,default='..')
parser.add_argument('--margin', type=float, default=0.5)
parser.add_argument('--htune', type=int, default=0)                                                               
args = parser.parse_args()

#Main Code
base_data_dir='data/'
if args.htune:
    if args.dataset_name == 'adult':
        if args.const_case==1:
            base_model_dir='htune/oracle-age-ed/'
        else:
            base_model_dir='htune/oracle-age/'
    else:
        base_model_dir='htune/oracle/'
else:
        base_model_dir='models/'
dataset_name=args.dataset_name

if not os.path.exists(base_model_dir+args.oracle_data[:-5]):
    os.mkdir(base_model_dir+args.oracle_data[:-5])

#Dataset
if dataset_name=='bn1':
    dataset= pd.read_csv(base_data_dir+dataset_name+'.csv')
    dataset.drop(['Unnamed: 0'], axis=1, inplace=True)  
    params= {'dataframe':dataset.copy(), 'continuous_features':['x1','x2','x3'], 'outcome_name':'y'}
    d = DataLoader(params)
    temp=d.train_df.copy()
    temp.drop('y', axis=1, inplace=True) 

elif dataset_name=='adult':
    dataset = load_adult_income_dataset()
    params= {'dataframe':dataset.copy(), 'continuous_features':['age','hours_per_week'], 'outcome_name':'income'}
    d = DataLoader(params)  
    temp=d.train_df.copy()
    temp.drop('income', axis=1, inplace=True) 
    
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
    print(d.encoded_feature_names)   
    
    temp=d.train_df.copy()
    temp.drop('outcome', axis=1, inplace=True) 
    
#Load Train, Val, Test Dataset
vae_val_dataset= np.load(base_data_dir+dataset_name+'-val-set.npy')

if dataset_name == 'adult':
    vae_val_dataset= vae_val_dataset[vae_val_dataset[:,-1]==0,:]

vae_val_dataset= vae_val_dataset[:,:-1]

with open(base_data_dir+dataset_name+'-normalise_weights.json') as f:
    normalise_weights= json.load(f)
normalise_weights = {int(k):v for k,v in normalise_weights.items()}

with open(base_data_dir+dataset_name+'-mad.json') as f:
    mad_feature_weights= json.load(f)

print(normalise_weights)
print(mad_feature_weights)

# Loading the fine tune dataset
columns=temp.columns
out1= pd.DataFrame(columns=columns)
out2= []
out3= []
counter=0


f=open( base_data_dir+args.oracle_data, 'r')
gen_cf_dict= json.load(f)
for key in gen_cf_dict.keys():
    out1.loc[counter] = pd.read_json(gen_cf_dict[key][0]).loc[0] 
    out2.append( gen_cf_dict[key][1][0] ) 
    out3.append( gen_cf_dict[key][2] )
    counter+=1

gen_cf_pd= out1
gen_cf_inv= np.array(out2)
gen_cf_label= np.array(out3)

print(gen_cf_pd)
print(gen_cf_pd.shape)
print(gen_cf_inv.shape)
print(gen_cf_label.shape)

# For generating proper one hot encodings: Merge with the whole dataset to get all the categories
corr_temp= pd.concat([gen_cf_pd,temp],keys=[0,1])
corr_temp= d.one_hot_encode_data(corr_temp)
print(corr_temp.xs(0).shape, corr_temp.xs(1).shape)

gen_cf_encoded_data= corr_temp.xs(0)
gen_cf_encoded_data= d.normalize_data(gen_cf_encoded_data)
gen_cf_dataset = gen_cf_encoded_data.to_numpy()

#Load Black Box Prediction Model
data_size= len(d.encoded_feature_names)
pred_model= BlackBox(data_size).to(cuda)
path= base_model_dir + dataset_name +'.pth'
pred_model.load_state_dict(torch.load(path))
pred_model.eval()


#Load CF Generator Model
wm1=1e-2
wm2=1e-2
wm3=1e-2
wm4=1e-2

encoded_size=10
path= base_model_dir + args.cf_path
cf_vae = CF_VAE(data_size, encoded_size, d).to(cuda)
cf_vae.load_state_dict(torch.load(path))
cf_vae.eval()
learning_rate = 1e-2
batch_size= args.batch_size
cf_vae_optimizer = optim.Adam([
    {'params': filter(lambda p: p.requires_grad, cf_vae.encoder_mean.parameters()),'weight_decay': wm1},
    {'params': filter(lambda p: p.requires_grad, cf_vae.encoder_var.parameters()),'weight_decay': wm2},
    {'params': filter(lambda p: p.requires_grad, cf_vae.decoder_mean.parameters()),'weight_decay': wm3}
], lr=learning_rate)


#Contraint Score vs Supervision Amount evaluation case
if args.eval_case:    
    idx = np.random.choice(np.arange(len(gen_cf_label)), args.supervision_limit, replace=False)    
    gen_cf_dataset= gen_cf_dataset[idx]
    gen_cf_inv= gen_cf_inv[idx]
    gen_cf_label= gen_cf_label[idx]      
    
#Train CFVAE
loss_val = []
likelihood_val = []
valid_cf_count = []

validity_reg= args.validity_reg
oracle_reg= args.oracle_reg
margin= args.margin
if not os.path.exists(base_model_dir+args.oracle_data[:-5]):
    os.mkdir(base_model_dir+args.oracle_data[:-5])    
    
if args.htune==0:
    base_model_dir= base_model_dir  + args.oracle_data[:-5] + '/'

# Include the label prediction term loss while training the VAE
loss_val = []
likelihood_val = []
valid_cf_count = []

#Evaluation Time
eval_time=0.0
for epoch in range(args.epoch):    
    wrapped = wrapper(train_cflabel_likelihood_loss, cf_vae, cf_vae_optimizer, normalise_weights, validity_reg, oracle_reg, margin, 0, 1, batch_size)
    eval_time+= timeit.timeit(wrapped, number=1)
    print('-----------------------------------')
    print('Time taken: ', eval_time)
    print('-----------------------------------')

    print('Done training for epoch: ', epoch)

#Saving the final model
torch.save(cf_vae.state_dict(),  base_model_dir + dataset_name + '-eval-case-' + str(args.eval_case) + '-supervision-limit-' + str(args.supervision_limit) +'-const-case-' + str(args.const_case) + '-margin-' + str(args.margin)  + '-oracle_reg-'+ str(args.oracle_reg) +  '-validity_reg-'+ str(args.validity_reg) + '-epoch-' + str(args.epoch) + '-' + 'oracle-gen' + '.pth')
