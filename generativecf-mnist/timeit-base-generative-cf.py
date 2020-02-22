#Important Classes
from scripts.dataloader import DataLoader
from scripts.vae_model import CF_VAE, AutoEncoder
from scripts.blackboxmodel import BlackBox
from scripts.helpers import *

#Normie stuff
import sys
import random
import pandas as pd
import numpy as np
import json
import argparse
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
cuda= torch.device('cuda:0')

def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped

def compute_loss( model, model_out, x, target_label, validity_reg, margin): 
    
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
    # s would be zero hence it won't make a difference, and it will be simply like a proximity term
    s= model.encoded_start_cat
    recon_err = -torch.sum( torch.abs(x[:,s:-1] - x_pred[:,s:-1]), axis=1 )

    count=0
    count+= torch.sum(x_pred[:,:s]<0,axis=1).float()
    count+= torch.sum(x_pred[:,:s]>1,axis=1).float()    
    
    #Validity         
    temp_logits = pred_model(x_pred)
    #validity_loss = -F.cross_entropy(temp_logits, target_label)    
    validity_loss= torch.zeros(1).to(cuda)
    
    #Loop over total number of classes to compute the Hinge Loss
    num_classes=10
    for t_c in range(num_classes):
    # Compute the validity_loss for data points with target class t_c in the given batch
        temp= temp_logits[target_label==t_c,:]
        if temp.shape[0]==0:
            #No data point in this batch with the target class t_c
            continue
        target_class_batch_score= temp[:, t_c]
        if t_c==0:
            temp= temp[:,t_c+1:]
            # Max along the batch axis in the tensor; torch.max returns both (values, indices) and taking the first argument gives values
            other_class_batch_score= torch.max(temp, dim=1)[0]
        elif t_c == num_classes-1:
            temp= temp[:,:t_c]
            # Max along the batch axis in the tensor
            other_class_batch_score= torch.max(temp, dim=1)[0]          
        else:
            # Concatenate the tensors along the Non Batch Axis
            temp= torch.cat( (temp[:, :t_c], temp[:, t_c+1:]), dim=1 )
            # Max along the batch axis in the tensor
            other_class_batch_score= torch.max(temp, dim=1)[0] 
        
        validity_loss += F.hinge_embedding_loss( F.sigmoid(target_class_batch_score).to(cuda)-F.sigmoid(other_class_batch_score).to(cuda), torch.tensor(-1).to(cuda), margin, reduction='mean' )        
        
    for i in range(1,mc_samples):
        x_pred = dm[i]       

        recon_err += -torch.sum( torch.abs(x[:,s:-1] - x_pred[:,s:-1]), axis=1 )

        count+= torch.sum(x_pred[:,:s]<0,axis=1).float()
        count+= torch.sum(x_pred[:,:s]>1,axis=1).float()        
            
        #Validity
        temp_logits = pred_model(x_pred)
#         validity_loss += -F.cross_entropy(temp_logits, target_label)      

        #Loop over total number of classes to compute the Hinge Loss
        num_classes=10
        for t_c in range(num_classes):
        # Compute the validity_loss for data points with target class t_c in the given batch
            temp= temp_logits[target_label==t_c,:]
            if temp.shape[0]==0:
                #No data point in this batch with the target class t_c
                continue
            target_class_batch_score= temp[:, t_c]

            if t_c==0:
                temp= temp[:,t_c+1:]
                # Max along the batch axis in the tensor; torch.max returns both (values, indices) and taking the first argument gives values
                other_class_batch_score= torch.max(temp, dim=1)[0]
            elif t_c == num_classes-1:
                temp= temp[:,:t_c]
                # Max along the batch axis in the tensor
                other_class_batch_score= torch.max(temp, dim=1)[0]          
            else:
                # Concatenate the tensors along the Non Batch Axis
                temp= torch.cat( (temp[:, :t_c], temp[:, t_c+1:]), dim=1 )
                # Max along the batch axis in the tensor
                other_class_batch_score= torch.max(temp, dim=1)[0] 

            validity_loss += F.hinge_embedding_loss( F.sigmoid(target_class_batch_score).to(cuda)-F.sigmoid(other_class_batch_score).to(cuda),  torch.tensor(-1).to(cuda), margin, reduction='mean' )   

    recon_err = recon_err / mc_samples
    validity_loss = -1*validity_reg*validity_loss/mc_samples

    print('Avg wrong cont dim: ', torch.mean(count)/mc_samples)
    print('recon: ',-torch.mean(recon_err), ' KL: ', torch.mean(kl_divergence), ' Validity: ', -validity_loss)
    return -torch.mean(recon_err - kl_divergence) - validity_loss


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

def generate_target_labels(train_y):
    
    target_y= train_y.clone()
    #print(train_y[:10])

    target_y[ train_y == 0 ]= 8
    target_y[ train_y == 1 ]= 7 
    target_y[ train_y == 2 ]= 3
    target_y[ train_y == 3 ]= 5
    target_y[ train_y == 4 ]= 9
    target_y[ train_y == 5 ]= 8    
    target_y[ train_y == 6 ]= 8
    target_y[ train_y == 7 ]= 9   
    target_y[ train_y == 8 ]= 9    
    target_y[ train_y == 9 ]= 7           
    #print(target_y[:10])
        
    return target_y 

def ae_reconstruct_loss_im1( model, x ): 
    
    x=x.view(1,x.shape[0])
    
    model_out= model(x)
    em = model_out['em']
    ev = model_out['ev']
    z  = model_out['z']
    dm = model_out['x_pred']
    mc_samples = model_out['mc_samples']
            
    #Reconstruction Term
    x_pred = dm[0]        
    # No categorical variables
    s= 0
    recon_err = -torch.norm( torch.abs(x[:,s:-1] - x_pred[:,s:-1]), p=2, dim=1 )**2
            
#     for i in range(1,mc_samples):
#         x_pred = model.sample_latent_code(dm[i], dv[i])        
#         recon_err += -torch.sum( torch.abs(x[:,s:-1] - x_pred[:,s:-1]), axis=1 )
#     recon_err = recon_err / mc_samples
    return -recon_err 

def ae_reconstruct_loss_im2( model, model_all, x ): 
    
    x=x.view(1,x.shape[0])
    
    model_out= model(x)
    dm = model_out['x_pred']
    
    model_out_all= model_all(x)
    dm_all= model_out_all['x_pred']
            
    #Reconstruction Term
    x_pred = dm[0] 
    x_pred_all= dm_all[0]
    # No categorical variables
    s= 0
    recon_err = -torch.norm( torch.abs(x_pred_all[:,s:-1] - x_pred[:,s:-1]), p=2, dim=1 )**2
    recon_err = recon_err / torch.norm(x, p=1, dim=1)
        
#     for i in range(1,mc_samples):
#         x_pred = model.sample_latent_code(dm[i], dv[i])        
#         recon_err += -torch.sum( torch.abs(x[:,s:-1] - x_pred[:,s:-1]), axis=1 )
#     recon_err = recon_err / mc_samples
    return -recon_err 

def compute_im1( x, x_cf, y, y_t, ae_models ):
    
    '''
    || xcf- AE_t(xcf) || / || xcf - AE_o(xcf) ||
    '''
    
    cf_score= np.zeros((x_cf.shape[0]))
    for idx in range(x_cf.shape[0]):
        x_i= x_cf[idx, :]

        #Comptuting score for counterfactual with target class autoencoder        
        y_i= int(y_t[idx])
        model= ae_models[y_i]
        cf_score[idx]= ae_reconstruct_loss_im1( model, x_i ).detach().cpu().numpy()

        #Comptuting score for counterfactual with original class autoencoder    
        y_i= int(y[idx])
        model= ae_models[y_i]
        cf_score[idx]= cf_score[idx]/ae_reconstruct_loss_im1( model, x_i ).detach().cpu().numpy()
    
    return np.mean(cf_score)

def compute_im2( x, x_cf, y, y_t, ae_models ):
    
    '''
    || AE_t(xcf) - AE(xcf) || / |xcf|
    '''
    
    cf_score= np.zeros((x_cf.shape[0]))
    for idx in range(x_cf.shape[0]):
        x_i= x_cf[idx, :]

        #Comptuting score for counterfactual with target class autoencoder        
        y_i= int(y_t[idx])
        model= ae_models[y_i]
        # The last model in the list is the all class trained auto encoder
        model_all= ae_models[-1]
        cf_score[idx]= ae_reconstruct_loss_im2( model, model_all, x_i ).detach().cpu().numpy()

    return np.mean(cf_score)

def train(model, train_dataset, optimizer, validity_reg, margin, epochs=1000, batch_size=1024):
    batch_num=0
    train_loss=0.0
    train_size=0
    #train_dataset= np.array_split( train_dataset, train_dataset.shape[0]//batch_size ,axis=0 )
    train_dataset= torch.tensor( train_dataset ).float().to(cuda)
    train_dataset= torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     for i in range(len(train_dataset)):
    for train_x in enumerate(train_dataset):
        optimizer.zero_grad()
#         train_x = train_dataset[i]
        train_x= train_x[1]
    
        train_y = torch.argmax(pred_model(train_x), dim=1)
        train_y = generate_target_labels( train_y )
        # Generating CF with target class as one ahead of the original class
        #train_y = (1.0+torch.argmax( pred_model(train_x), dim=1 ))%10
        
        train_size += train_x.shape[0]

        out= model(train_x, train_y)
        loss = compute_loss(model, out, train_x, train_y, validity_reg, margin)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()                               

        batch_num+=1

    ret= loss/batch_num
    print('Train Avg Loss: ', ret, train_size)
    return ret

def test(model, train_dataset, auto_encoder_models, eval_time, epochs=1, batch_size=2048):
    batch_num=0
    likelihood=0.0
    valid_cf_count=0
    train_size=0
    batch_size=32
    train_dataset= np.array_split( train_dataset, train_dataset.shape[0],axis=0 )
    index=random.randrange(0,len(train_dataset),1)

    eval_res={}
    time=eval_time
    for i in range(len(train_dataset)):

        train_x = train_dataset[i]
        train_x= torch.tensor( train_x ).float().to(cuda)
        # Generate counterfatuals for a specific class        
        train_y = torch.argmax( pred_model(train_x), dim=1 )                
        target_class = generate_target_labels( train_y )
        train_size += train_x.shape[0]        
        
        #Forward Pass of the model
        out= model.forward(train_x, target_class)

        wrapped = wrapper(model.forward, train_x, target_class)
        time+= timeit.timeit(wrapped, number=1) 
        eval_res[str(i)]= time
        print('-----------------------------------')
        print('Time taken: ', eval_res[str(i)])
        print('-----------------------------------')        
        
    return eval_res

#Argparsing
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='bn1')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--validity_reg', type=float, default=20)
parser.add_argument('--margin', type=float, default=0.2)
parser.add_argument('--htune', type=int, default=0)
args = parser.parse_args()

#Main Code
base_data_dir='data/'
if args.htune:
    base_model_dir='htune/base/'
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
    
elif dataset_name=='sangiovese':
    dataset = pd.DataFrame.from_csv(  base_data_dir + dataset_name + '.csv', index_col=None )
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
    l.remove('Treatment')
    l.remove('outcome')

    params= {'dataframe':dataset.copy(), 'continuous_features':l, 'outcome_name':'outcome'}
    d = DataLoader(params)
    

#Load Train, Val, Test Dataset
vae_train_dataset= np.load(base_data_dir+dataset_name+'-train-set.npy')
vae_val_dataset= np.load(base_data_dir+dataset_name+'-val-set.npy')

#Removing data from selected digits
# vae_train_dataset= np.delete( vae_train_dataset, np.where( vae_train_dataset[:,-1] == 1 ) , axis=0 )
# vae_train_dataset= np.delete( vae_train_dataset, np.where( vae_train_dataset[:,-1] == 6 ) , axis=0 )
# vae_train_dataset= np.delete( vae_train_dataset, np.where( vae_train_dataset[:,-1] == 7 ) , axis=0 )
# vae_train_dataset= np.delete( vae_train_dataset, np.where( vae_train_dataset[:,-1] == 9 ) , axis=0 )

# vae_val_dataset= np.delete( vae_val_dataset, np.where( vae_val_dataset[:,-1] == 1 ) , axis=0 )
# vae_val_dataset= np.delete( vae_val_dataset, np.where( vae_val_dataset[:,-1] == 6 ) , axis=0 )
# vae_val_dataset= np.delete( vae_val_dataset, np.where( vae_val_dataset[:,-1] == 7 ) , axis=0 )
# vae_val_dataset= np.delete( vae_val_dataset, np.where( vae_val_dataset[:,-1] == 9 ) , axis=0 )
print(np.unique(vae_train_dataset[:,-1], return_counts=True), np.unique(vae_val_dataset[:,-1], return_counts=True))

vae_train_dataset= vae_train_dataset[:,:-1]
vae_val_dataset= vae_val_dataset[:,:-1]
print( vae_train_dataset.shape, vae_val_dataset.shape )

if dataset_name!='mnist':
    with open(base_data_dir+dataset_name+'-normalise_weights.json') as f:
        normalise_weights= json.load(f)
    normalise_weights = {int(k):v for k,v in normalise_weights.items()}

    with open(base_data_dir+dataset_name+'-mad.json') as f:
        mad_feature_weights= json.load(f)

    print(normalise_weights)
    print(mad_feature_weights)

#Load Black Box Prediction Model
data_size= 28 #MNIST Images as 28*28
encoded_size=10
pred_model= BlackBox().to(cuda)
path= base_model_dir + dataset_name +'.pth'
pred_model.load_state_dict(torch.load(path))
pred_model.eval()

# Loading all the Auto Encoder models
auto_encoder_models= []
for i in range(0,10):
    ae= AutoEncoder(data_size, encoded_size).to(cuda)
    path= base_model_dir + dataset_name + '-32-50-target-class-' + str(i) + '-auto-encoder.pth'
    ae.load_state_dict(torch.load(path))
    ae.eval()
    auto_encoder_models.append(ae)
# Adding the all data trained auto encoder
ae= AutoEncoder(data_size, encoded_size).to(cuda)
path= base_model_dir + dataset_name + '-32-50-target-class-' + str(-1) + '-auto-encoder.pth'
ae.load_state_dict(torch.load(path))
ae.eval()
auto_encoder_models.append(ae)

    
# Initiliase new model
wm1=1e-2
wm2=1e-2
wm3=1e-2
wm4=1e-2

cf_vae = CF_VAE(data_size, encoded_size).to(cuda)
learning_rate = 2*1e-2
batch_size= args.batch_size
marign= args.margin
cf_vae_optimizer = optim.Adam([
    {'params': filter(lambda p: p.requires_grad, cf_vae.encoder_mean_conv.parameters()),'weight_decay': wm1},
    {'params': filter(lambda p: p.requires_grad, cf_vae.encoder_mean_fc.parameters()),'weight_decay': wm1},       
    {'params': filter(lambda p: p.requires_grad, cf_vae.encoder_var_conv.parameters()),'weight_decay': wm2},
    {'params': filter(lambda p: p.requires_grad, cf_vae.encoder_var_fc.parameters()),'weight_decay': wm2},
    {'params': filter(lambda p: p.requires_grad, cf_vae.decoder_mean_conv.parameters()),'weight_decay': wm3},
    {'params': filter(lambda p: p.requires_grad, cf_vae.decoder_mean_fc.parameters()),'weight_decay': wm3},
    ], lr=learning_rate)

#Train CFVAE
loss_val = []
likelihood_val = []
valid_cf_count = []

validity_reg=args.validity_reg

#traverse(vae_train_dataset, 1, len(vae_train_dataset))

#Evaluation Time
eval_time=0.0
for epoch in range(args.epoch):
    np.random.shuffle(vae_train_dataset)
    wrapped = wrapper(train, cf_vae, vae_train_dataset, cf_vae_optimizer, validity_reg, 1, batch_size)
    eval_time+= timeit.timeit(wrapped, number=1)
    print('-----------------------------------')
    print('Time taken: ', eval_time)
    print('-----------------------------------')

encoded_size=10
path= base_model_dir + 'mnist-margin-0.05-validity_reg-100.0-epoch-150-base-gen.pth'
cf_vae = CF_VAE(data_size, encoded_size).to(cuda)
cf_vae.load_state_dict(torch.load(path))
cf_vae.eval()
cf_vae_optimizer = optim.Adam([
    {'params': filter(lambda p: p.requires_grad, cf_vae.encoder_mean_conv.parameters()),'weight_decay': wm1},
    {'params': filter(lambda p: p.requires_grad, cf_vae.encoder_mean_fc.parameters()),'weight_decay': wm1},       
    {'params': filter(lambda p: p.requires_grad, cf_vae.encoder_var_conv.parameters()),'weight_decay': wm2},
    {'params': filter(lambda p: p.requires_grad, cf_vae.encoder_var_fc.parameters()),'weight_decay': wm2},
    {'params': filter(lambda p: p.requires_grad, cf_vae.decoder_mean_conv.parameters()),'weight_decay': wm3},
    {'params': filter(lambda p: p.requires_grad, cf_vae.decoder_mean_fc.parameters()),'weight_decay': wm3},
    ], lr=learning_rate)
    
print('Test Score: ')
#eval_time=  172.95152673497796
eval_res= test( cf_vae, vae_val_dataset, auto_encoder_models, eval_time, 1, batch_size)
print(eval_res)
          
#Saving the final model
f=open(base_data_dir + dataset_name + '-time-eval-base-gen-cf.json', 'w')
f.write( json.dumps(eval_res) )
f.close()