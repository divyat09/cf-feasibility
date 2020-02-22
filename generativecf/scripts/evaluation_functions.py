#Important Import
from scripts.vae_model import CF_VAE
from scripts.vae_model import AutoEncoder
# from scripts.vae_model_sangiovese import CF_VAE as CF_VAE_BN

#Normie stuff
import sys
import random
import pandas as pd
import numpy as np
import json
import argparse
import matplotlib.pyplot as plt

#Pytorch
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable

# Tensorflow libraries
import tensorflow as tf
from tensorflow import keras
#Keras
from tensorflow.keras.layers import Input, Dense, Lambda, Layer, Add, Multiply
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend as K

#Seed for repoduability
torch.manual_seed(10000000)

offset=0.0

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

def nll(y_true, y_pred):
    """ Negative log likelihood (Bernoulli). """

    # keras.losses.binary_crossentropy gives the mean
    # over the last axis. we require the sum
    return K.sum(keras.losses.mean_squared_error(y_true, y_pred), axis=-1)
    
def de_normalise( x, normalise_weights ):
    return (normalise_weights[1] - normalise_weights[0])*x + normalise_weights[0]

def de_scale( x, normalise_weights ):
    return (normalise_weights[1] - normalise_weights[0])*x

def normal_likelihood(x, mean, logvar, raxis=1):
    return torch.sum( -.5 * ((x - mean)*(1./logvar)*(x-mean) + torch.log(logvar) ), axis=1)

def change_direction( x_true, x_pred, idx, key, offset ):        
    if x_pred.iloc[idx, key] - offset > x_true.iloc[idx, key]:
        return 1
    elif x_pred.iloc[idx, key] + offset < x_true.iloc[idx, key]:
        return 0    

def visualize_score(model, pred_model, train_dataset, d):
        
    valid_score_arr=[]
    invalid_score_arr=[]
    for sample_size in [5]:
        train_x= torch.tensor( train_dataset ).float() 
        train_y= torch.argmax( pred_model(train_x), dim=1 )                
        valid_change= 0
        invalid_change=0 
        
        for sample_iter in range(sample_size):        
            recon_err, kl_err, x_true, x_pred, cf_label = model.compute_elbo( train_x, 1.0-train_y, pred_model )            
            
            x_pred= d.de_normalize_data( d.get_decoded_data(x_pred.detach().numpy()) )
            x_true= d.de_normalize_data( d.get_decoded_data(x_true.detach().numpy()) )                

            print(x_pred)

def contrastive_validity_score(contrastive_exp, case, sample_range):
        
    validity_score_arr=[]
    total_test_size=0
    for sample_size in sample_range:        
        valid_cf_count= 0
        for idx in range(sample_size):
            exp_res= contrastive_exp[idx]                       
            for explanation in exp_res:
                if explanation['cf_label'] != explanation['label']:
                    valid_cf_count += 1
                
        test_size= len(exp_res)
        valid_cf_count=valid_cf_count/sample_size
        total_test_size+= test_size
        print( sample_size, test_size )
        validity_score_arr.append(100*valid_cf_count/test_size)
        
    if case:
        plt.plot(sample_range, validity_score_arr)
        plt.title('Valid CF')
        plt.xlabel('Sample Size')
        plt.ylabel('Percentage of CF')
        plt.show()
    print('Mean Validity Score: ', np.mean(np.array(validity_score_arr)) )
#     if total_test_size < 3*90:
#         validity_score_arr= 0*validity_score_arr
    return validity_score_arr

def contrastive_proximity_score(contrastive_exp, d, mad_feature_weights, cat, case, sample_range):
    
    prox_score_arr=[]
    for sample_size in sample_range:
        prox_count=0
        for idx in range(sample_size):
            exp_res= contrastive_exp[idx]
            x_pred=[]
            x_true=[]
            for explanation in exp_res:
                x_pred.append(explanation['x_cf'][0])
                x_true.append(explanation['x'][0])
            
            x_pred= d.de_normalize_data( d.get_decoded_data(np.array(x_pred)) )
            x_true= d.de_normalize_data( d.get_decoded_data(np.array(x_true) ))                            
            if cat:
                for column in d.categorical_feature_names:
                    prox_count += np.sum( np.array(x_true[column], dtype=pd.Series) != np.array(x_pred[column], dtype=pd.Series ))
            else:
                for column in d.continuous_feature_names:
                    prox_count += np.sum(np.abs(x_true[column] - x_pred[column]))/mad_feature_weights[column]                
                    
        test_size= len(exp_res)
        prox_count= prox_count/sample_size
        prox_score_arr.append( -1*prox_count/test_size )

    if case:
        plt.plot(sample_range, prox_score_arr)
        if cat:
            plt.title('Categorical Proximity')
        else:
            plt.title('Continuous Proximity')

        plt.xlabel('Sample Size')
        plt.ylabel('Magnitude')
        plt.show()
    print('Mean Proximity Score: ', np.mean(np.array(prox_score_arr)) )
    return prox_score_arr

def contrastive_causal_score_age_ed_constraint(contrastive_exp, d, normalise_weights, offset, case, sample_range):
        
    valid_score_arr=[]
    invalid_score_arr=[]
    count1=0
    count2=0
    count3=0
    pos1=0
    pos2=0
    pos3=0
    for sample_size in sample_range:
        valid_change= 0
        invalid_change=0 
        test_size=0
        for idx in range(sample_size):
            exp_res= contrastive_exp[idx]                       
            x_pred=[]
            x_true=[]
            for explanation in exp_res:
                #Only Low to High Counterfactuals
                if explanation['label']==0:
                    test_size+=1
                    x_pred.append(explanation['x_cf'][0])
                    x_true.append(explanation['x'][0])
                
            x_pred= d.de_normalize_data( d.get_decoded_data(np.array(x_pred)) )
            x_true= d.de_normalize_data( d.get_decoded_data(np.array(x_true) ))                            
   
            ed_idx = x_true.columns.get_loc('education')            
            age_idx = x_true.columns.get_loc('age')            

            for i in range(x_true.shape[0]): 
                if education_score[ x_pred.iloc[i,ed_idx] ] < education_score[ x_true.iloc[i,ed_idx] ]:
                    count3+=1
                    invalid_change +=1       
                elif education_score[ x_pred.iloc[i,ed_idx] ] == education_score[ x_true.iloc[i,ed_idx] ]:
                    count1+=1
                    if x_pred.iloc[i, age_idx] - de_scale( offset, normalise_weights[0]) >= x_true.iloc[i, age_idx]:
                        pos1+=1
                        valid_change += 1
                    else:
                        invalid_change +=1                    
                elif education_score[ x_pred.iloc[i,ed_idx] ] > education_score[ x_true.iloc[i,ed_idx] ]:
                    count2+=1
                    if x_pred.iloc[i, age_idx] - de_scale( offset, normalise_weights[0]) > x_true.iloc[i, age_idx]:
                        pos2+=1
                        valid_change += 1
                    else:
                        invalid_change +=1

        valid_change= valid_change/sample_size
        invalid_change= invalid_change/sample_size
        
#         test_size= len(exp_res)
        test_size= test_size/sample_size
        print('Test Size: ', test_size)    
        valid_score_arr.append( 100*valid_change/test_size )
        invalid_score_arr.append( 100*invalid_change/test_size )

    valid_score= np.mean(np.array(valid_score_arr))
    invalid_score= np.mean(np.array(invalid_score_arr))

    if case:
        plt.plot(sample_range, valid_score_arr, '*', label='Val Change')
        plt.plot(sample_range, invalid_score_arr, 's', label='Inval Change')
        plt.legend(loc='upper left')
        plt.ylim(ymin=0, ymax=100)
        plt.title('All Education Levels')
        plt.xlabel('Sample Size')
        plt.ylabel('Percentage of CF')
        plt.show()
    print('Mean Age-Ed Constraint Score: ', valid_score, invalid_score, valid_score/(valid_score+invalid_score))
    print('Count: ', count1, count2, count3, count1+count2+count3)
    print('Pos Count: ', pos1, pos2, pos3 )
    if count1 and count2 and count3:
        print('Pos Percentage: ', pos1/count1, pos2/count2, pos3/count3 )
        
    return valid_score_arr, invalid_score_arr


def contrastive_causal_score_age_constraint(contrastive_exp, d, normalise_weights, offset, case, sample_range):
        
    valid_score_arr=[]
    invalid_score_arr=[]
    for sample_size in sample_range:
        valid_change= 0
        invalid_change=0 
        test_size=0
        for idx in range(sample_size):
            exp_res= contrastive_exp[idx]                       
            x_pred=[]
            x_true=[]
            for explanation in exp_res:
                #Only Low to High Counterfactuals
                if explanation['label']==0:
                    test_size+=1
                    x_pred.append(explanation['x_cf'][0])
                    x_true.append(explanation['x'][0])                
                
            x_pred= d.de_normalize_data( d.get_decoded_data(np.array(x_pred)) )
            x_true= d.de_normalize_data( d.get_decoded_data(np.array(x_true) ))                            
            
            age_idx = x_true.columns.get_loc('age')            
            for i in range(x_true.shape[0]): 
                if x_pred.iloc[i, age_idx] - de_scale( offset, normalise_weights[0] ) >= x_true.iloc[i, age_idx]:
                    valid_change+=1
                else:
                    invalid_change+=1
                        
        valid_change= valid_change/sample_size
        invalid_change= invalid_change/sample_size

#         test_size= len(exp_res)
        test_size= test_size/sample_size
        valid_score_arr.append( 100*valid_change/test_size )
        invalid_score_arr.append( 100*invalid_change/test_size )

    valid_score= np.mean(np.array(valid_score_arr))
    invalid_score= np.mean(np.array(invalid_score_arr))

    if case:
        plt.plot(sample_range, valid_score_arr, '*', label='Val Age Change')
        plt.plot(sample_range, invalid_score_arr, 's', label='Inval Age Change')
        plt.legend(loc='upper left')
        plt.ylim(ymin=0, ymax=100)
        plt.title('Change in Age')
        plt.xlabel('Sample Size')
        plt.ylabel('Percentage of CF')
        plt.show()    
    print('Mean Age Constraint Score: ', valid_score, invalid_score, valid_score/(valid_score+invalid_score))
    return valid_score_arr, invalid_score_arr

def contrastive_causal_score_bn1_constraint(contrastive_exp, d, normalise_weights, offset, case, sample_range):
        
    valid_score_p_arr=[]
    invalid_score_p_arr=[]
    valid_score_n_arr=[]
    invalid_score_n_arr=[]

    for sample_size in sample_range:
        valid_change_p= 0
        invalid_change_p=0
        valid_change_n=0
        invalid_change_n=0
        for idx in range(sample_size):
            exp_res= contrastive_exp[idx]                       
            x_pred=[]
            x_true=[]
            for explanation in exp_res:
                x_pred.append(explanation['x_cf'][0])
                x_true.append(explanation['x'][0])
            
            x_pred= d.de_normalize_data( d.get_decoded_data(np.array(x_pred)) )
            x_true= d.de_normalize_data( d.get_decoded_data(np.array(x_true)) )                

            x1_idx = x_true.columns.get_loc('x1')            
            x2_idx = x_true.columns.get_loc('x2')            
            x3_idx = x_true.columns.get_loc('x3')            
            for i in range(x_true.shape[0]):
                if x_pred.iloc[i, x1_idx] < x_true.iloc[i, x1_idx] and x_pred.iloc[i,x2_idx] < x_true.iloc[i,x2_idx]:
                    if x_pred.iloc[i, x3_idx] + de_scale(offset, normalise_weights[2]) < x_true.iloc[i, x3_idx]:
                        valid_change_n+=1
                    else:
                        invalid_change_n += 1
                if x_pred.iloc[i, x1_idx] > x_true.iloc[i, x1_idx] and x_pred.iloc[i,x2_idx] > x_true.iloc[i,x2_idx]:
                    if x_pred.iloc[i, x3_idx] - de_scale(offset, normalise_weights[2]) > x_true.iloc[i, x3_idx]:
                        valid_change_p+=1
                    else:
                        invalid_change_p += 1                        

        valid_change_p= valid_change_p/sample_size
        invalid_change_p= invalid_change_p/sample_size
        valid_change_n= valid_change_n/sample_size
        invalid_change_n= invalid_change_n/sample_size
        
#         print(sample_size, low_change, valid_high_change, invalid_high_change)
        test_size= len(exp_res)
        valid_score_p_arr.append( 100*valid_change_p/test_size )
        invalid_score_p_arr.append( 100*invalid_change_p/test_size )
        valid_score_n_arr.append( 100*valid_change_n/test_size )
        invalid_score_n_arr.append( 100*invalid_change_n/test_size )

    valid_score_p_arr= np.array(valid_score_p_arr)
    invalid_score_p_arr= np.array(invalid_score_p_arr)
    valid_score_n_arr= np.array(valid_score_n_arr)
    invalid_score_n_arr= np.array(invalid_score_n_arr)
        
    score_p_arr= valid_score_p_arr/(valid_score_p_arr + invalid_score_p_arr )
    score_n_arr= valid_score_n_arr/(valid_score_n_arr + invalid_score_n_arr )
    score_p= np.mean(score_p_arr)
    score_n= np.mean(score_n_arr)
        
    if case:
        plt.plot(sample_range, valid_score_arr, '*', label='Val Change')
        plt.plot(sample_range, invalid_score_arr, 's', label='Inval Change')
        plt.legend(loc='upper left')
        plt.ylim(ymin=0, ymax=100)
        plt.title('Change in x3')
        plt.xlabel('Sample Size')
        plt.ylabel('Percentage of CF')
        plt.show()    

    print('Mean Monotonic Constraint Score: ', score_p, score_n, 2*score_p*score_n/(score_p+score_n))
    return score_p_arr, score_n_arr 


def contrastive_distribution_score( contrastive_exp, d, normalise_weights, case, sample_range):

    score_arr=[]
    for sample_size in sample_range:
        score= 0
        for idx in range(sample_size):
            exp_res= contrastive_exp[idx]                       
            x_pred=[]
            x_true=[]
            for explanation in exp_res:
                x_pred.append(explanation['x_cf'][0])
                x_true.append(explanation['x'][0])
            
            x_pred= torch.tensor( np.array(x_pred) ).float()
            x_true= torch.tensor( np.array(x_true) ).float()
            
            score+= oracle_score( x_pred, normalise_weights ) / oracle_score( x_true, normalise_weights )    
        score= score/sample_size
        score_arr.append( score )

    if case:
        plt.plot(sample_range, score_arr, '*')
        plt.legend(loc='upper left')
        plt.title('Likelihood Score')
        plt.xlabel('Sample Size')
        plt.ylabel('Likelihood Score')
        plt.show()    
    print('Mean Distribution Constraint Score: ', np.mean(np.array(score_arr)))
    return score_arr

def contrastive_causal_graph_score( contrastive_exp, d, normalise_weights, case, sample_range):

    score_arr=[]
    for sample_size in sample_range:
        score= 0
        for idx in range(sample_size):
            exp_res= contrastive_exp[idx]                       
            x_pred=[]
            x_true=[]
            for explanation in exp_res:
                x_pred.append(explanation['x_cf'][0])
                x_true.append(explanation['x'][0])
            
            x_pred= torch.tensor( np.array(x_pred) ).float()
            
            score+= oracle_causal_graph_score( x_pred, normalise_weights )
        score= score/sample_size
        score_arr.append( score )

    if case:
        plt.plot(sample_range, score_arr, '*')
        plt.legend(loc='upper left')
        plt.title('Causal Graph Score')
        plt.xlabel('Sample Size')
        plt.ylabel('Causal Graph Score')
        plt.show()    
    print('Mean Causal Graph Score: ',np.mean(np.array(score_arr)))    
    return score_arr


def contrastive_causal_score_bnlearn_constraint(contrastive_exp, d, normalise_weights, offset, scm_model, constraint_nodes, plot_case, sample_range):
        
    valid_score_p_arr=[]
    invalid_score_p_arr=[]
    valid_score_n_arr=[]
    invalid_score_n_arr=[]

    for sample_size in sample_range:
        valid_change_p= 0
        invalid_change_p=0
        valid_change_n=0
        invalid_change_n=0
        
        test_size=0
        for idx in range(sample_size):
            exp_res= contrastive_exp[idx]                       
            x_pred=[]
            x_true=[]
            for explanation in exp_res:
                x_pred.append(explanation['x_cf'][0])
                x_true.append(explanation['x'][0])
            
            x_pred= d.de_normalize_data( d.get_decoded_data(np.array(x_pred)) )
            x_true= d.de_normalize_data( d.get_decoded_data(np.array(x_true)) )
            
            # Monotonic Changes
            for node in constraint_nodes:
                parents= scm_model[node]['parent']
                if 'Treatment' in parents:
                    parents.remove('Treatment')
                for idx in range(x_true.shape[0]):
                    sign=-1
                    for p_idx in range(len(parents)):
                        key= x_true.columns.get_loc(parents[p_idx])
                        if p_idx==0:
                            sign= change_direction( x_true, x_pred, idx, key, 0 ) 
                        else:
                            new_sign= change_direction( x_true, x_pred, idx, key, 0 ) 
                            if sign!=new_sign:
                                sign=-1
                                break
                                
                    # No Monotonic change in all parents, go to next CF 
                    if sign==-1:
                        continue
                    # Check whether Monotonic change of all parents is consistent with the node
                    else:
                        key= x_true.columns.get_loc(node)
                        new_sign= change_direction( x_true, x_pred, idx, key, offset ) 

                        if sign==0:
                            if sign==new_sign:
                                valid_change_n+=1
                            else:
                                invalid_change_n+=1
                            test_size +=1
                        elif sign==1:
                            if sign==new_sign:
                                valid_change_p+=1
                            else:
                                invalid_change_p+=1
                            test_size+=1
                        
        valid_score_p_arr.append( valid_change_p )
        invalid_score_p_arr.append( invalid_change_p )
        valid_score_n_arr.append( valid_change_n )
        invalid_score_n_arr.append( invalid_change_n )

    valid_score_p_arr= np.array(valid_score_p_arr)
    invalid_score_p_arr= np.array(invalid_score_p_arr)
    valid_score_n_arr= np.array(valid_score_n_arr)
    invalid_score_n_arr= np.array(invalid_score_n_arr)

    score_p_arr= valid_score_p_arr/(valid_score_p_arr + invalid_score_p_arr )
    score_n_arr= valid_score_n_arr/(valid_score_n_arr + invalid_score_n_arr )
    score_p= np.mean(score_p_arr)
    score_n= np.mean(score_n_arr)
        
    if plot_case:
        plt.plot(sample_range, valid_score_arr, '*', label='Val Change')
        plt.plot(sample_range, invalid_score_arr, 's', label='Inval Change')
        plt.legend(loc='upper left')
        plt.ylim(ymin=0, ymax=100)
        plt.title('Change in x3')
        plt.xlabel('Sample Size')
        plt.ylabel('Percentage of CF')
        plt.show()    

    print('Mean Monotonic Constraint Score: ', score_p, score_n, 2*score_p*score_n/(score_p+score_n))
    return score_p_arr, score_n_arr 
                       

def contrastive_bnlearn_causal_graph_score( contrastive_exp, d, normalise_weights, scm_model, constraint_nodes, constraint_case, plot_case, sample_range ):

    score_arr=[]
    
    for sample_size in sample_range:
        score= 0
        test_size=0
        for idx in range(sample_size):
            exp_res= contrastive_exp[idx]                       
            x_pred=[]
            x_true=[]
            for explanation in exp_res:
                x_pred.append(explanation['x_cf'][0])
                x_true.append(explanation['x'][0])
            
            x_pred= torch.tensor(np.array(x_pred)).float()
            x_true= torch.tensor( np.array(x_true) ).float()
            
            score+= bnlearn_scm_score( x_pred, normalise_weights, d, scm_model, constraint_nodes, constraint_case ) / bnlearn_scm_score( x_true, normalise_weights, d, scm_model, constraint_nodes, constraint_case)
        
        score= score/sample_size
        score_arr.append( score )

    if constraint_case:
        title='Likelihood Score'
    else:
        title='Causal Graph Score'
        
    if plot_case:        
        plt.plot(sample_range, score_arr, '*')
        plt.legend(loc='upper left')
        plt.title(title)
        plt.xlabel('Sample Size')
        plt.ylabel(title)
        plt.show()    
    print( title, ': ', np.mean(np.array(score_arr)))
    return score_arr

def ae_reconstruct_loss_cem_im( model, x, dataset_name, normalise_weights ): 
    
    #Reconstruction Term
    x_pred = model(x)       
    if dataset_name == 'adult':
        s=2
    elif dataset_name == 'sangiovese' or dataset_name=='bn1':
        s=0
        
    recon_err = -K.sum( K.square(x[:,s:-1] - x_pred[:,s:-1]), axis=-1 )
    for key in normalise_weights.keys():
        recon_err+= -(normalise_weights[key][1] - normalise_weights[key][0])*((x[:,key] - x_pred[:,key])**2) 

    return -recon_err 

def contrastive_im_score(contrastive_exp, d, normalise_weights, ae_models, dataset_name, div_case, case, sample_range):
        
    im_score_arr=[]
    total_test_size=0
    for sample_size in sample_range:        
        im_score= 0.0
        test_size=0
        for idx in range(sample_size):
            exp_res= contrastive_exp[idx]         
            x_pred=[]
            x_true=[]
            train_y=[]
            for explanation in exp_res:
                if dataset_name == 'adult' and explanation['label']==1:
                    #Only Low to High Counterfactuals
                    continue
                    
                test_size +=1
                x_pred.append(explanation['x_cf'][0])
                x_true.append(explanation['x'][0])
                train_y.append(explanation['label'])
            
            #x_pred= torch.tensor(np.array(x_pred)).float()
            #x_true= torch.tensor( np.array(x_true) ).float()
            x_pred= np.array(x_pred)
            x_true= np.array(x_true)
            train_y= np.array(train_y)
            # Assuming binary outcome case with target class opposite of original class 
            target_label= 1 - train_y            
            org_label= train_y
            #print( 'CEM: ', np.unique(org_label, return_counts=True), np.unique(target_label, return_counts=True))
            #print( 'DataSize:', test_size )
            cf_score= np.zeros((x_pred.shape[0]))
            
            #Assuming Binary Outcome Variable
            for t_c in range(0,2):
                if np.sum(target_label==t_c)==0:
                    continue                
                x_i= K.constant(x_pred[target_label==t_c,:])
                ae_model= ae_models[t_c]
                #print('Num:', np.mean(K.eval( K.sum( K.square(x_i - ae_model(x_i)), axis=-1 )) ))         
                #cf_score[target_label==t_c]= K.eval( K.sum( K.square(x_i - ae_model(x_i)), axis=-1 ) )
                cf_score[target_label==t_c]= K.eval( ae_reconstruct_loss_cem_im( ae_model, x_i, dataset_name, normalise_weights)  )
                
                if div_case:
                    ae_model= ae_models[1-t_c]
                    #print('Denom:', np.mean(K.eval( K.sum( K.square(x_i - ae_model(x_i)), axis=-1 )) ))
                    #cf_score[target_label==t_c]= cf_score[target_label==t_c]/ K.eval( K.sum( K.square(x_i - ae_model(x_i)), axis=-1 ) )
                    cf_score[target_label==t_c]= cf_score[target_label==t_c]/ K.eval( ae_reconstruct_loss_cem_im( ae_model, x_i, dataset_name, normalise_weights) )
                                    
            '''
            
            #Assuming Binary Outcome Variable
            for t_c in range(0,2):
                if np.sum(target_label==t_c)==0:
                    continue
                x_i= x_pred[target_label==t_c,:]
                ae_model= ae_models[t_c]
                print('Num:', np.mean(ae_reconstruct_loss_im( ae_model, x_i, normalise_weights ).detach().cpu().numpy()))                
                cf_score[target_label==t_c]= ae_reconstruct_loss_im( ae_model, x_i, normalise_weights ).detach().cpu().numpy()
                
                if div_case:
                    ae_model= ae_models[1-t_c]
                    print('Denom:', np.mean(ae_reconstruct_loss_im( ae_model, x_i, normalise_weights ).detach().cpu().numpy()))
                    cf_score[target_label==t_c]= cf_score[target_label==t_c]/ae_reconstruct_loss_im( ae_model, x_i, normalise_weights ).detach().cpu().numpy()
            '''
            
            '''
            for idx in range(x_pred.shape[0]):
                x_i= x_pred[idx, :]

                #Comptuting score for counterfactual with target class autoencoder        
                y_i= int(target_label[idx])
                ae_model= ae_models[y_i]
                cf_score[idx]= ae_reconstruct_loss_im( ae_model, x_i, normalise_weights ).detach().cpu().numpy()

                #Comptuting score for counterfactual with original class autoencoder    
                y_i= int(org_label[idx])
                ae_model= ae_models[y_i]
                cf_score[idx]= cf_score[idx]/ae_reconstruct_loss_im( ae_model, x_i, normalise_weights ).detach().numpy()
            '''
            
            im_score += np.mean(cf_score)
            
        im_score=im_score/sample_size
        im_score_arr.append(im_score)

    if case:
        plt.plot(sample_range, im_score_arr)
        plt.title('Metric IM')
        plt.xlabel('Sample Size')
        plt.ylabel('IM Score')
        plt.show()
    print('Mean IM Score: ', np.mean(np.array(im_score_arr)) )
    return im_score_arr
    

def validity_score(model, pred_model, train_dataset, case, sample_range):
        
    validity_score_arr=[]
    for sample_size in sample_range:
        train_x= torch.tensor( train_dataset ).float() 
        train_y = torch.argmax( pred_model(train_x), dim=1 )                
        valid_cf_count=0
        for sample_iter in range(sample_size):        
            recon_err, kl_err, x_true, x_pred, cf_label = model.compute_elbo( train_x, 1.0-train_y, pred_model )
            cf_label = cf_label.numpy()
            valid_cf_count += np.sum( train_y.numpy() != cf_label )
            
        test_size= train_x.shape[0]
        valid_cf_count=valid_cf_count/sample_size
        validity_score_arr.append(100*valid_cf_count/test_size)

    if case:
        plt.plot(sample_range, validity_score_arr)
        plt.title('Valid CF')
        plt.xlabel('Sample Size')
        plt.ylabel('Percentage of CF')
        plt.show()
    print('Mean Validity Score: ', np.mean(np.array(validity_score_arr)) )
    return validity_score_arr

def proximity_score(model, pred_model, train_dataset, d, mad_feature_weights, cat, case, sample_range):
    
    prox_score_arr=[]
    for sample_size in sample_range:
        train_x= torch.tensor( train_dataset ).float() 
        train_y = torch.argmax( pred_model(train_x), dim=1 )                
        prox_count=0
        for sample_iter in range(sample_size):        
            recon_err, kl_err, x_true, x_pred, cf_label = model.compute_elbo( train_x, 1.0-train_y, pred_model )            
            
            x_pred= d.de_normalize_data( d.get_decoded_data(x_pred.detach().numpy()) )
            x_true= d.de_normalize_data( d.get_decoded_data(x_true.detach().numpy()) )                
#             x_pred= d.get_decoded_data(x_pred.detach().numpy()) 
#             x_true= d.get_decoded_data(x_true.detach().numpy())                 
            if cat:
                for column in d.categorical_feature_names:
#                     print(column, x_true[column].shape, x_pred[column].shape)
#                     for di in range(x_true[column].shape[0]):
#                         print(len(x_true[column][di]), len(x_pred[column][di]))
                    prox_count += np.sum( np.array(x_true[column], dtype=pd.Series) != np.array(x_pred[column], dtype=pd.Series ))
            else:
                for column in d.continuous_feature_names:
                    prox_count += np.sum(np.abs(x_true[column] - x_pred[column]))/mad_feature_weights[column]                
                    
        test_size= train_x.shape[0]
        prox_count= prox_count/sample_size
#         print(sample_size, prox_count)
        prox_score_arr.append( -1*prox_count/test_size )

    if case:
        plt.plot(sample_range, prox_score_arr)
        if cat:
            plt.title('Categorical Proximity')
        else:
            plt.title('Continuous Proximity')

        plt.xlabel('Sample Size')
        plt.ylabel('Magnitude')
        plt.show()
    print('Mean Proximity Score: ', np.mean(np.array(prox_score_arr)) )
    return prox_score_arr


def diversity_score(model, pred_model, train_dataset, d, mad_feature_weights, cat):
    
    divr_score_arr=[]
    for sample_size in [1,2,6,8,10]:
        train_x= torch.tensor( train_dataset ).float() 
        train_y = torch.argmax( pred_model(train_x), dim=1 )                
        divr_count=0        
        cf_gen=[]
        
        for sample_iter in range(sample_size):        
            recon_err, kl_err, x_true, x_pred, cf_label = model.compute_elbo( train_x, 1.0-train_y, pred_model )            
            x_pred= d.de_normalize_data( d.get_decoded_data(x_pred.detach().numpy()) )
            x_true= d.de_normalize_data( d.get_decoded_data(x_true.detach().numpy()) )                
            cf_gen.append( x_pred )

        for i in range(0, sample_size-1):
            for j in range(i+1, sample_size): 
                
                if cat:             
                    for column in d.categorical_feature_names:
                        divr_count += np.sum( np.array(cf_gen[i][column], dtype=pd.Series ) == np.array(cf_gen[j][column], dtype=pd.Series ) )
                else:
                    for column in d.continuous_feature_names:
                        divr_count += np.sum(np.abs(cf_gen[i][column] - cf_gen[j][column]))/mad_feature_weights[column]                
                
        test_size= train_x.shape[0]
        divr_count= divr_count/(sample_size**2)
#         print(sample_size, divr_count)
        divr_score_arr.append( divr_count/test_size )

    plt.plot([1,2,6,8,10], divr_score_arr)
    if cat:
        plt.title('Categorical Diversity')
    else:
        plt.title('Continuous Diversity')
        
    plt.xlabel('Sample Size')
    plt.ylabel('Magnitude')
    plt.show()
    return divr_score_arr    


def bnlearn_scm_score( xpred, normalise_weights, d, scm_model, constraint_nodes, constraint_case ):        

    score= torch.zeros( xpred.shape[0] )
    
    #Compute Likelihood for the full graph
    if constraint_case==0:
        constraint_nodes= list(scm_model.keys())
        constraint_nodes.remove( 'GrapeW' )
    
    for node in constraint_nodes:
        parents= scm_model[node]['parent']
        weights= scm_model[node]['weight']        
        deviations= scm_model[node]['sd']
        
        if len(parents)==0:
            continue
        if 'Treatment' in parents:
            # Choose the corresponding linear gaussian model based on treatment label
            # Get encoded feature indices correspodning to the feature Treatment
            
            weights= weights[2]
            deviations= deviations[2]
            mean= weights[0]
            for idx in range(len(parents)):
                if parents[idx] =='Treatment':
                    continue
                key= d.encoded_feature_names.index(parents[idx])
                w= weights[idx] # No neeed to add 1 because first parent is Treatment
                mean+= w*( de_normalise(xpred[:, key], normalise_weights[key]) )
            var= torch.zeros(xpred.shape[0],1).fill_(deviations[0]**2)
 
        else:
            #Intercept
            mean= weights[0]
            for idx in range(len(parents)):            
                key= d.encoded_feature_names.index(parents[idx])
                w= weights[idx+1] # Add 1 because the first case is intercept 
                #print(key, type(xpred))
                mean+= w*( de_normalise(xpred[:, key], normalise_weights[key]) )              
            var= torch.zeros(xpred.shape[0],1).fill_(deviations[0]**2)
            
        key= d.encoded_feature_names.index(node)
        delta_x= de_normalise(xpred[:, key], normalise_weights[key])
        #print( mean[0], delta_x[0], model.normal_likelihood(delta_x, mean, var)[0])
        score += normal_likelihood(delta_x, mean, var)
    
    return torch.mean(score).detach().numpy()    

def bnlearn_causal_graph_score(model, pred_model, train_dataset, normalise_weights, d, scm_model, constraint_nodes, constraint_case, plot_case, sample_range):

    score_arr=[]
    for sample_size in sample_range:
        train_x= torch.tensor( train_dataset ).float() 
        train_y= torch.argmax( pred_model(train_x), dim=1 )                
        score= 0
        
        for sample_iter in range(sample_size):        
            recon_err, kl_err, x_true, x_pred, cf_label = model.compute_elbo( train_x, 1.0- train_y, pred_model )                    
            score+= bnlearn_scm_score( x_pred, normalise_weights, d, scm_model, constraint_nodes, constraint_case ) / bnlearn_scm_score( x_true, normalise_weights, d, scm_model, constraint_nodes, constraint_case)
        
        score= score/sample_size
        score_arr.append( score )

    if constraint_case:
        title='Likelihood Score'
    else:
        title='Causal Graph Score'
        
    if plot_case:        
        plt.plot(sample_range, score_arr, '*')
        plt.legend(loc='upper left')
        plt.title(title)
        plt.xlabel('Sample Size')
        plt.ylabel(title)
        plt.show()    
    print( title, ': ', np.mean(np.array(score_arr)))
    return score_arr
    
def bnlearn_path_constraint_score(model, pred_model, train_dataset, d, scm_model, constraint_nodes, plot_case, sample_range):
        
    valid_score_arr=[]
    invalid_score_arr=[]
    for sample_size in sample_range:
        train_x= torch.tensor( train_dataset ).float() 
        train_y= torch.argmax( pred_model(train_x), dim=1 )                
        valid_change= 0
        invalid_change=0 
        test_size= 0
        
        for sample_iter in range(sample_size):        
            recon_err, kl_err, x_true, x_pred, cf_label = model.compute_elbo( train_x, 1.0-train_y, pred_model )            
            
            x_pred= d.de_normalize_data( d.get_decoded_data(x_pred.detach().numpy()) )
            x_true= d.de_normalize_data( d.get_decoded_data(x_true.detach().numpy()) )                
            
            # Path Changes
            for idx in range(x_true.shape[0]):
                for node_idx in range(len(constraint_nodes)):
                    key= x_true.columns.get_loc(constraint_nodes[node_idx])
                    if node_idx==0:
                        sign= change_direction( x_true, x_pred, idx, key, 0 ) 
                    else:
                        new_sign= change_direction( x_true, x_pred, idx, key, 0 ) 
                        if sign==new_sign:
                            valid_change+=1
                        else:
                            invalid_change+=1
                        test_size +=1
        
        print( test_size, train_x.shape[0])
        valid_score_arr.append( 100*valid_change/test_size )
        invalid_score_arr.append( 100*invalid_change/test_size )

    valid_score= np.mean(np.array(valid_score_arr))
    invalid_score= np.mean(np.array(invalid_score_arr))
    
    if plot_case:
        plt.plot(sample_range, valid_score_arr, '*', label='Val Change')
        plt.plot(sample_range, invalid_score_arr, 's', label='Inval Change')
        plt.legend(loc='upper left')
        plt.ylim(ymin=0, ymax=100)
        plt.title('Change in constrained node' )
        plt.xlabel('Sample Size')
        plt.ylabel('Percentage of CF')
        plt.show()    

    print('Mean Monotonic Constraint Score: ', valid_score, invalid_score, valid_score/(valid_score+invalid_score))
    return valid_score_arr, invalid_score_arr


def bnlearn_causal_score(model, pred_model, train_dataset, d, normalise_weights, offset, scm_model, constraint_nodes, plot_case, sample_range):
        
    valid_score_p_arr=[]
    invalid_score_p_arr=[]
    valid_score_n_arr=[]
    invalid_score_n_arr=[]
    
    for sample_size in sample_range:
        train_x= torch.tensor( train_dataset ).float() 
        train_y= torch.argmax( pred_model(train_x), dim=1 )                
        valid_change_p= 0
        invalid_change_p=0
        valid_change_n=0
        invalid_change_n=0
        
        test_size=0
        for sample_iter in range(sample_size):        
            recon_err, kl_err, x_true, x_pred, cf_label = model.compute_elbo( train_x, 1.0-train_y, pred_model )            
            
            x_pred= d.de_normalize_data( d.get_decoded_data(x_pred.detach().numpy()) )
            x_true= d.de_normalize_data( d.get_decoded_data(x_true.detach().numpy()) )                

            
            # Monotonic Changes
            for node in constraint_nodes:
                parents= scm_model[node]['parent']
                if 'Treatment' in parents:
                    parents.remove('Treatment')
                for idx in range(x_true.shape[0]):
                    sign=-1
                    for p_idx in range(len(parents)):
                        key= x_true.columns.get_loc(parents[p_idx])
                        if p_idx==0:
                            sign= change_direction( x_true, x_pred, idx, key, 0 ) 
                        else:
                            new_sign= change_direction( x_true, x_pred, idx, key, 0 ) 
                            if sign!=new_sign:
                                sign=-1
                                break
                                
                    # No Monotonic change in all parents, go to next CF 
                    if sign==-1:
                        continue
                    # Check whether Monotonic change of all parents is consistent with the node
                    else:
                        key= x_true.columns.get_loc(node)
                        new_sign= change_direction( x_true, x_pred, idx, key, offset ) 
                        if sign==0:
                            if sign==new_sign:
                                valid_change_n+=1
                            else:
                                invalid_change_n+=1
                            test_size +=1
                        elif sign==1:
                            if sign==new_sign:
                                valid_change_p+=1
                            else:
                                invalid_change_p+=1
                            test_size+=1
                        
        valid_score_p_arr.append( valid_change_p )
        invalid_score_p_arr.append( invalid_change_p )
        valid_score_n_arr.append( valid_change_n )
        invalid_score_n_arr.append( invalid_change_n )

    valid_score_p_arr= np.array(valid_score_p_arr)
    invalid_score_p_arr= np.array(invalid_score_p_arr)
    valid_score_n_arr= np.array(valid_score_n_arr)
    invalid_score_n_arr= np.array(invalid_score_n_arr)

    score_p_arr= valid_score_p_arr/(valid_score_p_arr + invalid_score_p_arr )
    score_n_arr= valid_score_n_arr/(valid_score_n_arr + invalid_score_n_arr )
    score_p= np.mean(score_p_arr)
    score_n= np.mean(score_n_arr)
        
    if plot_case:
        plt.plot(sample_range, valid_score_arr, '*', label='Val Change')
        plt.plot(sample_range, invalid_score_arr, 's', label='Inval Change')
        plt.legend(loc='upper left')
        plt.ylim(ymin=0, ymax=100)
        plt.title('Change in x3')
        plt.xlabel('Sample Size')
        plt.ylabel('Percentage of CF')
        plt.show()    

    print('Mean Monotonic Constraint Score: ', score_p, score_n, 2*score_p*score_n/(score_p+score_n))
    return score_p_arr, score_n_arr 


def oracle_func_approx_score(model, x, normalise_weights):    
    mean= 10*( ( de_normalise(x[:,0], normalise_weights[0]) + de_normalise(x[:,1], normalise_weights[1]) )**2/180**2) + 10
    score= torch.abs( de_normalise(x[:,2], normalise_weights[2]) - mean )
    
    return torch.mean(score).detach().numpy()


def oracle_score(x, normalise_weights):        
#     mean= 10*(( w0*x[:,0]+w1*x[:,1])**2/180**2) + 10
#     var= 0.5
#     score= ((w2*x[:,2] - mean)*(1./var)*(w2*x[:,2]-mean) )

    mean= 10*( ( de_normalise(x[:,0], normalise_weights[0]) + de_normalise(x[:,1], normalise_weights[1]) )**2/180**2) + 10
    var= torch.zeros(x.shape[0],1).fill_(0.5)
    score= normal_likelihood( de_normalise(x[:,2], normalise_weights[2]), mean, var )
    
    return torch.mean(score).detach().numpy()

def oracle_causal_graph_score(x, normalise_weights):
    
    mean= torch.zeros(x.shape[0],1).fill_(50.0)
    var= torch.zeros(x.shape[0],1).fill_(15.0)
    score= normal_likelihood( de_normalise(x[:,0], normalise_weights[0]), mean, var )

    mean= torch.zeros(x.shape[0],1).fill_(50.0)
    var= torch.zeros(x.shape[0],1).fill_(17.0)
    score+= normal_likelihood( de_normalise(x[:,1], normalise_weights[1]), mean, var )
    
    mean= 10*( ( de_normalise(x[:,0], normalise_weights[0]) + de_normalise(x[:,1], normalise_weights[1]) )**2/180**2) + 10
    var= torch.zeros(x.shape[0],1).fill_(0.5)
    score= normal_likelihood( de_normalise(x[:,2], normalise_weights[2]), mean, var )
    
    return torch.mean(score).detach().numpy()

def causal_graph_score(model,  pred_model, train_dataset, normalise_weights, case, sample_range):

    score_arr=[]
    for sample_size in sample_range:
        train_x= torch.tensor( train_dataset ).float() 
        train_y= torch.argmax( pred_model(train_x), dim=1 )                
        score= 0
        
        for sample_iter in range(sample_size):        
            recon_err, kl_err, x_true, x_pred, cf_label = model.compute_elbo( train_x, 1.0-train_y, pred_model )                    
            score+= oracle_causal_graph_score( x_pred, normalise_weights )
        
        score= score/sample_size
        score_arr.append( score )

    if case:
        plt.plot(sample_range, score_arr, '*')
        plt.legend(loc='upper left')
        plt.title('Causal Graph Score')
        plt.xlabel('Sample Size')
        plt.ylabel('Causal Graph Score')
        plt.show()    
    print('Mean Causal Graph Score: ',np.mean(np.array(score_arr)))    
    return score_arr

def distribution_score(model, pred_model, train_dataset, normalise_weights, case, sample_range):

    score_arr=[]
    for sample_size in sample_range:
        train_x= torch.tensor( train_dataset ).float() 
        train_y= torch.argmax( pred_model(train_x), dim=1 )                
        score= 0
        
        for sample_iter in range(sample_size):        
            recon_err, kl_err, x_true, x_pred, cf_label = model.compute_elbo( train_x, 1.0- train_y, pred_model )                    
            #print( oracle_score( x_pred, normalise_weights), oracle_score( x_true, normalise_weights) )
            score+= oracle_score( x_pred, normalise_weights ) / oracle_score( x_true, normalise_weights)
 
        score= score/sample_size
        score_arr.append( score )

    if case:
        plt.plot(sample_range, score_arr, '*')
        plt.legend(loc='upper left')
        plt.title('Likelihood Score')
        plt.xlabel('Sample Size')
        plt.ylabel('Likelihood Score')
        plt.show()    
    print('Mean Distribution Constraint Score: ', np.mean(np.array(score_arr)))
    return score_arr

def func_approx_score(model, pred_model, train_dataset, normalise_weights, case, sample_range):

    score_arr=[]
    for sample_size in sample_range:
        train_x= torch.tensor( train_dataset ).float() 
        train_y= torch.argmax( pred_model(train_x), dim=1 )                
        score= 0
        
        for sample_iter in range(sample_size):        
            recon_err, kl_err, x_true, x_pred, cf_label = model.compute_elbo( train_x, 1.0- train_y, pred_model )                    
            score+= oracle_func_approx_score( model, x_pred, normalise_weights )
        
        score= score/sample_size
        score_arr.append( score )

    if case:
        plt.plot(sample_range, score_arr, '*')
        plt.legend(loc='upper left')
        plt.title('Likelihood Score')
        plt.xlabel('Sample Size')
        plt.ylabel('Likelihood Score')
        plt.show()    
    print('Mean Func Approxmiation Score: ', np.mean(np.array(score_arr)))
    return score_arr

    
def causal_score(model, pred_model, train_dataset, d, normalise_weights, offset, case, sample_range):
        
    valid_score_p_arr=[]
    invalid_score_p_arr=[]
    valid_score_n_arr=[]
    invalid_score_n_arr=[]
   
    for sample_size in sample_range:
        train_x= torch.tensor( train_dataset ).float() 
        train_y= torch.argmax( pred_model(train_x), dim=1 )                
        valid_change_p= 0
        invalid_change_p=0
        valid_change_n= 0
        invalid_change_n=0
        
        for sample_iter in range(sample_size):        
            recon_err, kl_err, x_true, x_pred, cf_label = model.compute_elbo( train_x, 1.0-train_y, pred_model )            
            
            x_pred= d.de_normalize_data( d.get_decoded_data(x_pred.detach().numpy()) )
            x_true= d.de_normalize_data( d.get_decoded_data(x_true.detach().numpy()) )                

            x1_idx = x_true.columns.get_loc('x1')            
            x2_idx = x_true.columns.get_loc('x2')            
            x3_idx = x_true.columns.get_loc('x3')            
            for i in range(x_true.shape[0]): 
                
                if x_pred.iloc[i, x1_idx] < x_true.iloc[i, x1_idx] and x_pred.iloc[i,x2_idx] < x_true.iloc[i,x2_idx] :
                    if x_pred.iloc[i, x3_idx] + de_scale( offset, normalise_weights[2] ) < x_true.iloc[i, x3_idx]:
                        valid_change_n+=1
                    else:
                        invalid_change_n += 1
                if x_pred.iloc[i, x1_idx] > x_true.iloc[i, x1_idx] and x_pred.iloc[i,x2_idx] > x_true.iloc[i,x2_idx] :
                    if x_pred.iloc[i, x3_idx] - de_scale( offset, normalise_weights[2] ) > x_true.iloc[i, x3_idx]:
                        valid_change_p +=1
                    else:
                        invalid_change_p += 1                        

        valid_change_p= valid_change_p/sample_size
        invalid_change_p= invalid_change_p/sample_size
        valid_change_n= valid_change_p/sample_size
        invalid_change_n= invalid_change_p/sample_size
        
#         print(sample_size, low_change, valid_high_change, invalid_high_change)
        test_size= train_x.shape[0]
    
        valid_score_p_arr.append( 100*valid_change_p/test_size )
        invalid_score_p_arr.append( 100*invalid_change_p/test_size )
        valid_score_n_arr.append( 100*valid_change_n/test_size )
        invalid_score_n_arr.append( 100*invalid_change_n/test_size )
        
    valid_score_p_arr= np.array(valid_score_p_arr)
    invalid_score_p_arr= np.array(invalid_score_p_arr)
    valid_score_n_arr= np.array(valid_score_n_arr)
    invalid_score_n_arr= np.array(invalid_score_n_arr)

    score_p_arr= valid_score_p_arr/(valid_score_p_arr + invalid_score_p_arr )
    score_n_arr= valid_score_n_arr/(valid_score_n_arr + invalid_score_n_arr )
    score_p= np.mean(score_p_arr)
    score_n= np.mean(score_n_arr)
        
    if case:
        plt.plot(sample_range, valid_score_arr, '*', label='Val Change')
        plt.plot(sample_range, invalid_score_arr, 's', label='Inval Change')
        plt.legend(loc='upper left')
        plt.ylim(ymin=0, ymax=100)
        plt.title('Change in x3')
        plt.xlabel('Sample Size')
        plt.ylabel('Percentage of CF')
        plt.show()    

    print('Mean Monotonic Constraint Score: ', score_p, score_n, 2*score_p*score_n/(score_p+score_n))
    return score_p_arr, score_n_arr 


def causal_score_age_ed_constraint(model, pred_model, train_dataset, d, normalise_weights, offset, case, sample_range):
        
    valid_score_arr=[]
    invalid_score_arr=[]
    count1=0
    count2=0
    count3=0
    pos1=0
    pos2=0
    pos3=0
    
    for sample_size in sample_range:
        train_x= torch.tensor( train_dataset ).float() 
        train_y= torch.argmax( pred_model(train_x), dim=1 )                
        valid_change= 0
        invalid_change=0
        test_size=0
        
        for sample_iter in range(sample_size):        
            recon_err, kl_err, x_true, x_pred, cf_label= model.compute_elbo( train_x, 1.0-train_y, pred_model )            
            
            x_pred= d.de_normalize_data( d.get_decoded_data(x_pred.detach().numpy()) )
            x_true= d.de_normalize_data( d.get_decoded_data(x_true.detach().numpy()) )                

            ed_idx = x_true.columns.get_loc('education')            
            age_idx = x_true.columns.get_loc('age')            

            for i in range(x_true.shape[0]): 
                
                if cf_label[i]==0:
                    #print(cf_label[i])
                    continue                
                test_size+=1
                
                if education_score[ x_pred.iloc[i,ed_idx] ] < education_score[ x_true.iloc[i,ed_idx] ]:
                    count3+=1
                    invalid_change +=1       
                elif education_score[ x_pred.iloc[i,ed_idx] ] == education_score[ x_true.iloc[i,ed_idx] ]:
                    count1+=1
                    if x_pred.iloc[i, age_idx] - de_scale( offset, normalise_weights[0]) >= x_true.iloc[i, age_idx]:
                        pos1+=1
                        valid_change += 1
                    else:
                        invalid_change +=1                    
                elif education_score[ x_pred.iloc[i,ed_idx] ] > education_score[ x_true.iloc[i,ed_idx] ]:
                    count2+=1
                    if x_pred.iloc[i, age_idx] - de_scale( offset, normalise_weights[0]) > x_true.iloc[i, age_idx]:
                        pos2+=1
                        valid_change += 1
                    else:
                        invalid_change +=1

        valid_change= valid_change/sample_size
        invalid_change= invalid_change/sample_size
        
#         test_size= train_x.shape[0]
        test_size= test_size/sample_size
        valid_score_arr.append( 100*valid_change/test_size )
        invalid_score_arr.append( 100*invalid_change/test_size )
        print('Test Size', test_size)

    valid_score= np.mean(np.array(valid_score_arr))
    invalid_score= np.mean(np.array(invalid_score_arr))

    if case:
        plt.plot(sample_range, valid_score_arr, '*', label='Val Change')
        plt.plot(sample_range, invalid_score_arr, 's', label='Inval Change')
        plt.legend(loc='upper left')
        plt.ylim(ymin=0, ymax=100)
        plt.title('All Education Levels')
        plt.xlabel('Sample Size')
        plt.ylabel('Percentage of CF')
        plt.show()
    print('Mean Age-Ed Constraint Score: ', valid_score, invalid_score, valid_score/(valid_score+invalid_score))
    print('Count: ', count1, count2, count3, count1+count2+count3)
    print('Pos Count: ', pos1, pos2, pos3 )
    if count1 and count2 and count3:
        print('Pos Percentage: ', pos1/count1, pos2/count2, pos3/count3 )    
    return valid_score_arr, invalid_score_arr


def causal_score_age_constraint(model, pred_model, train_dataset, d, normalise_weights, offset, case, sample_range):
        
    valid_score_arr=[]
    invalid_score_arr=[]
    for sample_size in sample_range:
        train_x= torch.tensor( train_dataset ).float() 
        train_y= torch.argmax( pred_model(train_x), dim=1 )                
        valid_change= 0
        invalid_change=0 
        test_size=0
        
        for sample_iter in range(sample_size):
            recon_err, kl_err, x_true, x_pred, cf_label = model.compute_elbo( train_x, 1.0-train_y, pred_model )            
            
            x_pred= d.de_normalize_data( d.get_decoded_data(x_pred.detach().numpy()) )
            x_true= d.de_normalize_data( d.get_decoded_data(x_true.detach().numpy()) )                

            age_idx = x_true.columns.get_loc('age')            
            for i in range(x_true.shape[0]): 
                
                if cf_label[i]==0:
                    #print(cf_label[i])
                    continue                
                test_size+=1
                
                if x_pred.iloc[i, age_idx] - de_scale( offset, normalise_weights[0] ) >= x_true.iloc[i, age_idx]:
                    valid_change+=1
                else:
                    invalid_change+=1
        valid_change= valid_change/sample_size
        invalid_change= invalid_change/sample_size

#         test_size= train_x.shape[0]
        test_size= test_size/sample_size
        valid_score_arr.append( 100*valid_change/test_size )
        invalid_score_arr.append( 100*invalid_change/test_size )

    valid_score= np.mean(np.array(valid_score_arr))
    invalid_score= np.mean(np.array(invalid_score_arr))

    if case:
        plt.plot(sample_range, valid_score_arr, '*', label='Val Age Change')
        plt.plot(sample_range, invalid_score_arr, 's', label='Inval Age Change')
        plt.legend(loc='upper left')
        plt.ylim(ymin=0, ymax=100)
        plt.title('Change in Age')
        plt.xlabel('Sample Size')
        plt.ylabel('Percentage of CF')
        plt.show()    
    print('Mean Age Constraint Score: ', valid_score, invalid_score, valid_score/(valid_score+invalid_score))
    return valid_score_arr, invalid_score_arr

def ae_reconstruct_loss_im( model, x, normalise_weights ): 
    
    if len(x.shape) == 1:
        x=x.view(1,x.shape[0])
    
    model_out= model(x)
    em = model_out['em']
    ev = model_out['ev']
    z  = model_out['z']
    dm = model_out['x_pred']
    mc_samples = model_out['mc_samples']
            
    #Reconstruction Term
    x_pred = dm[0]        
    s= model.encoded_start_cat
#    s=0
    recon_err = -torch.sum( (x[:,s:-1] - x_pred[:,s:-1])**2, dim=1 )
    for key in normalise_weights.keys():
        recon_err+= -(normalise_weights[key][1] - normalise_weights[key][0])*((x[:,key] - x_pred[:,key])**2) 

    return -recon_err 

def im_score(model, pred_model, train_dataset, d, normalise_weights, ae_models, div_case, case, sample_range):
        
    im_score_arr=[]
    for sample_size in sample_range:
        train_x= torch.tensor( train_dataset ).float() 
        train_y = torch.argmax( pred_model(train_x), dim=1 )                        
        im_score=0.0
        for sample_iter in range(sample_size):        
            recon_err, kl_err, x_true, x_pred, cf_label = model.compute_elbo( train_x, 1.0-train_y, pred_model )
            
            # Assuming binary outcome case with target class opposite of original class 
            target_label= 1 - train_y.numpy()
            org_label= train_y.numpy()
            cf_label = cf_label.numpy()
            
            #print('Mine: ', np.unique(org_label, return_counts=True), np.unique(target_label, return_counts=True))
            #print('DataSize: ', train_x.shape[0] )
            cf_score= np.zeros((x_pred.shape[0]))
            
            #Assuming Binary Outcome Variable
            for t_c in range(0,2):
                if np.sum(target_label==t_c)==0:
                    continue                
                x_i= x_pred[target_label==t_c,:]
                ae_model= ae_models[t_c]
                cf_score[target_label==t_c]= ae_reconstruct_loss_im( ae_model, x_i, normalise_weights ).detach().cpu().numpy()
                #print('Num:', np.mean(ae_reconstruct_loss_im( ae_model, x_i, normalise_weights ).detach().cpu().numpy()))
                
                if div_case:
                    ae_model= ae_models[1-t_c]
                    #print('Denom:', np.mean(ae_reconstruct_loss_im( ae_model, x_i, normalise_weights ).detach().cpu().numpy()))
                    cf_score[target_label==t_c]= cf_score[target_label==t_c]/ae_reconstruct_loss_im( ae_model, x_i, normalise_weights ).detach().cpu().numpy()
                
            ''' 
            for idx in range(x_pred.shape[0]):
                x_i= x_pred[idx, :]

                #Comptuting score for counterfactual with target class autoencoder        
                y_i= int(target_label[idx])
                ae_model= ae_models[y_i]
                cf_score[idx]= ae_reconstruct_loss_im( ae_model, x_i, normalise_weights ).detach().cpu().numpy()

                #Comptuting score for counterfactual with original class autoencoder    
                y_i= int(org_label[idx])
                ae_model= ae_models[y_i]
                cf_score[idx]= cf_score[idx]/ae_reconstruct_loss_im( ae_model, x_i, normalise_weights ).detach().numpy()
            '''

            im_score += np.mean(cf_score)
            
        im_score=im_score/sample_size
        im_score_arr.append(im_score)

    if case:
        plt.plot(sample_range, im_score_arr)
        plt.title('Metric IM')
        plt.xlabel('Sample Size')
        plt.ylabel('IM Score')
        plt.show()
    print('Mean IM Score: ', np.mean(np.array(im_score_arr)) )
    return im_score_arr
    

def compute_eval_metrics_bn1( methods, base_model_dir, encoded_size, pred_model, val_dataset, d, normalise_weights, mad_feature_weights, div_case, case, sample_range, filename ):    
    count=0
    fsize=20
    fig = plt.figure(figsize=(7.7,6.5))
    final_res={}
    dataset_name= 'bn1'

    np.random.shuffle(val_dataset)
    x_sample= val_dataset[0,:]
    x_sample= np.reshape( x_sample, (1, val_dataset.shape[1]))
    print('Input Data Sample: ', d.de_normalize_data( d.get_decoded_data(x_sample) ))
    
    # Loading all the Auto Encoder models
    auto_encoder_models= []
    for i in range(0,2):
        ae= AutoEncoder(len(d.encoded_feature_names), encoded_size, d)
        path=  'models/bn1-64-50-target-class-' + str(i) + '-auto-encoder.pth'
        ae.load_state_dict(torch.load(path))
        ae.eval()
        auto_encoder_models.append(ae)
    
    # Loading all the Auto Encoder for CEM
    auto_encoder_models_cem= []
    for i in range(0,2):
        
        # load json and create model
        path='models/bn1-keras-ae-model-target-class-' + str(i)
        json_file = open(path+'.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        ae_model = tf.keras.models.model_from_json(loaded_model_json)

        # load weights into new model
        ae_model.load_weights(path+".h5")
        ae_model.compile(loss=nll, optimizer='adam')
        
        auto_encoder_models_cem.append(ae_model)
        
    
    for key in methods.keys():

        #Loading torch model
        wm1=1e-4
        wm2=1e-3
        wm3=1e-4
        wm4=1e-3
        
        path= methods[key]
        cf_val=[]
        if 'contrastive' in path and 'json' in path:
            f=open(path,'r')
            contrastive_exp= json.load(f)
            f.close()
            for i in range(10):
                if case==0:
                    cf_val.append( contrastive_validity_score(contrastive_exp, 0, sample_range) )
                elif case==1:
                    cf_val.append( contrastive_distribution_score( contrastive_exp, d, normalise_weights, 0, sample_range  ) )
                elif case==3:
                    val, inval= contrastive_causal_score_bn1_constraint(contrastive_exp, d, normalise_weights, offset, 0, sample_range)
                    cf_val.append( 2*100*np.array(val)*np.array(inval)/(np.array(val)+np.array(inval)) )
                elif case==4:
                    cf_val.append( contrastive_causal_graph_score(contrastive_exp, d, normalise_weights, 0, sample_range ) )
                elif case==5:
                    cf_val.append( contrastive_proximity_score(contrastive_exp, d, mad_feature_weights, 0, 0, sample_range) )
                elif case==6:
                    cf_val.append( contrastive_im_score(contrastive_exp, d, normalise_weights, auto_encoder_models_cem, dataset_name, div_case, 0, sample_range) )
                    
        else: 
            cf_vae = CF_VAE( len(d.encoded_feature_names), encoded_size, d )
            cf_vae.load_state_dict(torch.load(path))
            cf_vae.eval()
            learning_rate = 1e-3
            cf_vae_optimizer = optim.Adam([
                {'params': filter(lambda p: p.requires_grad, cf_vae.encoder_mean.parameters()),'weight_decay': wm1},
                {'params': filter(lambda p: p.requires_grad, cf_vae.encoder_var.parameters()), 'weight_decay': wm2},
                {'params': filter(lambda p: p.requires_grad, cf_vae.decoder_mean.parameters()),'weight_decay': wm3}
            ], lr=learning_rate)       

            if case==-1:        
                print('\n', 'Method: ', key, "\n")
                visualize_score(cf_vae, pred_model, x_sample, d)
                continue                      

            for i in range(10):
                if case==0:
                    cf_val.append( validity_score(cf_vae, pred_model, val_dataset, 0, sample_range) )
                if case==1:
                    cf_val.append( distribution_score(cf_vae, pred_model, val_dataset, normalise_weights, 0, sample_range) )
                if case==2:
                    cf_val.append( func_approx_score(cf_vae, pred_model, val_dataset, normalise_weights, 0, sample_range) )
                elif case==3:
                    val, inval= causal_score(cf_vae, pred_model, val_dataset, d, normalise_weights, offset, 0, sample_range)
                    cf_val.append( 100*2*np.array(val)*np.array(inval)/(np.array(val)+np.array(inval)) )
                elif case==4:
                    cf_val.append( causal_graph_score(cf_vae, pred_model, val_dataset, normalise_weights, 0, sample_range) )
                elif case==5:
                    cf_val.append( proximity_score(cf_vae, pred_model, val_dataset, d, mad_feature_weights, 0, 0, sample_range) )
                elif case==6:
                    cf_val.append( im_score(cf_vae, pred_model, val_dataset, d, normalise_weights, auto_encoder_models, div_case, 0, sample_range) )
                
        final_res[key]= cf_val        
        cf_val= np.mean( np.array(cf_val), axis=0 )
        if case==0:
            plt.title('Target Class Valid CF', fontsize=fsize)
            plt.xlabel('Total Counterfactuals requested per data point', fontsize=fsize)
            plt.ylabel('Percentage of valid CF w.r.t. ML Classifier',  fontsize=fsize)
        elif case==1:
            plt.title('Causal Edge Distribution Valid CF', fontsize=fsize)
            plt.xlabel('Total counterfactuals requested per data point', fontsize=fsize)
            plt.ylabel('Likelihood for known causal edges distribution', fontsize=fsize)
        elif case==2:
            plt.title('Causal Edge Function Valid CF', fontsize=fsize)
            plt.xlabel('Total counterfactuals requested per data point', fontsize=fsize)
            plt.ylabel('L1 norm for known causal edge function', fontsize=fsize)
        elif case==3:
            plt.title('Constraint Valid CF', fontsize=fsize)
            plt.xlabel('Total counterfactuals requested per data point', fontsize=fsize)
            plt.ylabel('Percentage of CF satisfying Constraint', fontsize=fsize)
        elif case==4:
            plt.title('Causal Graph Score', fontsize=14)
            plt.xlabel('Total counterfactuals requested per data point', fontsize=14)
            plt.ylabel('Likelihood', fontsize=14)
        elif case==5:
            plt.title('Continuous Proximity Score', fontsize=14)
            plt.xlabel('Total counterfactuals requested per data point', fontsize=14)
            plt.ylabel('Total change in continuous features ', fontsize=14)
        elif case==6:
            plt.title('Categorical Proximity Score', fontsize=14)
            plt.xlabel('Total counterfactuals requested per data point', fontsize=14)
            plt.ylabel('Total change in categorical features', fontsize=14)
                    
        if count==0:
            low = np.min(cf_val)
            high = np.max(cf_val)
        else:
            if low>np.min(cf_val):
                low=np.min(cf_val)
            elif high<np.max(cf_val):
                high=np.max(cf_val)

        if case ==0 or case ==3:
            plt.ylim(0,100)
        else:
            plt.ylim([np.ceil(low-0.5*(high-low)), np.ceil(high+0.5*(high-low))])        
            
        if len(sample_range)==1:
            plt.plot(sample_range, cf_val, '.', label=key)
        else:
            plt.plot(sample_range, cf_val, label=key)            
        
        count+=1        
        
    plt.legend(loc='best', fontsize=fsize/1.3)    
    plt.savefig('results/bn1/'+filename+'.jpg')
    plt.show()
    
    return final_res


def compute_eval_metrics_adult( methods, base_model_dir, encoded_size, pred_model, val_dataset, d, normalise_weights, mad_feature_weights, div_case, case, sample_range, filename ):   
    count=0
    fsize=20
    fig = plt.figure(figsize=(7.7,6.5))
    final_res= {}
    dataset_name= 'adult'
    
    np.random.shuffle(val_dataset)
    x_sample= val_dataset[0,:]    
    x_sample= np.reshape( x_sample, (1, val_dataset.shape[1]))
    np.save('adult-visualise-sample.npy', x_sample)
    print('Input Data Sample: ', d.de_normalize_data( d.get_decoded_data(x_sample) ))
    
    # Loading all the Auto Encoder models
    auto_encoder_models= []
    for i in range(0,2):
        ae= AutoEncoder(len(d.encoded_feature_names), encoded_size, d)
        path=  'models/adult-2048-50-target-class-' + str(i) + '-auto-encoder.pth'
        ae.load_state_dict(torch.load(path))
        ae.eval()
        auto_encoder_models.append(ae)    
    
    # Loading all the Auto Encoder for CEM
    auto_encoder_models_cem= []
    for i in range(0,2):
        
        # load json and create model
        path='models/adult-keras-ae-model-target-class-' + str(i)
        json_file = open(path+'.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        ae_model = tf.keras.models.model_from_json(loaded_model_json)

        # load weights into new model
        ae_model.load_weights(path+".h5")
        ae_model.compile(loss=nll, optimizer='adam')
        
        auto_encoder_models_cem.append(ae_model)    
    
    for key in methods.keys():

        #Loading torch model
        wm1=1e-2
        wm2=1e-2
        wm3=1e-2
        wm4=1e-2

        path= methods[key]
        cf_val=[]
        
        if 'contrastive' in path and 'json' in path:
            f=open(path,'r')
            contrastive_exp= json.load(f)
            f.close()

            if case==-1:        
                print('\n', 'Method: ', key, "\n")
                contrastive_visualize_score(contrastive_exp, x_sample, d)
                continue              
           
            for i in range(10):

                if case==0:
                    cf_val.append( contrastive_validity_score(contrastive_exp, 0, sample_range) )
                elif case==1:
                    val, inval= contrastive_causal_score_age_constraint(contrastive_exp, d, normalise_weights, offset, 0, sample_range)
                    cf_val.append( 100*np.array(val)/(np.array(val)+np.array(inval)) )
                elif case==2:
                    val, inval= contrastive_causal_score_age_ed_constraint(contrastive_exp, d, normalise_weights, offset, 0, sample_range)
                    cf_val.append( 100*np.array(val)/(np.array(val)+np.array(inval)) )
                elif case==3:
                    cf_val.append( contrastive_proximity_score(contrastive_exp, d, mad_feature_weights, 0, 0, sample_range) )
                elif case==4:
                    cf_val.append( contrastive_proximity_score(contrastive_exp, d, mad_feature_weights, 1, 0, sample_range) )  
                elif case==5:
                    cf_val.append( contrastive_im_score(contrastive_exp, d, normalise_weights, auto_encoder_models_cem, dataset_name, div_case, 0, sample_range) )                    
        else:
            cf_vae = CF_VAE( len(d.encoded_feature_names), encoded_size, d )
            cf_vae.load_state_dict(torch.load(path))
            cf_vae.eval()
            learning_rate = 1e-2
            cf_vae_optimizer = optim.Adam([
                {'params': filter(lambda p: p.requires_grad, cf_vae.encoder_mean.parameters()),'weight_decay': wm1},
                {'params': filter(lambda p: p.requires_grad, cf_vae.encoder_var.parameters()),'weight_decay': wm2},
                {'params': filter(lambda p: p.requires_grad, cf_vae.decoder_mean.parameters()),'weight_decay': wm3}
            ], lr=learning_rate)        

            if case==-1:        
                print('\n', 'Method: ', key, "\n")
                visualize_score(cf_vae, pred_model, x_sample, d)
                continue              

            # Put the check for only Low to High Income CF
            train_x= torch.tensor( val_dataset ).float() 
            train_y = torch.argmax( pred_model(train_x), dim=1 ).numpy()
            val_dataset= val_dataset[ train_y==0 ]
                
            for i in range(10):
                if case==0:
                    cf_val.append( validity_score(cf_vae, pred_model, val_dataset, 0, sample_range) )
                elif case==1:
                    val, inval= causal_score_age_constraint(cf_vae, pred_model, val_dataset, d, normalise_weights, offset, 0, sample_range)
                    cf_val.append( 100*np.array(val)/(np.array(val)+np.array(inval)) )
                elif case==2:
                    val, inval= causal_score_age_ed_constraint(cf_vae, pred_model, val_dataset, d, normalise_weights, offset, 0, sample_range)
                    cf_val.append( 100*np.array(val)/(np.array(val)+np.array(inval)) )
                elif case==3:
                    cf_val.append( proximity_score(cf_vae, pred_model, val_dataset, d, mad_feature_weights, 0, 0, sample_range) )
                elif case==4:
                    cf_val.append( proximity_score(cf_vae, pred_model, val_dataset, d, mad_feature_weights, 1, 0, sample_range) )
                elif case==5:
                    cf_val.append( im_score(cf_vae, pred_model, val_dataset, d, normalise_weights, auto_encoder_models, div_case, 0, sample_range) )
        
        final_res[key]= cf_val
        cf_val= np.mean( np.array(cf_val), axis=0 )
        if case==0:
            plt.title('Target Class Valid CF', fontsize=fsize)
            plt.xlabel('Total Counterfactuals requested per data point', fontsize=fsize)
            plt.ylabel('Percentage of valid CF w.r.t. ML Classifier',  fontsize=fsize)
        elif case==1:
            plt.title('Constraint Valid CF: Age Constraint', fontsize=fsize)
            plt.xlabel('Total counterfactuals requested per data point', fontsize=fsize)
            plt.ylabel('Percentage of CF satisfying Constraint', fontsize=fsize)
        elif case==2:
            plt.title('Constraint Valid CF: Age-Education Constraint', fontsize=fsize)
            plt.xlabel('Total counterfactuals requested per data point', fontsize=fsize)
            plt.ylabel('Percentage of CF satisfying Constraint', fontsize=fsize)
        elif case==3:
            plt.title('Continuous Proximity Score', fontsize=fsize)
            plt.xlabel('Total counterfactuals requested per data point', fontsize=fsize)
            plt.ylabel('Total change in continuous features', fontsize=fsize)
        elif case==4:
            plt.title('Categorical Proximity Score', fontsize=fsize)
            plt.xlabel('Total counterfactuals requested per data point', fontsize=fsize)
            plt.ylabel('Total change in categorical features', fontsize=fsize)
            
        if count==0:
            low = min(cf_val)
            high = max(cf_val)
        else:
            if low>min(cf_val):
                low=min(cf_val)
            elif high<max(cf_val):
                high=max(cf_val)
                
        if case ==0 or case ==1 or case ==2:
            plt.ylim(0,101)
        else:
            plt.ylim([np.ceil(low-0.5*(high-low)), np.ceil(high+0.5*(high-low))])  
        
        if len(sample_range)==1:
            plt.plot(sample_range, cf_val, '.', label=key)
        else:
            plt.plot(sample_range, cf_val, label=key)            
            
        count+=1    
    
    plt.legend(loc='lower left', fontsize=fsize/1.3)    
    plt.savefig('results/adult/'+filename+'.jpg')
    plt.show()
    
    return final_res

def compute_eval_metrics_sangiovese( methods, base_model_dir, encoded_size, pred_model, val_dataset, d, normalise_weights, mad_feature_weights, scm_model, constraint_nodes,  div_case, case, sample_range, filename ):    
    count=0
    fsize=20
    fig = plt.figure(figsize=(7.7,6.5))
    final_res={}
    dataset_name= 'sangiovese'
    
    np.random.shuffle(val_dataset)
    x_sample= val_dataset[0,:]
    x_sample= np.reshape( x_sample, (1, val_dataset.shape[1]))
    print('Input Data Sample: ', d.de_normalize_data( d.get_decoded_data(x_sample) ))

    # Loading all the Auto Encoder models
    auto_encoder_models= []
    for i in range(0,2):
        ae= AutoEncoder(len(d.encoded_feature_names), encoded_size, d)
        path=  'models/sangiovese-512-50-target-class-' + str(i) + '-auto-encoder.pth'
        ae.load_state_dict(torch.load(path))
        ae.eval()
        auto_encoder_models.append(ae)        
        
    # Loading all the Auto Encoder for CEM
    auto_encoder_models_cem= []
    for i in range(0,2):
        
        # load json and create model
        path='models/sangiovese-keras-ae-model-target-class-' + str(i)
        json_file = open(path+'.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        ae_model = tf.keras.models.model_from_json(loaded_model_json)

        # load weights into new model
        ae_model.load_weights(path+".h5")
        ae_model.compile(loss=nll, optimizer='adam')
        
        auto_encoder_models_cem.append(ae_model)        
    
    for key in methods.keys():

        #Loading torch model
        wm1=1e-4
        wm2=1e-3
        wm3=1e-4
        wm4=1e-3
        
        path= methods[key]
        cf_val=[]

        if 'contrastive' in path and 'json' in path:
            f=open(path,'r')
            contrastive_exp= json.load(f)
            f.close()
            for i in range(10):
                if case==0:
                    cf_val.append( contrastive_validity_score(contrastive_exp, 0, sample_range) )
                elif case==1:
                    cf_val.append( contrastive_bnlearn_causal_graph_score( contrastive_exp, d, normalise_weights, scm_model, constraint_nodes, 1, 0, sample_range ) )
                elif case==3:
                    val, inval= contrastive_causal_score_bnlearn_constraint(contrastive_exp, d, normalise_weights, offset, scm_model, constraint_nodes, 0, sample_range)
                    cf_val.append( 2*100*np.array(val)*np.array(inval)/(np.array(val)+np.array(inval)) )
                elif case==4:
                    cf_val.append( contrastive_bnlearn_causal_graph_score( contrastive_exp, d, normalise_weights, scm_model, constraint_nodes, 0, 0, sample_range ) )                    
                elif case==5:
                    cf_val.append( contrastive_proximity_score(contrastive_exp, d, mad_feature_weights, 0, 0, sample_range) )
                elif case==6:
                    cf_val.append( contrastive_proximity_score(contrastive_exp, d, mad_feature_weights, 1, 0, sample_range) )
                elif case==7:
                    cf_val.append( contrastive_im_score(contrastive_exp, d, normalise_weights, auto_encoder_models_cem, dataset_name, div_case, 0, sample_range) )                    
        else: 
            cf_vae = CF_VAE( len(d.encoded_feature_names), encoded_size, d )
            cf_vae.load_state_dict(torch.load(path))
            cf_vae.eval()
            learning_rate = 1e-3
            cf_vae_optimizer = optim.Adam([
                {'params': filter(lambda p: p.requires_grad, cf_vae.encoder_mean.parameters()),'weight_decay': wm1},
                {'params': filter(lambda p: p.requires_grad, cf_vae.encoder_var.parameters()), 'weight_decay': wm2},
                {'params': filter(lambda p: p.requires_grad, cf_vae.decoder_mean.parameters()),'weight_decay': wm3}
            ], lr=learning_rate)       

            cf_val=[]

            if case==-1:        
                print('\n', 'Method: ', key, "\n")
                visualize_score(cf_vae, pred_model, x_sample, d)
                continue                      

            for i in range(10):
                if case==0:
                    cf_val.append( validity_score(cf_vae, pred_model, val_dataset, 0, sample_range) )
                if case==1:
                    cf_val.append( bnlearn_causal_graph_score(cf_vae, pred_model, val_dataset, normalise_weights, d, scm_model, constraint_nodes, 1, 0, sample_range) )
                if case==2:
                    cf_val.append( func_approx_score(cf_vae, pred_model, val_dataset, normalise_weights, 0, sample_range) )
                elif case==3:
                    val, inval= bnlearn_causal_score(cf_vae, pred_model, val_dataset, d, normalise_weights, offset, scm_model, constraint_nodes, 0, sample_range)
                    cf_val.append( 2*100*np.array(val)*np.array(inval)/(np.array(val)+np.array(inval)) )
                elif case==4:
                     cf_val.append( bnlearn_causal_graph_score(cf_vae, pred_model, val_dataset, normalise_weights, d, scm_model, constraint_nodes, 0, 0, sample_range) )
                elif case==5:
                    cf_val.append( proximity_score(cf_vae, pred_model, val_dataset, d, mad_feature_weights, 0, 0, sample_range) )
                elif case==6:
                    cf_val.append( proximity_score(cf_vae, pred_model, val_dataset, d, mad_feature_weights, 1, 0, sample_range) )
                elif case==7:
                    cf_val.append( im_score(cf_vae, pred_model, val_dataset, d, normalise_weights, auto_encoder_models, div_case, 0, sample_range) ) 
                    
        print(key)
        
        final_res[key]= cf_val
        cf_val= np.mean( np.array(cf_val), axis=0 )
        if case==0:
            plt.title('Target Class Valid CF', fontsize=fsize)
            plt.xlabel('Total Counterfactuals requested per data point', fontsize=fsize)
            plt.ylabel('Percentage of valid CF w.r.t. ML Classifier',  fontsize=fsize)
        elif case==1:
            plt.title('Causal Edge Distribution Valid CF', fontsize=fsize)
            plt.xlabel('Total counterfactuals requested per data point', fontsize=fsize)
            plt.ylabel('Likelihood for known causal edges distribution', fontsize=fsize)
        elif case==2:
            plt.title('Causal Edge Function Valid CF', fontsize=fsize)
            plt.xlabel('Total counterfactuals requested per data point', fontsize=fsize)
            plt.ylabel('L1 norm for known causal edge function', fontsize=fsize)
        elif case==3:
            plt.title('Constraint Valid CF', fontsize=fsize)
            plt.xlabel('Total counterfactuals requested per data point', fontsize=fsize)
            plt.ylabel('Percentage of CF satisfying Constraint', fontsize=fsize)
        elif case==4:
            plt.title('Causal Graph Score', fontsize=14)
            plt.xlabel('Total counterfactuals requested per data point', fontsize=14)
            plt.ylabel('Likelihood', fontsize=14)
        elif case==5:
            plt.title('Continuous Proximity Score', fontsize=14)
            plt.xlabel('Total counterfactuals requested per data point', fontsize=14)
            plt.ylabel('Total change in continuous features ', fontsize=14)
        elif case==6:
            plt.title('Categorical Proximity Score', fontsize=14)
            plt.xlabel('Total counterfactuals requested per data point', fontsize=14)
            plt.ylabel('Total change in categorical features', fontsize=14)
                    
        if count==0:
            low = min(cf_val)
            high = max(cf_val)
        else:
            if low>min(cf_val):
                low=min(cf_val)
            elif high<max(cf_val):
                high=max(cf_val)

        if case ==0 or case ==3:
            plt.ylim(0,100)
        else:
            plt.ylim([np.ceil(low-0.5*(high-low)), np.ceil(high+0.5*(high-low))])        
            
        if len(sample_range)==1:
            plt.plot(sample_range, cf_val, '.', label=key)
        else:
            plt.plot(sample_range, cf_val, label=key)            
            
        count+=1
   
    plt.legend(loc='best', fontsize=fsize/1.3)    
    plt.savefig('results/sangiovese/'+filename+'.jpg')
    plt.show()

    return final_res