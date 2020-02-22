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

import itertools
def flip(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])

#Main dicionary to store the results
res={}
res['bn1']={}
res['adult']={}
res['sangiovese']={}


'''
BN1 Section
'''


base_data_dir='data/'
base_model_dir='models/'
dataset_name= 'bn1'
dataset= pd.read_csv(base_data_dir+dataset_name+'.csv')
dataset.drop(['Unnamed: 0'], axis=1, inplace=True)  
params= {'dataframe':dataset.copy(), 'continuous_features':['x1','x2','x3'], 'outcome_name':'y'}
d = DataLoader(params)

#Load Train, Val, Test Dataset
vae_test_dataset= np.load(base_data_dir+dataset_name+'-test-set.npy')
vae_test_dataset= vae_test_dataset[:,:-1]

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

encoded_size=10
sample_range=[1,2,3]
div_case=1

methods={'BaseGenCF': base_model_dir + 'bn1-margin-0.014-validity_reg-54.0-epoch-50-base-gen.pth',          
         'AEGenCF':  base_model_dir + 'bn1-margin-0.324-validity_reg-72.0-ae_reg-24.0-epoch-50-ae-gen.pth',
         'SCMGenCF': base_model_dir + 'bn1-delta-case-0-margin-0.015-scm_reg-55.0-validity_reg-85.0-epoch-50-scm-gen.pth',
         'ModelApproxGenCF': base_model_dir + 'bn1-delta-case-0-marign-0.087-ae-reg-0.0-constraint_reg-0.1-validity_reg-96.0-epoch-50-model-approx-gen.pth',
         'OracleGenCF' : base_model_dir + 'bn1-eval-case-0-supervision-limit-100-const-case-0-margin-0.15-oracle_reg-2350.0-validity_reg-150.0-epoch-50-oracle-gen.pth',
         'CEM': base_model_dir + 'bn1-sample-size-3-size-100-beta-0.608-kappa-0.021-gamma-8.0-cteps-3-maxiter-1000-contrastive-explanations.json'
        }

res[dataset_name]['validity']=  compute_eval_metrics_bn1( methods, base_model_dir, encoded_size, pred_model, vae_test_dataset, d, normalise_weights, mad_feature_weights, div_case, 0, sample_range, 'bn1-validity' )
res[dataset_name]['dist-score']= compute_eval_metrics_bn1( methods, base_model_dir, encoded_size, pred_model, vae_test_dataset, d, normalise_weights, mad_feature_weights, div_case, 1, sample_range, 'bn1-feature-dist-score' )
res[dataset_name]['const-score']= compute_eval_metrics_bn1( methods, base_model_dir, encoded_size, pred_model, vae_test_dataset, d, normalise_weights, mad_feature_weights, div_case, 3, sample_range, 'bn1-constraint-score' )
res[dataset_name]['cont-prox']= compute_eval_metrics_bn1( methods, base_model_dir, encoded_size, pred_model, vae_test_dataset, d, normalise_weights, mad_feature_weights, div_case, 5, sample_range, 'bn1-cont-proximity-score' )
res[dataset_name]['im']= compute_eval_metrics_bn1( methods, base_model_dir, encoded_size, pred_model, vae_test_dataset, d, normalise_weights, mad_feature_weights, div_case, 6, sample_range, 'bn1-im-score' )


'''
Sangiovese Section
'''


base_data_dir='data/'
base_model_dir='models/'
dataset_name='sangiovese'

dataset = pd.read_csv(  base_data_dir + dataset_name + '.csv', index_col=None )
dataset= dataset.drop(columns= ['Unnamed: 0'])
outcome=[]
for i in range(dataset.shape[0]):
    if dataset['GrapeW'][i] > 0: 
        outcome.append( 1 )
    else:
        outcome.append( 0 )
dataset['outcome'] = pd.Series(outcome)
dataset.drop(columns=['GrapeW'], axis=1 ,inplace=True)

# Continuous Features
l=list(dataset.columns)
# l.remove('Treatment')
l.remove('outcome')

params= {'dataframe':dataset.copy(), 'continuous_features':l, 'outcome_name':'outcome'}
d = DataLoader(params)    
print(d.encoded_feature_names) 

#Load Train, Val, Test Dataset
vae_test_dataset= np.load(base_data_dir+dataset_name+'-test-set.npy')
vae_test_dataset= vae_test_dataset[:,:-1]

with open(base_data_dir+dataset_name+'-normalise_weights.json') as f:
    normalise_weights= json.load(f)
normalise_weights = {int(k):v for k,v in normalise_weights.items()}

with open(base_data_dir+dataset_name+'-mad.json') as f:
    mad_feature_weights= json.load(f)

with open(base_data_dir+dataset_name+'-scm.json') as f:
    scm_model= json.load(f)   
    
print(normalise_weights)
print(mad_feature_weights)

#Load Black Box Prediction Model
data_size= len(d.encoded_feature_names)
pred_model= BlackBox(data_size)
path= base_model_dir + dataset_name +'.pth'
pred_model.load_state_dict(torch.load(path))
pred_model.eval()

#Load CF Generator Model
encoded_size=10

#Constraint Nodes
constraint_nodes=['BunchN']

sample_range=[1,2,3]
div_case=1

methods={'BaseGenCF': base_model_dir + 'sangiovese-margin-0.161-validity_reg-94.0-epoch-50-base-gen.pth', 
         'AEGenCF': base_model_dir + 'sangiovese-margin-0.12-validity_reg-44.0-ae_reg-84.0-epoch-25-ae-gen.pth',
         'ModelApproxGenCF': base_model_dir + 'sangiovese-delta-case-0-margin-0.306-ae-reg-0-constraint_reg-73.0-validity_reg-71.0-constrained_node-BunchN-epoch-25-model-approx-gen.pth',
         'SCMGenCF': base_model_dir + 'sangiovese-delta-case-0-margin-0.319-scm_reg-77.0-validity_reg-89.0-constraint_node-BunchN-epoch-25-scm-gen.pth',
         'OracleGenCF': base_model_dir + 'sangiovese-eval-case-0-supervision-limit-100-const-case-0-margin-0.02-oracle_reg-1085.1-validity_reg-25.2-epoch-25-oracle-gen.pth',
         'CEM': base_model_dir + 'sangiovese-sample-size-3-size-100-beta-0.652-kappa-0.041-gamma-9.0-cteps-5-maxiter-1000-contrastive-explanations.json'}


res[dataset_name]['validity']= compute_eval_metrics_sangiovese( methods, base_model_dir, encoded_size, pred_model, vae_test_dataset, d, normalise_weights, mad_feature_weights, scm_model, constraint_nodes, div_case, 0, sample_range, 'sangiovese-validity' )
res[dataset_name]['dist-score']= compute_eval_metrics_sangiovese( methods, base_model_dir, encoded_size, pred_model, vae_test_dataset, d, normalise_weights, mad_feature_weights, scm_model, constraint_nodes, div_case, 1, sample_range, 'sangiovese-feature-dist-score' )
res[dataset_name]['cont-prox']= compute_eval_metrics_sangiovese( methods, base_model_dir, encoded_size, pred_model, vae_test_dataset, d, normalise_weights, mad_feature_weights, scm_model, constraint_nodes, div_case, 5, sample_range, 'sangiovese-cont-proximity-score' )
res[dataset_name]['im']= compute_eval_metrics_sangiovese( methods, base_model_dir, encoded_size, pred_model, vae_test_dataset, d, normalise_weights, mad_feature_weights, scm_model, constraint_nodes, div_case, 7, sample_range, 'sangiovese-im-score' )
res[dataset_name]['const-score']= compute_eval_metrics_sangiovese( methods, base_model_dir, encoded_size, pred_model, vae_test_dataset, d, normalise_weights, mad_feature_weights, scm_model, constraint_nodes, div_case, 3, sample_range, 'sangiovese-constraint-score' )


'''
Adult Section
'''

base_data_dir='data/'
base_model_dir='models/'
dataset_name= 'adult'
dataset = load_adult_income_dataset()
params= {'dataframe':dataset.copy(), 'continuous_features':['age','hours_per_week'], 'outcome_name':'income'}
d = DataLoader(params)  

#Load Train, Val, Test Dataset
vae_test_dataset= np.load(base_data_dir+dataset_name+'-test-set.npy')
vae_test_dataset= vae_test_dataset[vae_test_dataset[:,-1]==0,:]
vae_test_dataset= vae_test_dataset[:,:-1]

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

encoded_size=10

sample_range= [1,2,3]
div_case=1

dataset_name='adult-age'
res[dataset_name]={}

methods={'BaseGenCF': base_model_dir + 'adult-margin-0.165-validity_reg-42.0-epoch-25-base-gen.pth', 
         'AEGenCF': base_model_dir + 'adult-margin-0.369-validity_reg-73.0-ae_reg-2.0-epoch-25-ae-gen.pth',
         'ModelApproxGenCF': base_model_dir + 'adult-margin-0.764-constraint-reg-192.0-validity_reg-29.0-epoch-25-unary-gen.pth',
         'CEM': base_model_dir + 'adult-sample-size-3-size-100-beta-0.911-kappa-0.241-gamma-0.0-cteps-9-maxiter-1000-contrastive-explanations.json',
         'OracleGenCF': base_model_dir + 'adult-eval-case-0-supervision-limit-100-const-case-0-margin-0.084-oracle_reg-5999.0-validity_reg-159.0-epoch-50-oracle-gen.pth'
        }

res[dataset_name]['validity']= compute_eval_metrics_adult( methods, base_model_dir, encoded_size, pred_model, vae_test_dataset, d, normalise_weights, mad_feature_weights, div_case, 0, sample_range, 'adult-validity' )
res[dataset_name]['const-score']= compute_eval_metrics_adult( methods, base_model_dir, encoded_size, pred_model, vae_test_dataset, d, normalise_weights, mad_feature_weights, div_case, 1, sample_range, 'adult-age-constraint-score' )
res[dataset_name]['cont-prox']= compute_eval_metrics_adult( methods, base_model_dir, encoded_size, pred_model, vae_test_dataset, d, normalise_weights, mad_feature_weights, div_case, 3, sample_range, 'adult-cont-proximity-score' )
res[dataset_name]['cat-prox']= compute_eval_metrics_adult( methods, base_model_dir, encoded_size, pred_model, vae_test_dataset, d, normalise_weights, mad_feature_weights, div_case, 4, sample_range, 'adult-cat-proximity-score' )
res[dataset_name]['im']= compute_eval_metrics_adult( methods, base_model_dir, encoded_size, pred_model, vae_test_dataset, d, normalise_weights, mad_feature_weights, div_case, 5, sample_range, 'im-score' )

sample_range= [1,2,3]

dataset_name='adult-age-ed'
res[dataset_name]={}

methods={'BaseGenCF':base_model_dir  + 'adult-margin-0.165-validity_reg-42.0-epoch-25-base-gen.pth', 
         'AEGenCF': base_model_dir + 'adult-margin-0.369-validity_reg-73.0-ae_reg-2.0-epoch-25-ae-gen.pth',
         'ModelApproxGenCF': base_model_dir + 'adult-margin-0.344-constraint-reg-87.0-validity_reg-76.0-epoch-25-unary-ed-gen.pth',
         'CEM': base_model_dir + 'adult-sample-size-3-size-100-beta-0.911-kappa-0.241-gamma-0.0-cteps-9-maxiter-1000-contrastive-explanations.json',
         'OracleGenCF':base_model_dir + 'adult-eval-case-0-supervision-limit-100-const-case-1-margin-0.117-oracle_reg-3807.0-validity_reg-175.0-epoch-50-oracle-gen.pth'
        }

res[dataset_name]['validity']= compute_eval_metrics_adult( methods, base_model_dir, encoded_size, pred_model, vae_test_dataset, d, normalise_weights, mad_feature_weights, div_case,0, sample_range, 'adult-validity' )
res[dataset_name]['const-score']= compute_eval_metrics_adult( methods, base_model_dir, encoded_size, pred_model, vae_test_dataset, d, normalise_weights, mad_feature_weights, div_case, 2, sample_range, 'adult-ed-age-constraint-score' )
res[dataset_name]['cont-prox']= compute_eval_metrics_adult( methods, base_model_dir, encoded_size, pred_model, vae_test_dataset, d, normalise_weights, mad_feature_weights, div_case, 3, sample_range, 'adult-cont-proximity-score' )
res[dataset_name]['cat-prox']= compute_eval_metrics_adult( methods, base_model_dir, encoded_size, pred_model, vae_test_dataset, d, normalise_weights, mad_feature_weights, div_case, 4, sample_range, 'adult-cat-proximity-score' )
res[dataset_name]['im']= compute_eval_metrics_adult( methods, base_model_dir, encoded_size, pred_model, vae_test_dataset, d, normalise_weights, mad_feature_weights, div_case, 5, sample_range, 'im-score' )


sample_range= [1,2,4,8,10]
div_case=1

dataset_name='adult-supervision'
res[dataset_name]={}

base_model_dir= 'models/'
base_model_path= 'adult-eval-case-0-supervision-limit-100-const-case-0-margin-0.037-oracle_reg-8557.0-validity_reg-88.0-epoch-25-oracle-gen.pth'

methods={ 'Sample-25-CF-10': base_model_dir + 'adult-fine-tune-size-25-upper-lim-10-age-good-cf-set/' + base_model_path,
         'Sample-50-CF-10': base_model_dir + 'adult-fine-tune-size-50-upper-lim-10-age-good-cf-set/' + base_model_path,
         'Sample-75-CF-10': base_model_dir + 'adult-fine-tune-size-75-upper-lim-10-age-good-cf-set/' + base_model_path,
         'Sample-100-CF-10': base_model_dir + 'adult-fine-tune-size-100-upper-lim-10-age-good-cf-set/' + base_model_path
        }

res[dataset_name]['validity']= compute_eval_metrics_adult( methods, base_model_dir, encoded_size, pred_model, vae_test_dataset, d, normalise_weights, mad_feature_weights, div_case, 0, sample_range, 'adult-validity' )
res[dataset_name]['const-score']= compute_eval_metrics_adult( methods, base_model_dir, encoded_size, pred_model, vae_test_dataset, d, normalise_weights, mad_feature_weights, div_case, 1, sample_range, 'adult-age-constraint-score' )
res[dataset_name]['cont-prox']= compute_eval_metrics_adult( methods, base_model_dir, encoded_size, pred_model, vae_test_dataset, d, normalise_weights, mad_feature_weights, div_case, 3, sample_range, 'adult-cont-proximity-score' )
res[dataset_name]['cat-prox']= compute_eval_metrics_adult( methods, base_model_dir, encoded_size, pred_model, vae_test_dataset, d, normalise_weights, mad_feature_weights, div_case, 4, sample_range, 'adult-cat-proximity-score' )
res[dataset_name]['im']= compute_eval_metrics_adult( methods, base_model_dir, encoded_size, pred_model, vae_test_dataset, d, normalise_weights, mad_feature_weights, div_case, 5, sample_range, 'im-score' )


'''
Master Evaluation
'''

color_range= {'BaseGenCF':'Grey',              
              'AEGenCF': 'Brown',
              'ModelApproxGenCF': 'purple',
              'SCMGenCF': 'blue',
              'CEM': 'darkorange',
              'OracleGenCF': 'Red'
             }

patterns = {'BaseGenCF':'/',              
              'AEGenCF': 'o',
              'ModelApproxGenCF': '\\',
              'SCMGenCF': '+',
              'CEM': '.',
              'OracleGenCF': 'x'
             }


fsize=21
width = 1   
group_gap=1

x_label_bn1=['BaseGenCF', 'AEGenCF', 'OracleGenCF', 'ModelApproxGenCF', 'SCMGenCF']
x_label_adult_age=['BaseGenCF', 'AEGenCF', 'OracleGenCF', 'ModelApproxGenCF', 'CEM']
x_label_adult_age_ed=['BaseGenCF', 'AEGenCF', 'OracleGenCF', 'CEM']
x_label_sangiovese= ['BaseGenCF', 'AEGenCF', 'OracleGenCF', 'ModelApproxGenCF', 'SCMGenCF']

x_label=['BaseGenCF', 'AEGenCF', 'ModelApproxGenCF', 'OracleGenCF', 'CEM', 'SCMGenCF' ]
dataset_label=['bn1', 'sangiovese', 'adult-age', 'adult-age-ed']
# dataset_label=['sangiovese']
# dataset_label=['adult-age', 'adult-age-ed']

eval_metric={}
for dataset_name in res.keys():
    for metric in res[dataset_name].keys():
        eval_metric[metric]=1

eval_metric= list(eval_metric.keys())
print(eval_metric)

for metric in eval_metric:
    fig, ax = plt.subplots(figsize=(7.0,7.0))    
    plt.gcf().subplots_adjust(bottom=0.20)    
    idx=0
    group_size=0
    title='..'
    label='..'
    start_idx=1
    ind=[]
    x_ticks=[]
    ind_case=1
    for method in x_label:
        plot_x=[]
        plot_y=[]
        plot_y_err=[]
        
        dataset_counter=0
        for dataset_name in dataset_label:        
            if metric not in res[dataset_name].keys():
                continue
                
#             if dataset_name=='adult-age' or dataset_name=='adult-age-ed':
#                 if method=='SCMGenCF':
#                     method='CEM'
#                 elif method=='CEM':
#                     method='SCMGenCF'
                    
            if method in res[dataset_name][metric].keys():                
                arr= np.array( res[dataset_name][metric][method] )
                arr= np.mean( arr, axis=1)
                
                if dataset_name=='sangiovese' and metric=='causal-graph':
                    arr= arr/10
                
                plot_y.append( np.mean(arr) )
                plot_y_err.append( np.std(arr) )                
                
            else:
                plot_y.append( 0.0 )
                plot_y_err.append( 0.0 )       
                
            dataset_counter+=1
            plot_x.append(start_idx + group_gap*dataset_counter*(2+len(x_label)))                
            if 'adult' in dataset_name:
                if method=='SCMGenCF':
                    plot_x[-1]= plot_x[-1]+1
                elif method=='cem':
                    plot_x[-1]= plot_x[-1]-1
            
            if ind_case:
                ind.append(3.5 + group_gap*dataset_counter*(2+len(x_label)))
                if dataset_name=='bn1':
                    x_ticks.append('simple-bn')
                else:
                    x_ticks.append(dataset_name)
                    
            if metric=='validity':
#                     title= 'Target Class Valid CF'
#                     label='Percentage of valid CF w.r.t. ML Classifier'
                label='Target-Class Validity'
            elif metric=='dist-score':
#                     title= 'Causal Edge Distribution Valid CF'
#                     label= 'Likelihood for known causal edges distribution'
                label= 'Causal-Edge Score'
            elif metric=='const-score' or metric=='const-score-age' or metric=='const-score-age-ed':
#                     title= 'Constraint Valid CF'
#                     label= 'Percentage of CF satisfying Constraint'
                label='Constraint Feasibility Score'
            elif metric=='causal-graph':
#                     title= 'Causal Graph Score'
#                     label = 'Likelihood'
                label= 'Causal-Graph Score'
            elif metric=='cont-prox':
#                     title= 'Continuous Proximity Score'
#                     label = 'Total change in continuous features'
                label= 'Continuous Proximity'
            elif metric=='cat-prox':
#                     title= 'Categorical Proximity Score'
#                     label= 'Total change in categorical features'   
                label= 'Categorical Proximity'
            elif metric=='im':
                label= 'Interpretability Score'
        
        if metric=='dist-score':
            print(plot_y)
        rects = ax.bar(plot_x, plot_y, width, yerr=plot_y_err, hatch= patterns[method], color=color_range[method], edgecolor= "black", label=method, error_kw=dict(lw=1.5, capsize=1.5, capthick=1.5) )
        start_idx +=1
        ind_case=0
        
    ax.set_title( title, fontsize=fsize )
    ax.set_ylabel(label, fontsize=fsize)
    ax.set_xticks(ind)
    ax.set_xticklabels( x_ticks ,fontsize=fsize, rotation=20)
    
#     if metric=='validity' or metric=='cat-prox':
    handles, labels = ax.get_legend_handles_labels() 
    lgd= plt.legend(  flip(handles, 2), flip(labels, 2), loc='right', fontsize=fsize, bbox_to_anchor=(1.2, -0.35), ncol=2, )
    text = ax.text(-0.2,1.05, "", transform=ax.transAxes)
    plt.tick_params(labelsize=fsize)
    plt.savefig('results/icml/'+  str(metric) +'.jpg', bbox_extra_artists=(lgd, text), bbox_inches='tight', dpi=100)
#     else:
#         plt.tick_params(labelsize=fsize)
#         plt.savefig('results/icml/'+  str(metric) +'.jpg', bbox_inches='tight')
         
    plt.show()
    plt.clf()

    
'''
Adult Oracle Supervision Analysis
'''
    
    
fsize=21
linewidth=4.0

dataset_name='adult-supervision'
x_label=[ 'Sample-25-CF-10', 'Sample-50-CF-10',  'Sample-75-CF-10', 'Sample-100-CF-10' ]
x_ticks=['25', '50', '75', '100']

eval_metric={}
for metric in res[dataset_name].keys():
    eval_metric[metric]=1
eval_metric= list(eval_metric.keys())
print(eval_metric)

for metric in eval_metric:
    fig = plt.figure(figsize=(9.5,6.0))    
    idx=0
    for dataset_name in res.keys():
        
        if dataset_name != 'adult-supervision':
            continue
        
        if metric in res[dataset_name].keys():
            plot_y=[]
            plot_y_err=[]
            plot_x= x_label
            for method in plot_x:
                if method in res[dataset_name][metric].keys():
                    arr= np.array( res[dataset_name][metric][method] )
                    arr= np.mean( arr, axis=1)
                    plot_y.append( np.mean(arr) )
                    plot_y_err.append( np.std(arr) )       
                else:
                    plot_y.append(0)
                    plot_y_err.append(0)

                if metric=='validity':
#                     plt.title('Target Class Valid CF', fontsize=fsize)
                    plt.xlabel('Labelled Set Size', fontsize=fsize)
                    plt.ylabel('Target-Class Validity',  fontsize=fsize)                    
                    plt.ylim(50,100)
                elif metric=='const-score':
#                     plt.title('Constraint Valid CF', fontsize=fsize)
                    plt.ylabel('Constraint Feasibility Score', fontsize=fsize)
                    plt.xlabel('Labelled Set Size', fontsize=fsize)
                elif metric=='cont-prox':
#                     plt.title('Continuous Proximity Score', fontsize=fsize)
                    plt.ylabel('Continuous Proximity', fontsize=fsize)
                    plt.xlabel('Labelled Set Size', fontsize=fsize)
                    plt.ylim(-8.0,-1.0)
                elif metric=='cat-prox':
#                     plt.title('Categorical Proximity Score', fontsize=fsize)
                    plt.ylabel('Categorical Proximity', fontsize=fsize)   
                    plt.xlabel('Labelled Set Size', fontsize=fsize)
                    plt.ylim(-4.0,-1.0)
                elif metric=='im':
#                     plt.title('Categorical Proximity Score', fontsize=fsize)
                    plt.ylabel('Interpretability Score', fontsize=fsize)   
                    plt.xlabel('Labelled Set Size', fontsize=fsize)
                    
            plt.xticks(range(len(x_ticks)), x_ticks, fontsize=0.75*fsize)
            plt.plot(plot_x, plot_y, '--', color='grey', linewidth=linewidth)
            plt.errorbar(plot_x, plot_y, yerr=plot_y_err, fmt='o', elinewidth=linewidth, markeredgewidth=linewidth, capsize=linewidth )
                
        idx=+1
    
    dataset_name='adult-supervision'
    plt.tick_params(labelsize=fsize)
    plt.savefig('results/icml/'+ dataset_name + '-' + str(metric) +'.jpg', dpi=100)
    plt.show()
    plt.clf()