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
import timeit

#Pytorch
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable

#Alibi
import alibi
from alibi.explainers import CEM

# Tensorflow libraries
import tensorflow as tf
from tensorflow import keras
#Keras
from tensorflow.keras.layers import Input, Dense, Lambda, Layer, Add, Multiply
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend as K

#Seed
from tensorflow import set_random_seed
seed=10000
set_random_seed(seed)

def gen_explanation(cem, explain_x):
    return cem.explain(explain_x, verbose=True)

# To time the function gen-explanation
def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped

#Argparsing
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='bn1')
parser.add_argument('--train_case_pred', type=int, default=0)
parser.add_argument('--train_case_ae', type=int, default=0)
parser.add_argument('--explain_case', type=int, default=1, help='0:PP, 1:PN')
parser.add_argument('--kappa', type=float, default=0.01)
parser.add_argument('--gamma', type=float, default=0)
parser.add_argument('--beta', type=float, default=0.1)
parser.add_argument('--c_init', type=float, default=10)
parser.add_argument('--c_steps', type=int, default=50)
parser.add_argument('--max_iterations', type=int, default=1000)
parser.add_argument('--sample_size', type=int, default=3)
parser.add_argument('--htune', type=int, default=0)
parser.add_argument('--test_case', type=int, default=0)
parser.add_argument('--timeit', type=int, default=0, help='0: Dont time')
args = parser.parse_args()

#Main Code
base_data_dir='data/'
if args.htune==1:
    base_model_dir='htune/cem/'
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
    
#Load Train, Val, Test Dataset
vae_train_dataset= np.load(base_data_dir+dataset_name+'-train-set.npy')
np.random.shuffle(vae_train_dataset)

# Validation or Testing Case
if args.test_case:
    vae_val_dataset= np.load(base_data_dir+dataset_name+'-test-set.npy')
    np.random.shuffle(vae_val_dataset)
else:
    vae_val_dataset= np.load(base_data_dir+dataset_name+'-val-small.npy')
    np.random.shuffle(vae_val_dataset)

with open(base_data_dir+dataset_name+'-normalise_weights.json') as f:
    normalise_weights= json.load(f)
normalise_weights = {int(k):v for k,v in normalise_weights.items()}

with open(base_data_dir+dataset_name+'-mad.json') as f:
    mad_feature_weights= json.load(f)

print(normalise_weights)
print(mad_feature_weights)

#Black Box Model
# train, _ = d.split_data(d.normalize_data(d.one_hot_encoded_data))
# X_train = train.loc[:, train.columns != params['outcome_name']].to_numpy()
# y_train = train.loc[:, train.columns == params['outcome_name']].to_numpy()
X_train= vae_train_dataset[:,:-1]
y_train= vae_train_dataset[:,-1]

train_case_pred=args.train_case_pred
train_case_ae=args.train_case_ae

if train_case_pred:
    pred_model = tf.keras.Sequential()
    inp_shape = len(d.encoded_feature_names)
    
    pred_model.add(keras.layers.Dense(10, input_shape=(inp_shape,)))
    pred_model.add(keras.layers.Dense(2, input_shape=(10, ), activation=tf.nn.sigmoid))
    pred_model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(0.01), metrics=['accuracy'])
    pred_model.summary()
    pred_model.fit(X_train, keras.utils.to_categorical(y_train), validation_split=0.20, epochs=100, verbose=1)

    path= base_model_dir + dataset_name + '-keras-pred-model'
    model_json = pred_model.to_json()
    with open(path+".json", "w") as json_file:
        json_file.write(model_json)
    pred_model.save_weights(path+".h5")
else:
    # load json and create model
    path=base_model_dir + dataset_name + '-keras-pred-model'
    json_file = open(path+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    pred_model = tf.keras.models.model_from_json(loaded_model_json)

    # load weights into new model
    pred_model.load_weights(path+".h5")
    pred_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    pred_model.evaluate(X_train, keras.utils.to_categorical(y_train), verbose=1)


#AE Model
data_size= len(d.encoded_feature_names)
encoding_size= 10
def nll(y_true, y_pred):
    """ Negative log likelihood (Bernoulli). """

    # keras.losses.binary_crossentropy gives the mean
    # over the last axis. we require the sum
    return K.sum(keras.losses.mean_squared_error(y_true, y_pred), axis=-1)

if train_case_ae:
    encoder = Sequential([
        Dense(20, input_dim=data_size, activation='relu'),
        Dense(15, input_dim=20, activation='relu'),
        Dense(10, input_dim=15, activation='relu'),
        Dense(encoding_size, input_dim=10),
    ])

    decoder = Sequential([
        Dense(12, input_dim=encoding_size, activation='relu'),
        Dense(14, input_dim=12, activation='relu'),
        Dense(16, input_dim=14, activation='relu'),
        Dense(20, input_dim=16, activation='relu'),
        Dense(data_size, input_dim=20),
    ])

    x = Input(shape=(data_size,))
    z= encoder(x)
    print(z.shape)

    x_pred = decoder(z)
    print(x_pred.shape)

    ae_model = Model(inputs=[x], outputs=x_pred)
    ae_model.compile(optimizer='adam', loss=nll)    
    
    #Train
    batch_size = 64
    epochs = 200
    ae_model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size)  
    
    #Save
    path= base_model_dir + dataset_name + '-keras-ae-model'
    model_json = ae_model.to_json()
    with open(path+".json", "w") as json_file:
        json_file.write(model_json)
    ae_model.save_weights(path+".h5")        
else:
    # load json and create model
    path=base_model_dir + dataset_name + '-keras-ae-model'
    json_file = open(path+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    ae_model = tf.keras.models.model_from_json(loaded_model_json)

    # load weights into new model
    ae_model.load_weights(path+".h5")
    ae_model.compile(loss=nll, optimizer='adam',)
    ae_model.evaluate(X_train, X_train, verbose=1)        
    
#CEM Explainer
#Getting Rid of the label, not required now
vae_val_dataset= vae_val_dataset[:,:-1]
vae_val_dataset= np.array_split( vae_val_dataset, vae_val_dataset.shape[0], axis=0 )

# 'PN' (pertinent negative) or 'PP' (pertinent positive)
if args.explain_case:
    mode = 'PN' 
else:
    mode = 'PP'
shape = (1, int(vae_val_dataset[0].shape[1]) ) # instance shape
print(shape)
kappa = args.kappa  # minimum difference needed between the prediction probability for the perturbed instance on the
            # class predicted by the original instance and the max probability on the other classes
            # in order for the first loss term to be minimized
beta = args.beta  # weight of the L1 loss term
gamma = args.gamma  # weight of the optional auto-encoder loss term
c_init = args.c_init  # initial weight c of the loss term encouraging to predict a different class (PN) or
              # the same class (PP) for the perturbed instance compared to the original instance to be explained
c_steps = args.c_steps  # nb of updates for c
max_iterations = args.max_iterations  # nb of iterations per value of c
feature_range = (0,1)  # feature range for the perturbed instance
clip = (-1000.,1000.)  # gradient clipping
lr = 1e-2  # initial learning rate
no_info_val = 0. # a value, float or feature-wise, which can be seen as containing no info to make a prediction
                  # perturbations towards this value means removing features, and away means adding features
                  # for our MNIST images, the background (-0.5) is the least informative,
                  # so positive/negative perturbations imply adding/removing features
            
# initialize TensorFlow session before model definition
sess = tf.Session()
K.set_session(sess)
sess.run(tf.global_variables_initializer())

# define models
# ae = load_model('mnist_ae.h5')

# initialize CEM explainer and explain instance
cem = CEM(sess, pred_model, mode, shape, ae_model=ae_model, kappa=kappa, beta=beta, feature_range=feature_range,
          gamma=gamma, max_iterations=max_iterations,
          c_init=c_init, c_steps=c_steps, learning_rate_init=lr, clip=clip, no_info_val=no_info_val)

#Generate Explanations
final_result=[]
eval_time={}
for epoch in range(0, args.sample_size):
    result=[]
    for idx in range(0, len(vae_val_dataset)):
        explain_x = vae_val_dataset[idx]
        print(explain_x.shape)       
        explanation = gen_explanation( cem, explain_x ) 
        print(explanation)
        
        if args.timeit:
            #Time the function
            if epoch==0:
                wrapped = wrapper(gen_explanation, cem, explain_x)
                eval_time[str(idx)]= timeit.timeit(wrapped, number=1)
                print('-----------------------------------')
                print('Time taken: ', eval_time[str(idx)])
                print('-----------------------------------')
            
        #Store the result 
        if 'PN' not in explanation.keys():
            continue
        temp={}
        temp['x']= explanation['X'].tolist()
        temp['x_cf'] = explanation['PN'].tolist()
        temp['label']= int(explanation['X_pred']) 
        temp['cf_label']= int(explanation['PN_pred']) 
        result.append(temp)
        
            
        print('Done for instance idx: ', idx)    

    final_result.append(result)
       
    # Save the generated counterfactuals
    if epoch ==0 and args.test_case:
        base_model_dir= base_model_dir + 'test/'   
        
    f=open(base_model_dir + dataset_name + '-sample-size-' + str(args.sample_size) + '-size-' + str(len(vae_val_dataset)) + '-beta-' + str(args.beta) + '-kappa-' + str(args.kappa) +  '-gamma-' + str(args.gamma) + '-cteps-' + str(args.c_steps) + '-maxiter-' + str(args.max_iterations) + '-contrastive-explanations.json', 'w')
    f.write( json.dumps(final_result) )                 
    f.close()    
    
    if args.timeit:
        #Save the time performance of the model
        f=open(base_data_dir+dataset_name + '-gamma-'  + str(args.gamma) + '-time-eval-contrastive-explanations.json', 'w')
        f.write( json.dumps(eval_time) )
        f.close()    

sess.close()
K.clear_session()
