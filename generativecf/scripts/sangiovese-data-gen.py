import sys
import pandas as pd
import numpy as np
import json

base_data_dir='../data/'
dataset_name=sys.argv[1]

dataset= pd.read_csv(base_data_dir + dataset_name +'_master.csv', index_col=None)
print(dataset.shape)

# BunchN=  1.21770061*dataset['SproutN'] -0.04485152
# print(dataset['BunchN']-BunchN)
# dataset['BunchN']= BunchN

#dataset= dataset[ dataset['Treatment'] == 'T2a' ]
dataset.drop(columns=['Treatment'], axis=1, inplace=True)
print(dataset.shape)
dataset.to_csv(base_data_dir + dataset_name +'.csv', index=False)