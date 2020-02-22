#Normie stuff
import sys
import pandas as pd
import numpy as np
import json
import sys

base_data_dir='../data/'
dataset_name= sys.argv[1]

f=open(base_data_dir+dataset_name+'-scm.txt', 'r')
data= f.readlines()
f.close()

result={}
idx=0
while( idx<len(data) ):
    old_idx= idx
    count=0
    content_count=0
    while(count!=2):
        line=data[idx]        
        if line=='\n':
            count+=1
        else:
            count=0
            content_count+=1
        idx+=1
            
#     #Debug
#     print('New Node: ', data[old_idx])
#     #Total Content
#     print('Content Num: ', content_count)
#     #Parents
#     print('Parents: ', data[old_idx+2])    
#     print('\n')
    
    #Node    
    node=data[old_idx].replace('\n','').replace('$','')
    parents= data[old_idx+2] 
    result[node]={}
    result[node]['parent']=[]
    result[node]['weight']=[]
    result[node]['sd']=[]
    
    if 'NULL' in data[old_idx+2]:
        continue
        
    for item in parents.split(' '):
        item= item.replace('\n', '')
        item= item.replace('"', '')
        if item=='':
            continue
        result[node]['parent'].append( item )  
        
    if 'Treatment' in data[old_idx+2]:
        intercept= []
        weight=[]
        sd=[]
        
        #Standard Deviation
        sd= data[old_idx+8].split(' ')
        if len(parents.split(' '))==2:
            sd= data[old_idx+9].split(' ')            
        for item in sd:
            temp=[]
            item=item.replace('\n','')
            if item=='':
                continue
            temp.append( float(item) )
            result[node]['sd'].append( temp )
        
        #Intercept
        for item in data[old_idx+6].split(' '):
            item= item.replace('\n', '')
            if item=='':
                continue
            intercept.append(item)
        
        #Weight
        if len( parents.split(' ') ) == 2:    
            for item in data[old_idx+7].split(' '):
                item= item.replace('\n', '')
                if item=='':
                    continue
                weight.append(item)
            
        print( len(intercept), len(weight) )
        for case in range(len(intercept)):
            temp=[]      
            temp.append(float(intercept[case]))            
            if len(weight):
                temp.append(float(weight[case]))                
            result[node]['weight'].append( temp )                        
                       
    else:
        weight= data[old_idx+7].split(' ')
        for item in weight:
            item= item.replace('\n', '')
            if item=='':
                continue
            result[node]['weight'].append( float(item) )
            
        sd= data[old_idx+9].split(' ')
        for item in sd:
            item=item.replace('\n','')
            if item=='':
                continue
            result[node]['sd'].append( float(item) )

for key in result.keys():
    print( 'Node: ', key)
    print( 'Parents: ', result[key]['parent'])
    print( 'Weights: ', result[key]['weight'])
    print( 'SD: ', result[key]['sd'])
    print('\n')

    
#Save the Result
f=open(base_data_dir+dataset_name+'-scm.json', 'w')    
f.write(json.dumps(result, indent=3))
f.close()