import sys
import pandas as pd
import numpy as np
import json

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#Debug
# import seaborn as  sns 
# x1=np.linspace(-10,110,10)
# x2=np.linspace(-10,110,10)
# x3=[]
# for i in range(x1.shape[0]):
#     temp=[]
#     for j in range(x2.shape[0]):
#         temp.append( 10*((x1[i]+x2[j])**2/180**2) + 10 ) 
#     x3.append( temp )
# x3=np.array(x3)
# print(x3.shape)

# ax = sns.heatmap(np.rot90(x3), linewidth=0.5, cmap='YlGnBu')
# plt.xticks( range(10), np.round(x1,2), rotation=45)
# plt.yticks( range(10), np.flip(np.round(x2,2)), rotation=45)
# plt.show()

# y=[]
# for i in range(x1.shape[0]):
#     temp=[]
#     for j in range(x2.shape[0]):
#         temp.append(  13*((x1[i]*x2[j])/8100) + 10 - x3[i][j] )
#     y.append(temp)
# y= sigmoid( np.array(y) )

# ax = sns.heatmap(np.rot90(y), linewidth=0.5, cmap='YlGnBu')
# plt.xticks( range(10), np.round(x1,2), rotation=45)
# plt.yticks( range(10), np.flip(np.round(x2,2)), rotation=45)
# plt.show()

# print(np.sum(y<0.5), np.sum(y>0.5))

graph_nodes_count=4
x1= np.random.normal(50, 15, 1000)
x2= np.random.normal(50, 17, 1000)
x3= 10*((x1+x2)**2/180**2) + 10 + np.random.normal(0,0.5,1000)
y= sigmoid( 10.5*((x1*x2)/8100) + 10 - x3 )

graph_data=np.zeros( (x1.shape[0], graph_nodes_count)  )
graph_data[:,0]= x1
graph_data[:,1]= x2
graph_data[:,2]= x3
for i in range(y.shape[0]):
    if y[i] > 0.5:
        graph_data[i,3]= 1
    else:
        graph_data[i,3]= 0        
graph_data= pd.DataFrame(graph_data, columns=['x1', 'x2', 'x3', 'y'] )
graph_data.to_csv('../data/bn1.csv')

print( x3- 10*((x1+x2)**2/81**2) - 10)
