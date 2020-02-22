import json
import numpy as np

path='../data/mnist-time-eval-base-gen-cf.json'
with open(path) as f:
    data = json.load(f)

size=60
exec_time=np.zeros(size)
for i in range(0,size):
    exec_time[i-1]=  data[str(i+1)] - data[str(i)]

print('Test Time Stats')
print(np.mean(exec_time))
print(np.std(exec_time))

train_time= np.array([ 172.9845052352175, 179.16136822476983, 168.70661583263427, 170.41028618440032])
print('Train Time Stats')
print(np.mean(train_time))
print(np.std(train_time))

im1= np.array([ 1.0657, 1.0707, 1.0891, 1.0685, 1.082, 1.0731, 1.0663, 1.07335, 1.0812, 1.0674 ])
print('IM1 Stats')
print(np.mean(im1))
print(np.std(im1))

im2= np.array([ 0.1265, 0.1243, 0.1286, 0.1257, 0.1186, 0.1236, 0.1261, 0.1149, 0.1177, 0.0123 ])
print('IM2 Stats')
print(np.mean(im2))
print(np.std(im2))

im1= np.array([ 3.22432366941796, 3.3280098672773017, 3.2917903446760333, 3.2189397655549596, 3.7864141112468284 ])
print('IM1 Num Stats')
print(np.mean(im1))
print(np.std(im1))

im1= np.array([3.295969411975048, 3.2278139219909416, 3.203402998017483, 3.0802462335492744, 3.774338503352931 ])
print('IM1 Denom Stats')
print(np.mean(im1))
print(np.std(im1))

im2= np.array([ 0.9505369203989623, 0.9875894648129823, 1.0273013974799485, 0.9575712964182994, 1.1870893992361475])
print('IM2 Num Stats')
print(np.mean(im2))
print(np.std(im2))

im2= np.array([ 106.63155377497438,  106.76553557349033, 107.88036484014792, 107.16229948450308, 106.7702422845559 ])
print('IM2 Denom Stats')
print(np.mean(im2))
print(np.std(im2))