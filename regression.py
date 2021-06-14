#Script to identify cells based on error threshold + ML training to identify cells ith error above threshold

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import yt
#matplotlib.use('Qt5Agg')
import os
#os.putenv('DISPLAY', ':0.0')
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


amrex_plt_file0 = '../PeleC/PeleC/Exec/RegTests/PMF/plt0_00020'
ds0 = yt.load(amrex_plt_file0)
ds0.print_stats()

amrex_plt_file1 = '../PeleC/PeleC/Exec/RegTests/PMF/plt1_00020'
ds1 = yt.load(amrex_plt_file1)
ds1.print_stats()

print("Variables in plt file: ", ds0.field_list)
print("Number of AMR levels: ", ds0.max_level)

data0 = ds0.all_data()
data1 = ds1.all_data()

print(data0['Temp'].shape)
print(data0['x'])

# Get data all projected to a uniform grid at the coarsest level
min_level = 0

ref0 = int(np.product(ds0.ref_factors[0:min_level]))
ref1 = int(np.product(ds1.ref_factors[0:min_level]))

low0 = ds0.domain_left_edge
low1 = ds1.domain_left_edge
dims0 = ds0.domain_dimensions * ref0
dims1 = ds1.domain_dimensions * ref1
# without interpolation
# data = ds.covering_grid(max_level, left_edge=low, dims=dims, num_ghost_zones=1)
# with interpolation
data0 = ds0.smoothed_covering_grid(min_level, left_edge=low0, dims=dims0, num_ghost_zones=1)
data1 = ds1.smoothed_covering_grid(min_level, left_edge=low1, dims=dims1, num_ghost_zones=1)

print(data0['Temp'].shape)
print(data1['Temp'].shape)
T = np.array(data0['Temp'])
T1 = np.array(data1['Temp'])

diff = np.abs(T1- T)
label = diff 

#Comment these lines for regression

#label[diff<1.e-10]  = 0
#label[diff>=1.e-10] = 1

print(np.max(label))

#print(diff[:,:,47])

#plt.figure()
#plt.imshow(diff[:,:,50])
#plt.show()

#Create appropriate training data (3x3 grid of Temp values centered around the point of interest)


u = np.array(data0['x_velocity'])
v = np.array(data0['y_velocity'])
w = np.array(data0['z_velocity'])
rho = np.array(data0['density'])
pr = np.array(data0['pressure'])

nvar =  1 + 5

T = (T-np.mean(T))/np.std(T) 
u = (u-np.mean(u))/np.std(u) 
v = (v-np.mean(v))/np.std(v) 
w = (w-np.mean(w))/np.std(w) 
rho = (rho-np.mean(rho))/np.std(rho) 
pr = (pr-np.mean(pr))/np.std(pr) 

T = np.stack((T,u,v,w,rho,pr),axis=-1)

Ti = T[1:-1,1:-1,1:-1,:] #Exclude boundary points
print(Ti.shape)

s = (Ti.size//nvar,3,3,3,nvar)

x = np.zeros(s)
print(x.shape)
xlabel = np.zeros(Ti.size//nvar)

for p in range(0,Ti.size//nvar):
    i = p%Ti.shape[0]
    j = (p%(Ti.shape[0]*Ti.shape[1]))//Ti.shape[1]
    k = p//(Ti.shape[0]*Ti.shape[1])
    #print(p)
    #print(i,j,k)
    for m in range(0,nvar):
        x[p,:,:,:,m] = T[i:i+3,j:j+3,k:k+3,m]
    xlabel[p]  = label[i+1,j+1,k+1]

#Test data and training data is created below

##################################################
#Data augmentation using swap axes

xT = np.rot90(x,k=1,axes=(1,2))
#
x = np.append(x,xT, axis=0)
xlabel = np.append(xlabel,xlabel, axis=0)

#################################################
x = x.reshape((x.shape[0],nvar*3**3))
##############################################################################
#Creating validation data set

val_index = np.random.choice(np.arange(0,x.shape[0]),500,replace='False')

x_val = x[val_index, :]
x_vallabel = xlabel[val_index]

x_train = np.delete(x, val_index, 0)
x_trainlabel = np.delete(xlabel, val_index, 0)

#############################################################################

test_index = np.random.choice(np.arange(0,Ti.size//nvar),100,replace='False')

x_test = x[test_index,:]
x_testlabel = xlabel[test_index]

x_train = np.delete(x_train, test_index, 0)
x_trainlabel = np.delete(x_trainlabel, test_index, 0)

print(x_train.shape,x_trainlabel.shape)

print(x.shape)

#Model creation

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(16, input_dim=nvar*3**3, activation='relu', kernel_regularizer='l1'))
model.add(tf.keras.layers.Dense(8, activation='relu', kernel_regularizer='l1'))
model.add(tf.keras.layers.Dense(1, activation='relu', kernel_regularizer='l1'))

model.summary()

model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.MeanAbsoluteError()])

#Fit on training data

history = model.fit(x_train, x_trainlabel, batch_size=32, epochs=100, validation_data=(x_val,x_vallabel))

#Test 

score = model.evaluate(x_test, x_testlabel)
print('Loss: %.2f' % (score[0]))
print('MSE: %.2f' % (score[1]))

print(model.predict(x_test[:100,:]))
print(x_testlabel)
