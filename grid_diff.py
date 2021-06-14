#Script to find difference in base grid values

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

path = '/projects/hpacf/pmadathi/jetcase/314_ambient/'
amrex_plt_file0 = path + 'plt85120'
ds0 = yt.load(amrex_plt_file0)
ds0.print_stats()

amrex_plt_file1 = path + 'plt1_85120'
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

diff = np.abs(np.array((data0['Temp']- data1['Temp'])))
label = diff

#Comment these lines for regression

label[diff<1.e-6]  = 0
label[diff>=1.e-6] = 1

print(label)

#print(diff[:,:,47])

#plt.figure()
#plt.imshow(diff[3,:,:])
#plt.show()
#exit()
#Create appropriate training data (3x3 grid of Temp values centered around the point of interest)

T = np.array(data0['Temp'])

Ti = T[1:-1,1:-1,1:-1] #Exclude boundary points

print(Ti.shape)

s = (Ti.size,3,3,3)
x = np.zeros(s)
xlabel = np.zeros(Ti.size)

for p in range(0,Ti.size):
    i = p%Ti.shape[0]
    j = (p%(Ti.shape[0]*Ti.shape[1]))//Ti.shape[1]
    k = p//(Ti.shape[0]*Ti.shape[1])
    #print(p)
    #print(i,j,k)
    x[p,:,:,:] = T[i:i+3,j:j+3,k:k+3]
    xlabel[p]  = label[i+1,j+1,k+1]

#Test data and training data is created below

##################################################
#Data augmentation using swap axes

xT = np.rot90(x,k=1,axes=(1,2))
y = x
ylabel = xlabel
x = np.append(x,xT, axis=0)
xlabel = np.append(xlabel,xlabel, axis=0)

#################################################

x = x.reshape((x.shape[0],27))
y = y.reshape((y.shape[0],27))
y = (y-np.mean(y))/np.std(y)

##############################################################################
#Creating validation data set

val_index = np.random.choice(np.arange(0,x.shape[0]),500,replace='False')

x_val = x[val_index, :]
x_vallabel = xlabel[val_index]

x_train = np.delete(x, val_index, 0)
x_trainlabel = np.delete(xlabel, val_index, 0)

#############################################################################

test_index = np.random.choice(np.arange(0,Ti.size),100,replace='False')

x_test = x[test_index,:]
x_testlabel = xlabel[test_index]

x_train = np.delete(x_train, test_index, 0)
x_trainlabel = np.delete(x_trainlabel, test_index, 0)

print(x_train.shape,x_trainlabel.shape)

#Normalize data

train_mean = np.mean(x_train)
train_std  = np.std(x_train)

x_train = (x_train - train_mean)/train_std
x_test  = (x_test - train_mean)/train_std
x_val   = (x_val - train_mean)/train_std
    
print(x.shape)

#Model creation

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(8, input_dim=27, activation='relu'))
#model.add(tf.keras.layers.Dense(4, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='relu'))

model.summary()

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

#Fit on training data

history = model.fit(x_train, x_trainlabel, batch_size=32, epochs=5, validation_data=(x_val,x_vallabel))

#Test 

score = model.evaluate(x_test, x_testlabel)
print('Loss: %.2f' % (score[0]))
print('Accuracy: %.2f' % (score[1]*100))

y_predict = model.predict(y[:,:])
err = y_predict[:,0]-ylabel
#print(err.shape, ylabel.shape)
err = np.reshape(err, (T.shape[0]-2,T.shape[1]-2,T.shape[3]-2))

#plt.figure()
#plt.imshow(err[3,:,:])
#plt.show()

exit()

#Test on data at previous time step

#Prepare data

amrex_plt_file3 = '../PeleC/PeleC/Exec/RegTests/PMF/plt0_00010'
ds0 = yt.load(amrex_plt_file3)
ds0.print_stats()

amrex_plt_file4 = '../PeleC/PeleC/Exec/RegTests/PMF/plt1_00010'
ds1 = yt.load(amrex_plt_file4)
ds1.print_stats()

data0 = ds0.all_data()
data1 = ds1.all_data()

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

diff = np.abs(np.array((data0['Temp']- data1['Temp'])))
label = diff

#Comment these lines for regression

label[diff<1.e-6]  = 0
label[diff>=1.e-6] = 1

#print(label)

T = np.array(data0['Temp'])

Ti = T[1:-1,1:-1,1:-1] #Exclude boundary points

print(Ti.shape)

s = (Ti.size,3,3,3)
x = np.zeros(s)
xlabel = np.zeros(Ti.size)

for p in range(0,Ti.size):
    i = p%Ti.shape[0]
    j = (p%(Ti.shape[0]*Ti.shape[1]))//Ti.shape[1]
    k = p//(Ti.shape[0]*Ti.shape[1])
    #print(p)
    #print(i,j,k)
    x[p,:,:,:] = T[i:i+3,j:j+3,k:k+3]
    xlabel[p]  = label[i+1,j+1,k+1]

x = x.reshape((x.shape[0],27))
x = (x-np.mean(x))/np.std(x)

score = model.evaluate(x, xlabel)
print('Loss: %.2f' % (score[0]))
print('Accuracy: %.2f' % (score[1]*100))


