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

def extract_frm_pltfile(path):

    amrex_plt_file0 = path + 'plt2_85101'
    ds0 = yt.load(amrex_plt_file0)
    ds0.print_stats()

    amrex_plt_file1 = path + 'plt1_85101'
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

    print('Max error =',np.max(np.abs(label)), 'Min error =', np.min(np.abs(label)))

    #print(diff[:,:,47])

    #plt.figure()
    #plt.imshow(diff[:,:,50])
    #plt.show()

    #Create appropriate training data (3x3 grid of Temp values centered around the point of interest)
    nvar =  4

    u = np.array(data0['divu'])
    v = np.array(data0['magvort'])
    w = np.array(data0['magvel'])
    #rho = np.array(data0['density'])
    #pr = np.array(data0['pressure'])
    #
    T = (T-np.mean(T))/np.std(T) 
    u = (u-np.mean(u))/np.std(u) 
    v = (v-np.mean(v))/np.std(v) 
    w = (w-np.mean(w))/np.std(w) 
    #rho = (rho-np.mean(rho))/np.std(rho) 
    #pr = (pr-np.mean(pr))/np.std(pr) 
    #
    T = np.stack((T,u,v,w),axis=-1)
    #T = np.reshape(T,(T.shape[0], T.shape[1], T.shape[2], 1))

    l=1 #size of box
    m = l//2

    Ti = T #[m:-m,m:-m,m:-m,:] #Exclude boundary points
    print(Ti.shape)

    s = (Ti.size//nvar,l,l,l,nvar)

    x = np.zeros(s)
    print(x.shape)
    xlabel = np.zeros(Ti.size//nvar)

    for p in range(0,Ti.size//nvar):
        i = p%Ti.shape[0]
        j = (p%(Ti.shape[0]*Ti.shape[1]))//Ti.shape[1]
        k = p//(Ti.shape[0]*Ti.shape[1])
        #print(p)
        #print(i,j,k)
        for m in range(0, nvar):
            x[p,:,:,:,m] = T[i:i+l,j:j+l,k:k+l,m]
            #x[p,:,:,:,m] = T[i+l//2,j+l//2,k+l//2,m]
        xlabel[p]  = label[i+l//2,j+l//2,k+l//2]

    #Test data and training data is created below

    #xlabel = (xlabel-np.mean(xlabel))/np.std(xlabel)
    #print('Max error =',np.max(np.abs(xlabel)), 'Min error =', np.min(np.abs(xlabel)))

    x = x.reshape((x.shape[0],nvar*l**3))
    #xlabel = xlabel.reshape((xlabel.shape[0],1))

    #x = np.append(x,xlabel,axis=1)

    return x, xlabel, nvar, l

def extract_frm_downsampledfile(file):
    ds = np.load(file)
    print(ds.files)
    ds_index = ds['indices']
    #print(ds_index)
    return ds_index

path = '/projects/hpacf/pmadathi/jetcase/314_ambient/'
x, xlabel, nvar, l = extract_frm_pltfile(path)
xlabel = xlabel*1.e+4
print('Max error =',np.max(np.abs(xlabel)), 'Min error =', np.min(np.abs(xlabel)))
#exit()
##############################################################################
#Downsampled data
##############################################################################

file = '/home/pmadathi/PhaseSpaceSampling/downSampledData_10000.npz'
ds_index = extract_frm_downsampledfile(file)
x = x[ds_index,:]
xlabel = xlabel[ds_index]
print('Max error =',np.max(np.abs(xlabel)), 'Min error =', np.min(np.abs(xlabel)))
##############################################################################
#Creating validation data set
##############################################################################

val_index = np.random.choice(np.arange(0,x.shape[0]),x.shape[0]//5,replace='False')

x_val = x[val_index, :]
x_vallabel = xlabel[val_index]

x_train = np.delete(x, val_index, 0)
x_trainlabel = np.delete(xlabel, val_index, 0)

##############################################################################
#Creating test data set
##############################################################################
test_index = np.random.choice(np.arange(0,x.shape[0]),x.shape[0]//2,replace='False')

x_test = x[test_index,:]
x_testlabel = xlabel[test_index]

x_train = np.delete(x_train, test_index, 0)
x_trainlabel = np.delete(x_trainlabel, test_index, 0)

print(x_train.shape,x_trainlabel.shape)

#exit()

#print(x.shape)

#############################################################################
#Model creation
#############################################################################

model = tf.keras.Sequential()
#Fully connected network
model.add(tf.keras.layers.Dense(128, input_dim=nvar*l**3, activation='relu', kernel_regularizer='l1'))
#model.add(tf.keras.layers.Dense(128, input_dim=nvar*l**3, kernel_regularizer='l1'))
#model.add(tf.keras.layers.LeakyReLU(alpha=0.05))
model.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer='l1'))
#model.add(tf.keras.layers.Dense(128, kernel_regularizer='l1'))
#model.add(tf.keras.layers.LeakyReLU(alpha=0.05))
model.add(tf.keras.layers.Dense(1, activation='relu', kernel_regularizer='l1'))
#model.add(tf.keras.layers.Dense(1, kernel_regularizer='l1'))
#model.add(tf.keras.layers.LeakyReLU(alpha=0.05))

model.summary()
opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer= opt, loss='mse', metrics=[tf.keras.metrics.MeanAbsoluteError()])

#############################################################################
#Fit on training data
#############################################################################

history = model.fit(x_train, x_trainlabel, batch_size=128, epochs=5000, validation_data=(x_val,x_vallabel))

#############################################################################
#Test 
#############################################################################

score = model.evaluate(x_test, x_testlabel)
print('Loss: %.2f' % (score[0]))
print('MSE: %.2f' % (score[1]))

pred = model.predict(x_test[:100:])
print(x_testlabel[:100], pred[:,0])
pred = model.predict(x)
print(np.max(xlabel), pred[np.argmax(xlabel),0])
print(np.max(xlabel-pred[:,0]))
