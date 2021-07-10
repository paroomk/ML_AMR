#Script to identify cells based on error threshold + ML training to identify cells ith error above threshold

import numpy as np
#from mpi4py import MPI
import matplotlib.pyplot as plt
import matplotlib
import yt
import sherpa
#matplotlib.use('Qt5Agg')
#import os
#os.putenv('DISPLAY', ':0.0')
import tensorflow as tf
from tensorflow import keras


def extract_frm_pltfile(path1, path2):
    min_level = 0

    amrex_plt_file0 = path1
    ds0 = yt.load(amrex_plt_file0)
    ds0.print_stats()

    amrex_plt_file1 = path2
    ds1 = yt.load(amrex_plt_file1)
    ds1.print_stats()

    print("Variables in plt file: ", ds0.field_list)
    print("Number of AMR levels: ", ds0.max_level)

    data0 = ds0.all_data()
    data1 = ds1.all_data()

    print(data0['Temp'].shape)
    print(data0['x'])

    # Get data all projected to a uniform grid at min level
    
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

    print('Max error =',np.max(np.abs(T-T1)), 'Min error =', np.min(np.abs(T-T1)))

    #Get variables of interest

    u = np.array(data0['x_velocity'])
    u1 = np.array(data1['x_velocity'])

    v = np.array(data0['y_velocity'])
    v1 = np.array(data1['y_velocity'])

    w = np.array(data0['z_velocity'])
    w1 = np.array(data1['z_velocity'])

    e = np.array(data0['eint_e'])
    e1 = np.array(data1['eint_e'])

    rho = np.array(data0['density'])
    rho1 = np.array(data1['density'])

    pr = np.array(data0['pressure'])
    pr1 = np.array(data1['pressure'])

    print('Max error =',np.max(np.abs(T-T1)), 'Min error =', np.min(np.abs(T-T1)))

    #split here
    nvar = 7
    X0 = np.stack((T,u,v,w,rho,e,pr),axis=-1)
    X1 = np.stack((T1,u1,v1,w1,rho1,e1,pr1),axis=-1)

    diff = np.abs(X0-X1)
    label = diff

    return X0, X1, nvar

def process_data(x,x_mean,x_std,x1,x1_mean,x1_std):

    n = x.shape[-1]
    print(n)
    for i in range(0,n):
        x[:,:,:,i] = (x[:,:,:,i]/x_mean[i])#/x_std[i] 
        x1[:,:,:,i] = (x1[:,:,:,i]/x1_mean[i])#/x1_std[i] 

    label = np.abs(x1-x)

    #Comment these lines for regression

    #label[label<1.e-4]  = 0
    #label[label>=1.e-4] = 1


    nvar =  7

    Tx, Ty, Tz = np.gradient(x[:,:,:,0])
    ux, uy, uz = np.gradient(x[:,:,:,1])
    vx, vy, vz = np.gradient(x[:,:,:,2])
    wx, wy, wz = np.gradient(x[:,:,:,3])

    T   = x[:,:,:,0]
    u   = x[:,:,:,1]
    v   = x[:,:,:,2]
    w   = x[:,:,:,3]
    rho = x[:,:,:,4]
    e   = x[:,:,:,5]
    pr  = x[:,:,:,6]

    #x = np.stack((Tx,Ty,Tz,ux,uy,uz,vx,vy,vz,wx,wy,wz,rho,e,pr),axis=-1)
    x = np.stack((T,u,v,w,rho,e,pr), axis = -1)
    print(x.shape,label.shape)
    #T = np.reshape(T,(T.shape[0], T.shape[1], T.shape[2], 1))

    xlabel = label[:,:,:,1]
    print('Max error =',np.max(np.abs(xlabel)), 'Min error =', np.min(np.abs(xlabel)))

    ##################################################
    #Data augmentation using swap axes
    
    xT = np.rot90(x,k=1,axes=(0,2))
    xT1 = np.rot90(x,k=2,axes=(0,2))
    xT2 = np.rot90(x,k=3,axes=(0,2))
    

    x = np.append(x,xT, axis=0)
    xlabel = np.append(xlabel,label[:,:,:,1], axis=0)
    x = np.append(x,xT1, axis=0)
    xlabel = np.append(xlabel,label[:,:,:,1], axis=0)
    x = np.append(x,xT2, axis=0)
    xlabel = np.append(xlabel,label[:,:,:,1], axis=0)
    
    print(x.shape,xlabel.shape)
    #################################################

    return x, xlabel, nvar

path = '/projects/hpacf/pmadathi/jetcase/314_ambient/'
path1 = path + 'plt0_85101' #'plt0_75346'
path2 = path + 'plt1_85101' #'plt1_75346'

x, x1, nvar = extract_frm_pltfile(path1, path2)

print(x.shape)

x_mean = np.zeros(nvar)
x_std  = np.zeros(nvar)

x1_mean = np.zeros(nvar)
x1_std  = np.zeros(nvar)

#label_mean = np.zeros(nvar)
#label_std  = np.zeros(nvar)

for i in range(0, nvar): 
    x_mean[i] = np.mean(x[:,:,:,i])
    x_std[i]  = np.std(x[:,:,:,i])

    #x1_mean[i] = np.mean(x1[:,:,:,i])
    #x1_std[i]  = np.std(x1[:,:,:,i])

    #label_mean[i] = np.mean(xlabel[:,:,:,i])
    #label_std[i] = np.std(xlabel[:,:,:,i])
 
print(x_mean, x_std)

x, xlabel, nvar = process_data(x, x_mean, x_std, x1, x_mean, x_std)
print('Max error =',np.max(np.abs(xlabel)), 'Min error =', np.min(np.abs(xlabel)))

ly = x.shape[1]
lz = x.shape[2]

xlabel = xlabel.reshape(xlabel.shape[0],ly*lz)

sf = 1.e+2
xlabel = xlabel*sf

y = x
ylabel = xlabel

print(y.shape, ylabel.shape)

##############################################################################
#Creating validation and test data set
##############################################################################
val_index = np.random.choice(np.arange(0,x.shape[0]),x.shape[0]//5,replace='False')

x_val = x[val_index, :,:,:]
x_vallabel = xlabel[val_index,:]

x_train = np.delete(x, val_index, 0)
x_trainlabel = np.delete(xlabel, val_index, 0)

test_index = np.random.choice(np.arange(0,x_train.shape[0]),5,replace='False')

x_test = x[test_index,:,:,:]
x_testlabel = xlabel[test_index,:]

x_train = np.delete(x_train, test_index, 0)
x_trainlabel = np.delete(x_trainlabel, test_index, 0)

print(x_train.shape,x_trainlabel.shape)

#############################################################################
#Normalization (features + labels)
#############################################################################

train_mean = np.zeros(nvar)
train_std = np.zeros(nvar)

for i in range(0, nvar):
    train_mean[i] = np.mean(x_train[:,i])
    train_std[i]  = np.std(x_train[:,i])
    
    x_train[:,i] = (x_train[:,i] - train_mean[i])/train_std[i]
    x_test[:,i]  = (x_test[:,i]- train_mean[i])/train_std[i]
    x_val[:,i]   = (x_val[:,i] - train_mean[i])/train_std[i]
    y[:,i]       = (y[:,i] - train_mean[i])/train_std[i]

label_mean = np.mean(x_trainlabel)
label_std  = np.std(x_trainlabel)

#x_trainlabel = (x_trainlabel - label_mean)/label_std
#x_testlabel = (x_testlabel - label_mean)/label_std
#x_vallabel = (x_vallabel - label_mean)/label_std
#ylabel = (ylabel - label_mean)/label_std
print()
print('Max error =',np.max(np.abs(ylabel)), 'Min error =', np.min(np.abs(ylabel)))

#############################################################################
#Model creation
#############################################################################

model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(ly,lz,nvar)))
#model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', kernel_regularizer='l1'))
#model.add(tf.keras.layers.LeakyReLU(alpha=0.05))
#model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu', kernel_regularizer='l1'))
#model.add(tf.keras.layers.LeakyReLU(alpha=0.05))
#model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
#model.add(tf.keras.layers.Conv2D(filters=14, kernel_size=(3,3), activation='relu'))
#model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(32, kernel_regularizer='l1', activation='relu'))
#model.add(tf.keras.layers.LeakyReLU(alpha=0.05))
#model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.11))
model.add(tf.keras.layers.Dense(4, activation='relu', kernel_regularizer='l1'))
model.add(tf.keras.layers.Dense(ly*lz, kernel_regularizer='l1', activation='relu'))
#model.add(tf.keras.layers.LeakyReLU(alpha=0.05))

model.summary()

opt = keras.optimizers.Adam(learning_rate=0.0001)
#reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
#                              patience=5, min_lr=0.00005)
model.compile(optimizer= opt, loss='mse', metrics=[tf.keras.metrics.MeanAbsoluteError()])

###############################################################################################
#Fit on training data
###############################################################################################

history = model.fit(x_train, x_trainlabel, batch_size=32, epochs=200, validation_data=(x_val,x_vallabel))

###############################################################################################
#Test 
###############################################################################################

score = model.evaluate(x_test, x_testlabel)
print('Loss: %.2f' % (score[0]))
print('MAE: %.2f' % (score[1]))

###########################################################################
#Test on different case
###########################################################################
path = '/projects/hpacf/pmadathi/jetcase/350_ambient/'
path1 = path + 'plt0_75346'
path2 = path + 'plt1_75346'

x, x1, nvar = extract_frm_pltfile(path1, path2)
for i in range(0, nvar): 
    x_mean[i] = np.mean(x[:,:,:,i])
    x_std[i]  = np.std(x[:,:,:,i])
    #x1_mean[i] = np.mean(x1[:,:,:,i])
    #x1_std[i]  = np.std(x1[:,:,:,i])
    #label_mean[i] = np.mean(xlabel[:,:,:,i])
    #label_std[i] = np.std(xlabel[:,:,:,i])

y, ylabel, nvar = process_data(x, x_mean, x_std, x1, x_mean, x_std)
ylabel = ylabel.reshape(ylabel.shape[0],ly*lz)
##xlabel = xlabel 
print('Max error =',np.max(np.abs(xlabel)), 'Min error =', np.min(np.abs(xlabel)))
ylabel = ylabel*sf
#ylabel = (ylabel - label_mean)/label_std
####exit
###
for i in range(0, nvar):
    y[:,i] = (y[:,i]-train_mean[i])/train_std[i]
##
#print('Max error =',np.max(np.abs(ylabel)), 'Min error =', np.min(np.abs(ylabel)))



#ax =plt.gca()
#ax.set_yscale('log')
#ax.set_xscale('log')
#ax.scatter(ylabel, y_predict)
#plt.xlabel('Actual error')
#plt.ylabel('Predicted error')
#plt.title('Case: 314 ambient')
#ax.set_xlim([min(ylabel),max(ylabel)])
#ax.set_ylim([min(ylabel),max(ylabel)])
#plt.show()
m = 40
y_predict = model.predict(y[m:m+1,:,:,:])
ylabel = ylabel/sf
y_predict = y_predict/sf
err = np.abs(y_predict-ylabel[m,:])
print(np.max(err))
err = np.reshape(err,(ly,lz))
y_predict = np.reshape(y_predict,(ly,lz))
y_label = np.reshape(ylabel[m,:],(ly,lz))

plt.figure()
plt.imshow(err, cmap = 'Reds') #, vmin = 0, vmax = 50)
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(y_predict, cmap = 'Reds') #, vmin = 0, vmax = 50)
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(y_label, cmap = 'Reds') #, vmin = 0, vmax = 50)
plt.colorbar()
plt.show()
