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
import sherpa
#from keras import layers


path = '/projects/hpacf/pmadathi/jetcase/314_ambient/'
path1 = path + 'plt0_85101'
path2 = path + 'plt1_85101'

def extract_frm_pltfile(path1,path2):
    # Get data all projected to a uniform grid at the coarsest level
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

    T = (T-np.mean(T))/np.std(T) 
    T1 = (T1-np.mean(T1))/np.std(T1) 

    diff = np.abs(T-T1)
    label = diff 
    
    #Comment these lines for regression
    
    #label[diff<1.e-10]  = 0
    #label[diff>=1.e-10] = 1
    
    print('Max error =',np.max(np.abs(label)), 'Min error =', np.min(np.abs(label)))
    
    #print(diff[:,:,47])
    
    #plt.figure()
    #plt.imshow(diff[:,:,50])
    #plt.show()
    
    #Create appropriate training data (3x3 grid of Temp values centered around the point of interest)
    nvar =  7
    
    u = np.array(data0['x_velocity'])
    v = np.array(data0['y_velocity'])
    w = np.array(data0['z_velocity'])
    e = np.array(data0['eint_e'])
    rho = np.array(data0['density'])
    pr = np.array(data0['pressure'])
    #
    #T = (T-np.mean(T))/np.std(T) 
    u = (u-np.mean(u))/np.std(u) 
    v = (v-np.mean(v))/np.std(v) 
    w = (w-np.mean(w))/np.std(w) 
    e = (e-np.mean(e))/np.std(e) 
    rho = (rho-np.mean(rho))/np.std(rho) 
    pr = (pr-np.mean(pr))/np.std(pr) 
    #

    Tx, Ty, Tz = np.gradient(np.array(data0['Temp']))
    ux, uy, uz = np.gradient(np.array(data0['Temp']))
    vx, vy, vz = np.gradient(np.array(data0['Temp']))
    wx, wy, wz = np.gradient(np.array(data0['Temp']))
    
    T = np.stack((T,u,v,w,e,rho,pr),axis=-1)
    #T = np.stack((Tx,Ty,Tz,ux,uy,uz,vx,vy,vz,wx,wy,wz,pr),axis=-1)

    #T = np.reshape(T,(T.shape[0], T.shape[1], T.shape[2], 1))
    
    l=5 #size of box
    m = l//2
    
    Ti = T[m:-m,m:-m,m:-m,:] #Exclude boundary points
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
        xlabel[p]  = label[i+l//2,j+l//2,k+l//2]
    
    #Test data and training data is created below
    
    #Remove quiscent regions
    #x = np.delete(x, (np.abs(xlabel)<1.e-3).nonzero(), 0)
    #xlabel = np.delete(xlabel, (np.abs(xlabel)<1.e-3).nonzero(), 0)
    #xlabel = (xlabel-np.mean(xlabel))/np.std(xlabel)
    print('Max error =',np.max(np.abs(xlabel)), 'Min error =', np.min(np.abs(xlabel)))
    
    return x, xlabel, nvar, l, T
def extract_frm_downsampledfile(file):
    ds = np.load(file)
    print(ds.files)
    ds_index = ds['indices']
    #print(ds_index)
    return ds_index

##################################################
#Data augmentation using swap axes

#xT = np.rot90(x,k=1,axes=(1,2))
#
#x = np.append(x,xT, axis=0)
#xlabel = np.append(xlabel,xlabel, axis=0)

#################################################
x, xlabel, nvar, l, T = extract_frm_pltfile(path1, path2)
y = x
ylabel = xlabel
#####################################################################################
#Downsample
#####################################################################################

file = '/home/pmadathi/PhaseSpaceSampling/downSampledData_01/downSampledData_10000.npz'
ds_index = extract_frm_downsampledfile(file)
x = x[ds_index,:]
sf = 1.e+3
xlabel = xlabel[ds_index]*sf
ylabel = ylabel*sf
print('Max error =',np.max(np.abs(xlabel)), 'Min error =', np.min(np.abs(xlabel)))
##############################################################################
#Creating validation data set

val_index = np.random.choice(np.arange(0,x.shape[0]),100,replace='False')

x_val = x[val_index, :,:,:,:]
x_vallabel = xlabel[val_index]

x_train = np.delete(x, val_index, 0)
x_trainlabel = np.delete(xlabel, val_index, 0)

#############################################################################

test_index = np.random.choice(np.arange(0,x_train.shape[0]),100,replace='False')

x_test = x[test_index,:,:,:,:]
x_testlabel = xlabel[test_index]

x_train = np.delete(x_train, test_index, 0)
x_trainlabel = np.delete(x_trainlabel, test_index, 0)

print(x_train.shape,x_trainlabel.shape)

print(x.shape)

train_mean = np.mean(x_train)
train_std  = np.std(x_train)

label_mean = np.mean(x_trainlabel)
label_std  = np.std(x_trainlabel)

x_train = (x_train - train_mean)/train_std
x_test  = (x_test - train_mean)/train_std
x_val   = (x_val - train_mean)/train_std

y = (y - train_mean)/train_std

#############################################################################
#Sherpa set up
#############################################################################

parameters = [sherpa.Continuous(name='lr', range=[0.0001, 0.001], scale='log'),
              sherpa.Continuous(name='dropout', range=[0, 0.4]),
              sherpa.Ordinal(name='batch_size', range=[16, 32, 64, 128, 256]),
              sherpa.Ordinal(name='num_filters', range=[4, 8, 16, 32, 64]),
              sherpa.Ordinal(name='num_hidden_units', range=[4, 8, 16, 32])
              ]

alg = sherpa.algorithms.RandomSearch(max_num_trials=50)

study = sherpa.Study(parameters=parameters,algorithm=alg,lower_is_better=False)


#############################################################################
#Model creation
#############################################################################
for trial in study:
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(l,l,l,nvar)))
    #model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Conv3D(filters=trial.parameters['num_filters'], kernel_size=(3,3,3)))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.05))
    model.add(tf.keras.layers.AveragePooling3D(pool_size=(2,2,2)))
    #model.add(tf.keras.layers.Conv3D(filters=8, kernel_size=(3,3,3)))
    #model.add(tf.keras.layers.LeakyReLU(alpha=0.05))
    #model.add(tf.keras.layers.MaxPooling3D(pool_size=(2,2,2)))
    #model.add(tf.keras.layers.Conv3D(filters=64, kernel_size=(3,3,3), activation='relu',input_shape=(l,l,l,nvar)))
    #model.add(tf.keras.layers.MaxPooling3D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(trial.parameters['num_hidden_units'], kernel_regularizer='l1'))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.05))
    model.add(tf.keras.layers.Dropout(trial.parameters['dropout']))
    #model.add(tf.keras.layers.Dense(4, activation='relu', kernel_regularizer='l1'))
    model.add(tf.keras.layers.Dense(1, kernel_regularizer='l1', activation=tf.keras.activations.softplus))
    
    model.summary()
    
    opt = keras.optimizers.Adam(learning_rate=trial.parameters['lr'])
    #reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
    #                              patience=5, min_lr=0.00005)
    model.compile(optimizer= opt, loss='mse', metrics=[tf.keras.metrics.MeanAbsoluteError()])
    for i in range(30):
        model.fit(x_train, x_trainlabel)
        loss, mae = model.evaluate(x_val, x_vallabel)
        study.add_observation(trial=trial, iteration=i,
                          objective=loss,
                          context={'mae': mae})
        if study.should_trial_stop(trial):
           break 
    study.finalize(trial=trial)

#Fit on training data

#history = model.fit(x_train, x_trainlabel, batch_size=128, epochs=30, validation_data=(x_val,x_vallabel))
print(study.get_best_result())
exit()
#Test 

score = model.evaluate(x_test, x_testlabel)
print('Loss: %.2f' % (score[0]))
print('MSE: %.2f' % (score[1]))

y_predict = model.predict(y)
print(y_predict.shape, ylabel.shape)
err = np.abs(y_predict[:,0]-ylabel)
print(np.max(err))

###########################################################################
#Test on different case
###########################################################################
path = '/projects/hpacf/pmadathi/jetcase/350_ambient/'
path1 = path + 'plt0_75346'
path2 = path + 'plt1_75346'

x, xlabel, nvar, l, T = extract_frm_pltfile(path1, path2)
#xlabel = xlabel 
print('Max error =',np.max(np.abs(xlabel)), 'Min error =', np.min(np.abs(xlabel)))
y = x
xlabel = xlabel*sf
ylabel = xlabel
y = (y - train_mean)/train_std

y_predict = model.predict(y)
print(y_predict.shape, ylabel.shape)
ylabel = ylabel/sf
y_predict = y_predict/sf
err = np.abs(y_predict[:,0]-ylabel)
print(np.max(err))


###########################################################################
#Plotting
###########################################################################
#print(err.shape, ylabel.shape)
m =l//2
err = np.reshape(err, (T.shape[0]-2*m,T.shape[1]-2*m,T.shape[2]-2*m))
ax =plt.gca()
ax.set_yscale('log')
ax.set_xscale('log')
ax.scatter(ylabel, y_predict)
plt.xlabel('Actual error')
plt.ylabel('Predicted error')
plt.title('Case: 350 ambient')
ax.set_xlim([min(ylabel*sf),max(ylabel*sf)])
ax.set_ylim([min(ylabel*sf),max(ylabel*sf)])
plt.show()

plt.figure()
plt.imshow(err[:,:,40], cmap = 'Reds') #, vmin = 0, vmax = 50)
plt.colorbar()
plt.show()

y_predict = np.reshape(y_predict, (T.shape[0]-2*m,T.shape[1]-2*m,T.shape[2]-2*m))

plt.figure()
plt.imshow(y_predict[:,:,40], cmap ='Reds') #, vmin = 0, vmax = np.max(ylabel))
plt.colorbar()
plt.show()

ylabel = np.reshape(ylabel, (T.shape[0]-2*m,T.shape[1]-2*m,T.shape[2]-2*m))
plt.figure()
plt.imshow(ylabel[:,:,40], cmap = 'Reds') #, vmin = 0, vmax = np.max(ylabel))
plt.colorbar()

plt.show()

#print(np.max(xlabel-pred[:,0]))