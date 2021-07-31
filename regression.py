#Script to identify cells based on error threshold + ML training to identify cells ith error above threshold

from socket import MSG_EOR
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


def extract_frm_pltfile(path1, path2, level, train, file):
    min_level = level

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
    u1 = np.array(data1['x_velocity']) 
    
    print('Max error =',np.max(np.abs(T-T1)), 'Min error =', np.min(np.abs(T-T1)))

    #print(diff[:,:,47])

    #plt.figure()
    #plt.imshow(diff[:,:,50])
    #plt.show()

    #Create appropriate training data (3x3 grid of Temp values centered around the point of interest)

    u = np.array(data0['x_velocity'])
    v = np.array(data0['y_velocity'])
    w = np.array(data0['z_velocity'])
    e = np.array(data0['eint_e'])
    rho = np.array(data0['density'])
    pr = np.array(data0['pressure'])
    
    print(np.std(T),np.std(u),np.std(v),np.std(w),np.std(rho),np.std(e),np.std(pr))
    print(np.mean(rho),np.mean(e),np.mean(pr))
    #exit()
    #split here
    X0 = T
    # return X0, X1
    ##############################################################
    #def process_data(X0,X0.mean(axis = 0), X0.std(axis = 0),X1):
    #for loop here:
    u = (u-np.mean(u))/np.std(u) 
    v = (v-np.mean(v))/np.std(v) 
    w = (w-np.mean(w))/np.std(w) 
    e = (e-np.mean(e))/np.std(e) 
    rho = (rho-np.mean(rho))/np.std(rho) 
    pr = (pr-np.mean(pr))/np.std(pr) 

    T = (T-np.mean(T))/np.std(T) 
    T1 = (T1-np.mean(T1))/np.std(T1) 
    u1 = (u1-np.mean(u1))/np.std(u1) 

    diff = np.abs(u-u1)
    label = diff

    #Comment these lines for regression
    
    #label[diff<1.e-3]  = 0
    #label[diff>=1.e-3] = 1

    ratio = np.sum(label)/np.size(label)
    print(ratio)
    #exit()

    print('Max error =',np.max(np.abs(T-T1)), 'Min error =', np.min(np.abs(T-T1)))

    nvar =  15

    Tx, Ty, Tz = np.gradient(np.array(data0['Temp']))
    ux, uy, uz = np.gradient(np.array(data0['x_velocity']))
    vx, vy, vz = np.gradient(np.array(data0['y_velocity']))
    wx, wy, wz = np.gradient(np.array(data0['z_velocity']))
    
    #T = np.stack((T,u,v,w,rho,e,pr),axis=-1)
    T = np.stack((Tx,Ty,Tz,ux,uy,uz,vx,vy,vz,wx,wy,wz,rho,e,pr),axis=-1)
    #T = np.reshape(T,(T.shape[0], T.shape[1], T.shape[2], 1))

    l=3
     #size of box
    m = l//2

    if m==0:
        Ti = T
    else:
        Ti = T [m:-m,m:-m,m:-m,:] #Exclude boundary points

    print(Ti.shape)

    if (train):
       ds_index = extract_frm_downsampledfile(file)
       indices = ds_index
    else:
       i, j = np.meshgrid(np.arange(Ti.shape[0]),np.arange(Ti.shape[1]))
       indices = 40*Ti.shape[0]*Ti.shape[1] + i + j*Ti.shape[0]
       indices = indices.reshape((indices.size))
       #indices = np.arange(0,Ti.size//nvar)
       
    s = (indices.size,l,l,l,nvar)
    x = np.zeros(s)
    print(x.shape)
    xlabel = np.zeros(indices.size)
    n=0

    for p in indices:
        i = p%Ti.shape[0]
        j = (p%(Ti.shape[0]*Ti.shape[1]))//Ti.shape[0]
        k = p//(Ti.shape[0]*Ti.shape[1])
        #print(p)
        #print(i,j,k)
        for m in range(0, nvar):
            x[n,:,:,:,m] = T[i:i+l,j:j+l,k:k+l,m]
        xlabel[n]  = label[i+l//2,j+l//2,k+l//2]
        n = n+1

    #xlabel = (xlabel-np.mean(xlabel))/np.std(xlabel)
    print('Max error =',np.max(np.abs(xlabel)), 'Min error =', np.min(np.abs(xlabel)))

    x = x.reshape((x.shape[0],nvar*l**3))
    xlabel = xlabel.reshape((xlabel.shape[0],1))

    #x = np.append(x,xlabel,axis=1)

    return x, xlabel, nvar, l, T

def extract_frm_downsampledfile(file):
    ds = np.load(file)
    print(ds.files)
    ds_index = ds['indices']
    #print(ds_index)
    return ds_index

path = '/projects/hpacf/pmadathi/jetcase/314_ambient/'
path1 = path + 'plt0_85101'
path2 = path + 'plt1_85101'
#path = '/projects/hpacf/pmadathi/PeleC/Exec/RegTests/pelecjetcase/'
#path1 = path + 'plt85092' #'plt0_75346'
#path2 = path + 'plt185092' #'plt1_75346'

level = 0
train = True
file = '/home/pmadathi/PhaseSpaceSampling/downSampledData_01/downSampledData_10000.npz'
x, xlabel, nvar, l, T = extract_frm_pltfile(path1, path2, level, train, file)
print('Max error =',np.max(np.abs(xlabel)), 'Min error =', np.min(np.abs(xlabel)))
##############################################################################
#Downsampled data
##############################################################################

sf = 1.e+0
xlabel = xlabel*sf

print('Max error (02) =',np.max(np.abs(xlabel)), 'Min error =', np.min(np.abs(xlabel)))

xx = x #np.concatenate((x,x1),axis=0)
xxlabel = xlabel #np.concatenate((xlabel,xlabel1),axis=0)
##############################################################################
#Creating validation data set
##############################################################################

val_index = np.random.choice(np.arange(0,xx.shape[0]),xx.shape[0]//5,replace='False')

x_val = xx[val_index, :]
x_vallabel = xxlabel[val_index]

x_train = np.delete(xx, val_index, 0)
x_trainlabel = np.delete(xxlabel, val_index, 0)

##############################################################################
#Creating test data set
##############################################################################
test_index = np.random.choice(np.arange(0,x_train.shape[0]),xx.shape[0]//5,replace='False')

x_test = x_train[test_index,:]
x_testlabel = x_trainlabel[test_index]

x_train = np.delete(x_train, test_index, 0)
x_trainlabel = np.delete(x_trainlabel, test_index, 0)
train_mean = np.zeros(nvar)
train_std = np.zeros(nvar)

for i in range(0, nvar):
    train_mean[i] = np.mean(x_train[:,i])
    train_std[i]  = np.std(x_train[:,i])
    
    x_train[:,i] = (x_train[:,i] - train_mean[i])/train_std[i]
    x_test[:,i]  = (x_test[:,i]- train_mean[i])/train_std[i]
    x_val[:,i]   = (x_val[:,i] - train_mean[i])/train_std[i]

label_mean = np.mean(x_trainlabel)
label_std  = np.std(x_trainlabel)
print(train_mean)
print(train_std)
#exit()
#x_trainlabel = (x_trainlabel - label_mean)/label_std
#x_testlabel = (x_testlabel - label_mean)/label_std
#x_vallabel = (x_vallabel - label_mean)/label_std

print('Max error =',np.max(np.abs(x_trainlabel)), 'Min error =', np.min(np.abs(x_trainlabel)))
#############################################################################
#Sherpa set up
#############################################################################

#parameters = [sherpa.Continuous(name='lr', range=[0.0001, 0.001], scale='log'),
#              sherpa.Continuous(name='dropout', range=[0, 0.4]),
#              sherpa.Ordinal(name='batch_size', range=[16, 32, 64, 128, 256]),
#              sherpa.Ordinal(name='num_hidden_units1', range=[16, 32, 64, 128]),
#              sherpa.Ordinal(name='num_hidden_units2', range=[8, 16, 32, 64, 128])]
#
#alg = sherpa.algorithms.RandomSearch(max_num_trials=50)
#
#study = sherpa.Study(parameters=parameters,algorithm=alg,lower_is_better=False)

#############################################################################
#Model creation
#############################################################################

#for trial in study:
#    model = tf.keras.Sequential()
#    #Fully connected network
#    model.add(tf.keras.layers.Dense(trial.parameters['num_hidden_units1'], input_dim=nvar*l**3, activation='relu', kernel_regularizer='l1'))
#    model.add(tf.keras.layers.BatchNormalization())
#    #model.add(tf.keras.layers.Dense(128, input_dim=nvar*l**3, kernel_regularizer='l1'))
#    #model.add(tf.keras.layers.LeakyReLU(alpha=0.05))
#    model.add(tf.keras.layers.Dense(trial.parameters['num_hidden_units2'], activation='relu', kernel_regularizer='l1'))
#    model.add(tf.keras.layers.BatchNormalization())
#    model.add(tf.keras.layers.Dropout(trial.parameters['dropout']))
#    model.add(tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer='l1'))
#    #model.add(tf.keras.layers.LeakyReLU(alpha=0.05))
#    model.summary()
#    opt = keras.optimizers.Adam(learning_rate=trial.parameters['lr'])
#    #reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
#                                  #patience=5, min_lr=1.e-6)
#    model.compile(optimizer= opt, loss='mse', metrics=['accuracy'])
#    
#    #############################################################################
#    #Fit on training data
#    #############################################################################
#    for i in range(30):
#        model.fit(x_train, x_trainlabel)
#        loss, accuracy = model.evaluate(x_val, x_vallabel)
#        study.add_observation(trial=trial, iteration=i,
#                          objective=loss,
#                          context={'accuracy': accuracy})
#        if study.should_trial_stop(trial):
#           break 
#    study.finalize(trial=trial)
#
#print(study.get_best_result())
#exit()
    
#history = model.fit(x_train, x_trainlabel, batch_size=256, epochs=250, validation_data=(x_val,x_vallabel)) #, callbacks=[study.keras_callback(trial, objective_name='val_loss')])

class fcnn(tf.keras.Model):
  def __init__(self):
    super(fcnn, self).__init__()
    self.dense1 = tf.keras.layers.Dense(64, input_dim=nvar*l**3, activation=tf.nn.relu, kernel_regularizer='l1')
    self.bn1    = tf.keras.layers.BatchNormalization()
    self.dense2 = tf.keras.layers.Dense(128, activation=tf.nn.relu, kernel_regularizer='l1')
    self.bn2    = tf.keras.layers.BatchNormalization()
    self.drop   = tf.keras.layers.Dropout(0.37)
    self.dense3 = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, kernel_regularizer='l1')
  def call(self, inputs):
    x = self.dense1(inputs)
    x = self.bn1(x)
    x = self.dense2(x)
    x = self.bn2(x)
    x = self.drop(x)
    return self.dense3(x)

model = fcnn()
model.compile(
          loss      = tf.keras.losses.MeanSquaredError(),
          metrics   = ['accuracy'],
          optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001))
# fit 
model.fit(x_train, x_trainlabel, batch_size=32, epochs=150, validation_data=(x_val,x_vallabel))

model.save('fcnn', save_format='tf')
#############################################################################
#Test 
#############################################################################


score = model.evaluate(x_test, x_testlabel)
print('Loss: %.2f' % (score[0]))#
print('MSE: %.2f' % (score[1]))
###########################################################################
#Test on different case
###########################################################################
#path = '/projects/hpacf/pmadathi/jetcase/350_ambient/'
#path1 = path + 'plt1_75346'
#path2 = path + 'plt2_75346'

path = '/projects/hpacf/pmadathi/PeleC/Exec/RegTests/pelecjetcase/'
path1 = path + 'plt85092'
path2 = path + 'plt185092'

y, ylabel, nvar, l, T = extract_frm_pltfile(path1, path2, 0, False, file)
print('Max error =',np.max(np.abs(xlabel)), 'Min error =', np.min(np.abs(xlabel)))
ylabel = ylabel*sf

for i in range(0, nvar):
    y[:,i] = (y[:,i] - train_mean[i])/train_std[i]

#ylabel = (ylabel - label_mean)/label_std
print('Max error =',np.max(np.abs(ylabel)), 'Min error =', np.min(np.abs(ylabel)))
#exit()
score = model.evaluate(y, ylabel)
print('Loss: %.2f' % (score[0]))
print('Accuracy: %.2f' % (score[1]))

print(y[14892,:])
y_predict = model.predict(y)
print(y_predict[14892])

ratio = np.sum(np.round(y_predict))/np.size(y_predict)
print(ratio)
ratio = np.size(ylabel[ylabel>0.9])/np.size(ylabel)
print(ratio)
#exit()

print(y_predict.shape, ylabel.shape)
ylabel = ylabel/sf
y_predict = y_predict/sf
err = np.abs((y_predict)-ylabel)
print(np.max(err))
ylabel = np.abs(ylabel)
y_predict = np.abs(y_predict)
###########################################################################
#Plotting
###########################################################################
#print(err.shape, ylabel.shape)
m =l//2
#err = np.reshape(err, (T.shape[0]-2*m,T.shape[1]-2*m,T.shape[2]-2*m))
err = np.reshape(err, (T.shape[1]-2*m,T.shape[0]-2*m,1))
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

plt.figure()
plt.imshow(err[:,:,0], cmap = 'Reds') #, vmin = -1, vmax = 1)
plt.colorbar()
plt.show()

#y_predict = np.reshape(y_predict, (T.shape[0]-2*m,T.shape[1]-2*m,T.shape[2]-2*m))
y_predict = np.reshape(y_predict, (T.shape[1]-2*m,T.shape[0]-2*m,1))

plt.figure()
plt.imshow(y_predict[:,:,0], cmap ='Reds')#, vmin = 0, vmax = 1)
plt.colorbar()
plt.show()

#ylabel = np.reshape(ylabel, (T.shape[0]-2*m,T.shape[1]-2*m,T.shape[2]-2*m))
ylabel = np.reshape(ylabel, (T.shape[1]-2*m,T.shape[0]-2*m,1))
plt.figure()
plt.imshow(ylabel[:,:,0], cmap = 'Reds')#, vmin = 0, vmax = 1)
plt.colorbar()

plt.show()
