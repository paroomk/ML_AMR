#Script to identify cells based on error threshold + ML training to identify cells ith error above threshold

import numpy as np
import matplotlib.pyplot as plt
import yt
import tensorflow as tf
from tensorflow import keras

def extract_frm_pltfile(path1,path2, level, train, file):
    # Get data all projected to a uniform grid at the coarsest level
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
    
    T = np.array(data0['Temp'])
    T1 = np.array(data1['Temp'])

    print(np.mean(T),np.mean(T1))
    
    #print(diff[:,:,47])
    
    #plt.figure()
    #plt.imshow(diff[:,:,50])
    #plt.show()
    
    #Create appropriate training data (3x3 grid of Temp values centered around the point of interest)
    nvar =  15

    Tmean = np.mean(T)

    T = (T/Tmean)#/np.std(T) 
    T1 = (T1/Tmean)#/np.std(T) 

    u = np.array(data0['x_velocity'])
    u1 = np.array(data1['x_velocity'])

    umean = np.mean(u)

    v = np.array(data0['y_velocity'])
    w = np.array(data0['z_velocity'])
    e = np.array(data0['eint_e'])
    rho = np.array(data0['density'])
    pr = np.array(data0['pressure'])

    u1 = (u1/np.mean(u))#/np.std(u)
    u = (u/np.mean(u))#/np.std(u) 
    v = (v/np.mean(v))#/np.std(v) 
    w = (w/np.mean(w))#/np.std(w) 
    e = (e/np.mean(e))#/np.std(e) 
    rho = (rho/np.mean(rho))#/np.std(rho) 
    pr = (pr/np.mean(pr))#/np.std(pr) 
    #
    diff = np.abs(u-u1)
    label = diff 

    #Comment these lines for regression
    
    label[diff<1.e-2]  = 0
    label[diff>=1.e-2] = 1

    print('Max error =',np.max(np.abs(label)), 'Min error =', np.min(np.abs(label)))

    Tx, Ty, Tz = np.gradient(data0['Temp'])
    ux, uy, uz = np.gradient(data0['x_velocity'])
    vx, vy, vz = np.gradient(data0['y_velocity'])
    wx, wy, wz = np.gradient(data0['z_velocity'])
    
    #T = np.stack((T,u,v,w,e,rho,pr),axis=-1)
    T = np.stack((Tx,Ty,Tz,ux,uy,uz,vx,vy,vz,wx,wy,wz,e,rho,pr),axis=-1)

    #T = np.reshape(T,(T.shape[0], T.shape[1], T.shape[2], 1))
    
    l=5 #size of box
    m = l//2
    
    Ti = T[m:-m,m:-m,m:-m,:] #Exclude boundary points
    print(Ti.shape)

    if (train):
       ds_index = extract_frm_downsampledfile(file)
       ds_index = ds_index[ds_index<(Ti.size//nvar)]
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
    
    print('Max error =',np.max(np.abs(xlabel)), 'Min error =', np.min(np.abs(xlabel)))
    
    return x, xlabel, nvar, l, T
def extract_frm_downsampledfile(file):
    ds = np.load(file)
    print(ds.files)
    ds_index = ds['indices']
    #print(ds_index)
    return ds_index

#####################################################################################
path = '/projects/hpacf/pmadathi/jetcase/314_ambient/'
path1 = path + 'plt0_85101' #'plt0_75346'
path2 = path + 'plt2_85101' #'plt1_75346'


level=0
train = True
file = '/home/pmadathi/PhaseSpaceSampling/downSampledData_01/downSampledData_10000.npz'
x, xlabel, nvar, l, T = extract_frm_pltfile(path1, path2, level, train, file)
y = x
ylabel = xlabel

path = '/projects/hpacf/pmadathi/jetcase/314_ambient/'
path1 = path + 'plt1_85101' #'plt0_75346'
path2 = path + 'plt2_85101' #'plt1_75346'

#level=1
#x1, xlabel1, nvar, l, T1 = extract_frm_pltfile(path1, path2, level)
#print('Max error =',np.max(np.abs(xlabel)), 'Min error =', np.min(np.abs(xlabel)))
#y1 = x1
#ylabel1 = xlabel1
#####################################################################################
#Downsample
#####################################################################################

sf = 1.e+0
xlabel = xlabel*sf
ylabel = ylabel*sf
print('Max error (01) =',np.max(np.abs(xlabel)), 'Min error =', np.min(np.abs(xlabel)))

#file = '/home/pmadathi/PhaseSpaceSampling/downSampledData_12/downSampledData_10000.npz'
#ds_index = extract_frm_downsampledfile(file)
#ds_index = ds_index[ds_index<x.shape[0]]
#x1= x1[ds_index,:,:,:]
#xlabel1 = xlabel1[ds_index]*sf
#ylabel1 = ylabel1*sf

#print('Max error (12) =',np.max(np.abs(xlabel1)), 'Min error =', np.min(np.abs(xlabel1)))

xx = x #np.concatenate((x,x1),axis=0)
xxlabel = xlabel #np.concatenate((xlabel,xlabel1),axis=0)


##############################################################################
#Creating validation data set
############################################################################## 
val_index = np.random.choice(np.arange(0,xx.shape[0]),xx.shape[0]//5,replace='False')

x_val = xx[val_index, :,:,:,:]
x_vallabel = xxlabel[val_index]

x_train = np.delete(xx, val_index, 0)
x_trainlabel = np.delete(xxlabel, val_index, 0)
#############################################################################
#Creating test data set
#############################################################################
test_index = np.random.choice(np.arange(0,x_train.shape[0]),xx.shape[0]//5,replace='False')

x_test = x_train[test_index,:,:,:,:]
x_testlabel = x_trainlabel[test_index]

x_train = np.delete(x_train, test_index, 0)
x_trainlabel = np.delete(x_trainlabel, test_index, 0)

print(x_train.shape,x_trainlabel.shape)
#############################################################################
#Normalization
#############################################################################

train_mean = np.mean(x_train)
train_std = np.std(x_train)

x_train = (x_train - train_mean)/train_std
x_test  = (x_test- train_mean)/train_std
x_val   = (x_val - train_mean)/train_std
y       = (y - train_mean)/train_std
    
l_mean = np.mean(x_trainlabel)
l_std  = np.std(x_trainlabel)

l_min = np.min(x_trainlabel)
l_max  = np.max(x_trainlabel)

print(l_std)

#x_trainlabel = (x_trainlabel)/l_std
#x_testlabel = (x_testlabel)/l_std
#x_vallabel = (x_vallabel)/l_std

print('Max error =',np.max(np.abs(x_trainlabel)), 'Min error =', np.min(np.abs(x_trainlabel)))
#############################################################################
#Model creation
#############################################################################
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(l,l,l,nvar)))
#model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Conv3D(filters=4, kernel_size=(3,3,3), activation='relu'))
#model.add(tf.keras.layers.LeakyReLU(alpha=0.05))
model.add(tf.keras.layers.AveragePooling3D(pool_size=(2,2,2)))
#model.add(tf.keras.layers.Conv3D(filters=8, kernel_size=(3,3,3)))
#model.add(tf.keras.layers.LeakyReLU(alpha=0.05))
#model.add(tf.keras.layers.MaxPooling3D(pool_size=(2,2,2)))
#model.add(tf.keras.layers.Conv3D(filters=64, kernel_size=(3,3,3), activation='relu',input_shape=(l,l,l,nvar)))
#model.add(tf.keras.layers.MaxPooling3D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(8, kernel_regularizer='l1', activation='relu'))
#model.add(tf.keras.layers.LeakyReLU(alpha=0.05))
model.add(tf.keras.layers.Dropout(0.11))
#model.add(tf.keras.layers.Dense(4, activation='relu', kernel_regularizer='l1'))
model.add(tf.keras.layers.Dense(1, kernel_regularizer='l1', activation='sigmoid'))#, activation=tf.keras.activations.softplus))
#model.add(tf.keras.layers.LeakyReLU(alpha=0.05))

model.summary()

opt = keras.optimizers.Adam(learning_rate=0.0001)
#reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
#                              patience=5, min_lr=0.00005)
model.compile(optimizer= opt, loss='mse', metrics=['accuracy'])
###############################################################################################
#Fit on training data
###############################################################################################
history = model.fit(x_train, x_trainlabel, batch_size=128, epochs=200, validation_data=(x_val,x_vallabel))
###############################################################################################
#Test 
###############################################################################################
score = model.evaluate(x_test, x_testlabel)
print('Loss: %.2f' % (score[0]))
print('MSE: %.2f' % (score[1]))
###############################################################################################
#Visualize filters
###############################################################################################
# summarize filter shapes
#for layer in model.layers:
#	# check for convolutional layer
#	if 'conv' not in layer.name:
#		continue
#	# get filter weights
#	filters, biases = layer.get_weights()
#	print(layer.name, filters.shape, layer.output.shape)
#
#f_min, f_max = filters.min(), filters.max()
#filters = (filters - f_min) / (f_max - f_min)
#
#n_filters, ix = 4, 1
#for i in range(n_filters):
#	# get the filter
#	f = filters[:, :, :, :, i]
#	# plot each channel separately
#	for j in range(3):
#		# specify subplot and turn of axis
#		ax = plt.subplot(n_filters, 3, ix)
#		ax.set_xticks([])
#		ax.set_yticks([])
#		# plot filter channel in grayscale
#		plt.imshow(f[:, :, j, 1], cmap='gray')
#		ix += 1
#
#plt.show()
#################################################################################################
#Visualize feautures
#################################################################################################
# redefine model to output right after the first hidden layer
#submodel = tf.keras.Model(inputs=model.inputs, outputs=model.layers[1].output)
#feature_maps = submodel.predict(x_test[:1,:,:,:,:])

#plt.imshow(x_test[0,:,:,0,0], cmap='gray')
#plt.show()

#square = 2
#ix = 1
#for _ in range(square):
#	for _ in range(square):
#		# specify subplot and turn of axis
#		ax = plt.subplot(square, square, ix)
#		ax.set_xticks([])
#		ax.set_yticks([])
#		# plot filter channel in grayscale
#		plt.imshow(feature_maps[0, :, :, 0, ix-1], cmap='gray')
#		ix += 1
## show the figure
#plt.show()

#exit()

#y_predict = model.predict(y)
#print(y_predict.shape, ylabel.shape)
#err = np.abs(y_predict[:,0]-ylabel)
#print(np.max(err))
#
#ylabel = np.abs(ylabel)
#y_predict = np.abs(y_predict)

###########################################################################
#Test on a different case
###########################################################################

path = '/projects/hpacf/pmadathi/jetcase/350_ambient/'
path1 = path + 'plt0_75346'
path2 = path + 'plt2_75346'

x, xlabel, nvar, l, T = extract_frm_pltfile(path1, path2, 0, False, file)
#xlabel = xlabel 
print('Max error =',np.max(np.abs(xlabel)), 'Min error =', np.min(np.abs(xlabel)))
y = x
xlabel = xlabel*sf
ylabel = xlabel
print('Max error =',np.max(np.abs(xlabel)), 'Min error =', np.min(np.abs(xlabel)))

y   = (y - train_mean)/train_std
#ylabel = (ylabel)/l_std

y_predict = model.predict(y)
print(y_predict.shape, ylabel.shape)
ylabel = ylabel/sf
y_predict = y_predict/sf
err = np.abs(y_predict[:,0]-ylabel)
print(np.max(err))

score = model.evaluate(y, ylabel)
print('Loss: %.2f' % (score[0]))
print('MSE: %.2f' % (score[1]))
#exit()
##########################################################################
#Plotting
###########################################################################
#print(err.shape, ylabel.shape)
m =l//2
err = np.reshape(err, (T.shape[1]-2*m,T.shape[0]-2*m,1))
ax =plt.gca()
ax.set_yscale('log')
ax.set_xscale('log')
ax.scatter(ylabel, y_predict)
plt.xlabel('Actual error')
plt.ylabel('Predicted error')
plt.title('Case: 314 ambient')
ax.set_xlim([min(ylabel),max(ylabel)])
ax.set_ylim([min(ylabel),max(ylabel)])
plt.show()

plt.figure()
plt.imshow(err[:,:,0], cmap = 'Reds') #, vmin = -1, vmax = 1)
plt.colorbar()
plt.show()

y_predict = np.reshape(y_predict, (T.shape[1]-2*m,T.shape[0]-2*m,1))

plt.figure()
plt.imshow(y_predict[:,:,0], cmap ='Reds') #, vmin = 0, vmax = 1)
plt.colorbar()
plt.show()

ylabel = np.reshape(ylabel, (T.shape[1]-2*m,T.shape[0]-2*m,1))
plt.imshow(ylabel[:,:,0], cmap = 'Reds') #, vmin = 0, vmax = 1)
plt.colorbar()

plt.show()

#print(np.max(xlabel-pred[:,0]))
