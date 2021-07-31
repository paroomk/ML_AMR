import numpy as np
import yt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class testModel(tf.keras.Model):
  def __init__(self):
    super(testModel, self).__init__()
    self.dense1 = tf.keras.layers.Dense(32, input_dim=nvar*3**3,kernel_initializer='Ones', activation=tf.nn.relu, kernel_regularizer='l1')
    self.bn1    = tf.keras.layers.BatchNormalization()
    self.dense2 = tf.keras.layers.Dense(64, kernel_initializer='Ones', activation=tf.nn.relu, kernel_regularizer='l1')
    self.bn2    = tf.keras.layers.BatchNormalization()
    self.drop   = tf.keras.layers.Dropout(0.4)
    self.dense3 = tf.keras.layers.Dense(1, kernel_initializer='Ones', activation=tf.nn.relu, kernel_regularizer='l1')
  def call(self, inputs):
    x = self.dense1(inputs)
    x = self.bn1(x)
    x = self.dense2(x)
    x = self.bn2(x)
    x = self.drop(x)
    return self.dense3(x)

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

    #merge here
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
        x[:,:,:,i] = (x[:,:,:,i]-x_mean[i])/x_std[i] 
        x1[:,:,:,i] = (x1[:,:,:,i]-x1_mean[i])/x1_std[i] 

    label = np.abs(x1-x)

    nvar =  15

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

    X = np.stack((Tx,Ty,Tz,ux,uy,uz,vx,vy,vz,wx,wy,wz,rho,e,pr),axis=-1)
    #X = np.stack((T,u,v,w,rho,e,pr), axis = -1)
    #T = np.reshape(T,(T.shape[0], T.shape[1], T.shape[2], 1))

    l=3 #size of box
    m = l//2

    if m==0:
        Xi = X
    else:
        Xi = X[m:-m,m:-m,m:-m,:] #Exclude boundary points

    print('size of interior:',Xi.shape)

    s = (Xi.size//nvar,l,l,l,nvar)

    x = np.zeros(s)
    print(x.shape)
    xlabel = np.zeros(Xi.size//nvar)

    for p in range(0,Xi.size//nvar):
        i = p%Xi.shape[0]
        j = (p%(Xi.shape[0]*Xi.shape[1]))//Xi.shape[1]
        k = p//(Xi.shape[0]*Xi.shape[1])
        #print(p)
        #print(i,j,k)
        for m in range(0, nvar):
            x[p,:,:,:,m] = X[i:i+l,j:j+l,k:k+l,m]
            #x[p,:,:,:,m] = T[i+l//2,j+l//2,k+l//2,m]
        xlabel[p]  = label[i+l//2,j+l//2,k+l//2,0] #change last index according to variable to be predicted

    #Add Data augmentation here

    #xlabel = (xlabel-np.mean(xlabel))/np.std(xlabel)
    print('Max error =',np.max(np.abs(xlabel)), 'Min error =', np.min(np.abs(xlabel)))

    dim = Xi[:,:,:,1].shape

    x = x.reshape((x.shape[0],nvar*l**3))
    xlabel = xlabel.reshape((xlabel.shape[0],1))

    #x = np.append(x,xlabel,axis=1)

    return x, xlabel, nvar, l, dim

def extract_frm_downsampledfile(file):
    ds = np.load(file)
    print(ds.files)
    ds_index = ds['indices']
    #print(ds_index)
    return ds_index

path = '/projects/hpacf/pmadathi/jetcase/314_ambient/'
path1 = path + 'plt0_85101' #'plt0_75346'
path2 = path + 'plt2_85101' #'plt1_75346'
x, x1, nvar = extract_frm_pltfile(path1, path2)

x_mean = np.zeros(nvar)
x_std  = np.zeros(nvar)

x1_mean = np.zeros(nvar)
x1_std  = np.zeros(nvar)

for i in range(0, nvar): 
    x_mean[i] = np.mean(x[:,:,:,i])
    x_std[i]  = np.std(x[:,:,:,i])

    x1_mean[i] = np.mean(x1[:,:,:,i])
    x1_std[i]  = np.std(x1[:,:,:,i])

x, xlabel, nvar, l, dim = process_data(x, x_mean, x_std, x1, x1_mean, x1_std)
y = x
ylabel = xlabel
##############################################################################
#Downsampled data
##############################################################################
file = '/home/pmadathi/PhaseSpaceSampling/downSampledData_01/downSampledData_100000.npz'
ds_index = extract_frm_downsampledfile(file)
x = x[ds_index,:]
sf = 1.e+3
xlabel = xlabel[ds_index]*sf
ylabel = ylabel*sf
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
test_index = np.random.choice(np.arange(0,x.shape[0]),x.shape[0]//5,replace='False')
x_test = x[test_index,:]
x_testlabel = xlabel[test_index]
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
    y[:,i]       = (y[:,i] - train_mean[i])/train_std[i]

l_mean = np.mean(x_trainlabel)
l_std  = np.std(x_trainlabel)
print(l_std)
x_trainlabel = (x_trainlabel - l_mean)/l_std
x_testlabel = (x_testlabel - l_mean)/l_std
x_vallabel = (x_vallabel - l_mean)/l_std
ylabel = (ylabel - l_mean)/l_std
print('Training data: Max error =',np.max(np.abs(x_trainlabel)), 'Min error =', np.min(np.abs(x_trainlabel)))
print('Max error =',np.max(np.abs(ylabel)), 'Min error =', np.min(np.abs(ylabel)))
print(x_train.shape)
#exit()
model = testModel()
model.compile(
          loss      = tf.keras.losses.MeanSquaredError(),
          metrics   = [tf.keras.metrics.MeanAbsoluteError()],
          optimizer = tf.keras.optimizers.Adam())
# fit 
model.fit(x_train, x_trainlabel, batch_size=128, epochs=1)

#input_data = np.asarray([[10]])
#module = testModel()
#module._set_inputs(input_data)
#print(module(input_data))

# Export the model to a SavedModel
model.save('fcnn', save_format='tf')
