#Script to find difference in base grid values

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import yt
matplotlib.use('Qt5Agg')
import os
os.putenv('DISPLAY', ':0.0')


amrex_plt_file0 = '../PeleC/PeleC/Exec/RegTests/PMF/plt0_00020'
ds0 = yt.load(amrex_plt_file0)
ds0.print_stats()

amrex_plt_file1 = '../PeleC/PeleC/Exec/RegTests/PMF/plt1_00020'
ds1 = yt.load(amrex_plt_file1)
ds1.print_stats()

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
label[diff<1.e-10]  = 0
label[diff>=1.e-10] = 1

print(label)

#print(diff[:,:,47])

#plt.figure()
#plt.imshow(diff[:,:,50])
#plt.show()

#Create appropriate training data (3x3 grid of Temp values centered around the point of interest)

T = np.array(data0['Temp'])

T = T[1:-1,1:-1,1:-1] #Exclude boundary points

print(T.shape)

s = (T.size,3,3)
x = np.zeros(s)

print(x.shape)

#x_test = np.random.choice()
