#Script to find difference in base grid values

import numpy as np
import matplotlib.pyplot as plt
import yt

amrex_plt_file0 = '../practice-repo/PeleC/PeleC/Exec/RegTests/PMF/plt00020'
ds0 = yt.load(amrex_plt_file0)

amrex_plt_file1 = '../practice-repo/PeleC/PeleC/Exec/RegTests/PMF/plt00020'
ds1 = yt.load(amrex_plt_file1)

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

plt.figure()
plt.imshow(diff[0,:,:])
plt.show()

