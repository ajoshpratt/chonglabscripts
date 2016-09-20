import numpy as np
import networkx as nx
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.cm as cm
from matplotlib.collections import LineCollection

from scipy.spatial.distance import cdist, pdist, squareform

import os
import sys

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import itertools

#WEST_ROOT = os.environ['WEST_ROOT']
WEST_ROOT = '/home/varus/apps/westpa/wexplore/'
for lib in ['lib/wwmgr', 'src', 'lib/west_tools']:
    path = os.path.join(WEST_ROOT, lib)
    if path not in sys.path:
        sys.path.append(path)

sys.path.append('/home/varus/work/P53.IMPLICIT/wexplore')
sys.path.append('/home/varus/work/P53.IMPLICIT/03/')

# h5py storage types
vstr_dtype = h5py.new_vlen(str)
idtype = np.dtype([('iter_name', vstr_dtype), ('string_index', np.int32)])

fig = plt.figure(figsize=(10,40))
ax = fig.add_subplot(111)
import h5py
from matplotlib.image import NonUniformImage
import matplotlib
ax.set_xlim(0,10)
ax.set_ylim(0,40)
ax.set_ylabel("Binding RMSD", weight="bold")
ax.set_xlabel("Minimum Distance", weight="bold")
font = {'family' : 'normal',
        'weight' : 'regular',
        'size'   : 20}
matplotlib.rc('font', **font) 

vmin=None
vmax=None
norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

# Let's put a point down on successful bins (that is, bins which lead immediately to flux events)
west = h5py.File('west.h5')
bins = []
pcoords_x = []
pcoords_y = []
weights = []
histo_x = []
histo_y = []
all_weights = []
for iiter,iter in west['iterations'].iteritems():
    #print((iter['pcoord'][:,:,0][...]).shape)
    histo_x.append(iter['pcoord'][:,:,0][...])
    histo_y.append(iter['pcoord'][:,:,1][...])
    all_weights.append(np.repeat(iter['seg_index']['weight'][:][...], 20))


histo_x = np.array(histo_x).flatten()
histo_y = np.array(histo_y).flatten()
all_weights = np.array(all_weights).flatten()
all_weights /= np.nansum(all_weights)
all_x = histo_x
all_y = histo_y

x = np.linspace(0,10,100)
y = np.linspace(0,40,400)
xg, yg = np.meshgrid(x,y)
xyg = np.vstack([yg.ravel(), xg.ravel()]).T

xcen = (x[1:] - x[:-1])/2
ycen = (y[1:] - y[:-1])/2

print(x.dtype)
print(y.dtype)
print(y.shape, x.shape)
print(histo_x.shape)
print(histo_y.shape)
print(xcen.dtype)
print(ycen.dtype)
print(xg.dtype)
print(yg.dtype)
print(all_weights.dtype)
print(xyg.shape)
def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    import math
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)  # Fast and numerically precise
    return (average, math.sqrt(variance))

hx = weighted_avg_and_std(histo_x, all_weights)[1]*3*3.5*west['summary'].shape[0]**(-1/3)*2
hy = weighted_avg_and_std(histo_y, all_weights)[1]*3*3.5*west['summary'].shape[0]**(-1/3)*2
#hx = 2
#hy = 2

# Here, we're gonna write some CUDA code.

# Convert the arrays...

all_weights = all_weights.astype(np.float32)
x = x.astype(np.float32)
y = y.astype(np.float32)
histo_x = histo_x.astype(np.float32)
histo_y = histo_y.astype(np.float32)
#hx = np.float32([4*hx**2])
#hy = np.float32([4*hy**2])
hx = np.float32([hx])
hy = np.float32([hy])
m = np.float32([histo_x.shape[0]])
# Now, allocate the memory on the device.

pdf = np.zeros((int(x.shape[0]), int(y.shape[0]), (m/(256*1))), dtype=np.float32)
gpu_weights = cuda.mem_alloc(all_weights.nbytes)
gpu_x = cuda.mem_alloc(x.nbytes)
gpu_y = cuda.mem_alloc(y.nbytes)
gpu_hx = cuda.mem_alloc(hx.nbytes)
gpu_hy = cuda.mem_alloc(hy.nbytes)
gpu_histo_x = cuda.mem_alloc(histo_x.nbytes)
gpu_histo_y = cuda.mem_alloc(histo_y.nbytes)
p = np.int32([2])
gpu_p = cuda.mem_alloc(p.nbytes)
gpu_pdf = cuda.mem_alloc(pdf.nbytes)
gpu_m = cuda.mem_alloc(m.nbytes)

pdf_return = np.empty_like(pdf)

# Now, transfer the data to the GPU

for gpui, i in itertools.izip((gpu_pdf, gpu_weights, gpu_x, gpu_y, gpu_histo_x, gpu_histo_y, gpu_p, gpu_hx, gpu_hy, gpu_m), (pdf, all_weights, x, y, histo_x, histo_y, p, hx, hy, m)):
    cuda.memcpy_htod(gpui, i)

# Now, write the kernel and the operations.

kc_temp = """
#include <math.h>

__global__ void kde(float *pdf,
                    const float *x,
                    const float *y,
                    const float *xi,
                    const float *yi,
                    const float *w,
                    const float *hx,
                    const float *hy,
                    const int m)
{
    // Points we're evaluating.
    signed int i = blockIdx.x;
    signed int j = blockIdx.y;
    signed int l = blockIdx.z;
    // Thread inside of the block (local block id).
    signed int thr = threadIdx.x * blockDim.y + threadIdx.y;
    // Overall datapoint in weight vector (that is, which gaussian are we evaluating?)
    // Also, the blocksize, and the flattened array point that we'll return to the calling program.
    signed int blocksize = blockDim.x * blockDim.y;
    signed int k = thr + blockIdx.z * blocksize;
    signed int flat_sum = (i * gridDim.y * gridDim.z) + (j * gridDim.z) + l;
    // Hack for now.
    __shared__ float smem[256*1];
    for ( signed int b = thr; b < (blocksize); b += blocksize) smem[b] = 0;
    __syncthreads();

    if ( ( i < gridDim.x ) && ( j < gridDim.y ) && ( k < (gridDim.z * blocksize) )) {
            // Add each item to the appropriate spot in the local memory storage.
            // Each block has access to the same 'smem'.
            // Also, use the fast functions.  Afterwards, wait for every thread to finish, them sum and adjust the global matrix.
            // Man, this is slow as fucking molasses.
            // Given the blah of single point, it doesn't seem like fast math is a great idea.  But maybe just not the exponentiation?
            //atomicAdd(&smem[thr], expf(logf(w[k]) - ((((x[i]-xi[k])*(x[i]-xi[k]))/(*hx) + (((y[j]-yi[k])*(y[j]-yi[k]))/(*hy))))));
            //smem[thr] = expf(logf(w[k]) - ((((x[i]-xi[k])*(x[i]-xi[k]))/(*hx) + (((y[j]-yi[k])*(y[j]-yi[k]))/(*hy)))));
            smem[thr] = expf(logf(w[k]) - ((((x[i]-xi[k])*(x[i]-xi[k]))/(*hx) + (((y[j]-yi[k])*(y[j]-yi[k]))/(*hy)))));
            __syncthreads();
            if ( thr == 0 ) {
                for ( signed int z = 0; z < blocksize; z += 1 ) {
                    pdf[flat_sum] += smem[z];
                }
            }
    }
}

"""

kernel_code = kc_temp % {
        'BLOCKSIZE': 16*16
        }
bdim = (256, 1, 1)
blocksize=bdim[0]*bdim[1]
dx, mx = divmod(x, bdim[0])
dy, my = divmod(y, bdim[1])
print(m)
# That's m, right there.
# AH!  The final dimension of gdim SHOULD be the number of calculations you have...
gdim = (100, 400, int(m[0]/blocksize))
mod = SourceModule(kernel_code)
func = mod.get_function("kde")
func(gpu_pdf, gpu_x, gpu_y, gpu_histo_x, gpu_histo_y, gpu_weights, gpu_hx, gpu_hy, gpu_m, block=bdim, grid=gdim)
cuda.memcpy_dtoh(pdf, gpu_pdf)
print(pdf)
pdf = pdf[:,:,:].sum(axis=2)

# That should work more or less for the actual data in and out.  Just write the kernel code and call it a day.


pdf /= np.nansum(pdf)
pdf = -np.log(pdf)
pdf -= np.nanmin(pdf)

clines = np.arange(50)

CS = ax.contour(x, y, pdf.T, clines, vmin=None, origin='lower')
CS2 = ax.contourf(x, y, pdf.T, clines, vmin=None, origin='lower', alpha=0.3)


from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax)

cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(CS2, cax=cax)
label = r'$\Delta F(x)\,/\,kT$' +'\n' + r'$\left[-\ln\,P(x)\right]$'
cbar.set_label(label)

plt.savefig('wexplore.cuda.pdf')

