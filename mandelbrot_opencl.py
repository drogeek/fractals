import pyopencl as cl
from pyopencl import array
import numpy as np
import matplotlib.pyplot as plt
import itertools as it

screen_w = 5000
screen_h = 5000 
limit = 10000
platform = cl.get_platforms()[0]

device = platform.get_devices()[0]

context = cl.Context([device])

program = cl.Program(context, """
        typedef float2 Complex;

        inline float squared_mod(Complex c){
            return dot(c,c);
        }

        inline Complex mult(Complex a, Complex b){
            Complex result;
            result.x = a.x*b.x - a.y*b.y;
            result.y = a.y*b.x + a.x*b.y;
            return result;
        }

        __kernel void compute_mandelbrot(const int limit, __global const Complex *matrix,
        __global float3 *result, __global float3 *colormap, const int colormap_size, const int size)
        {
          const int local_limit = limit;
          const int local_size = size;
          int i;
          const long gid = get_global_id(0);
          Complex x = matrix[gid];
          Complex z = (Complex)(0,0);

          for(i = 0; i<local_limit ; i++){
            if(squared_mod(z) <= 4){
              z=mult(z,z)+x;
            }
            else{
                break;
            }
          }
          if(i == local_limit){
            result[gid] = (0,0,0);
          }
          else{
            result[gid] = colormap[(i*colormap_size)/local_limit];
          }

        }
        """).build()

queue = cl.CommandQueue(context)

xmin,xmax = 0.27910,0.27945
ymin,ymax = 0.01005, 0.01040
colormap=np.array([(*x,1) for x in plt.get_cmap('inferno').colors], dtype=array.cltypes.float4)
img = np.array([(x[1],x[0]) for x in it.product(np.linspace(ymax,ymin,screen_h),np.linspace(xmin,xmax,screen_w))], dtype=array.cltypes.float2)
print(img.shape)
mem_flags = cl.mem_flags
img_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=img)
colormap_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=colormap)
half_img = np.zeros(screen_h*screen_w*4,dtype=np.float32)
half_img_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, half_img.nbytes)

event = program.compute_mandelbrot(queue, img.shape, None, np.int32(limit), img_buf, half_img_buf, colormap_buf, np.int32(len(colormap)), np.int32(img.size))
event.wait()
cl.enqueue_copy(queue, half_img, half_img_buf)

import sys
#np.set_printoptions(threshold=sys.maxsize)
#print(half_img.reshape(screen_h//2,screen_w,4)[:,:,:-1])
final_img = half_img.reshape(screen_h,screen_w,4)[:,:,:-1]
fig,ax = plt.subplots()
ax.imshow(final_img)
dpi=600
ax.grid(False)
ax.set_axis_off()
fig.savefig('mandelbrot_{}x{}_{}dpi'.format(screen_w,screen_h,dpi),dpi=dpi,bbox_inches="tight")
#plt.show()
