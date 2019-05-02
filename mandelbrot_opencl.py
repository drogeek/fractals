import pyopencl as cl
from pyopencl import array
import numpy as np
import matplotlib.pyplot as plt
import itertools as it
import time

start = time.time()
screen_w = 10000
screen_h = int(np.round(screen_w*9/16))
limit = 30000
platform = cl.get_platforms()[0]

device = platform.get_devices()[0]

context = cl.Context([device])

program = cl.Program(context, """
        typedef double2 Complex;

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
        __global double3 *result, __global double3 *colormap, const int colormap_size, const int size)
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
colormap=np.array([(*x,1) for x in plt.get_cmap('inferno').colors], dtype=array.cltypes.double3)
img = np.array([(x[1],x[0]) for x in it.product(np.linspace(ymax,ymin,screen_h),np.linspace(xmin,xmax,screen_w))], dtype=array.cltypes.double2)
end = time.time()
mem_flags = cl.mem_flags
print("initialisation time:",end-start)
start = end
img_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=img)
colormap_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=colormap)
half_img = np.zeros(screen_h*screen_w*4,dtype=np.float64)
half_img_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, half_img.nbytes)
end = time.time()
print("copy to gpu time:",end-start)
start = end

event = program.compute_mandelbrot(queue, img.shape, None, np.int32(limit), img_buf, half_img_buf, colormap_buf, np.int32(len(colormap)), np.int32(img.size))
event.wait()
end = time.time()
print("gpu computing time:",end-start)
start = end
cl.enqueue_copy(queue, half_img, half_img_buf)

end = time.time()
print("copy back to cpu time:",end-start)
start = end

cl.enqueue_copy(queue, half_img, half_img_buf)
import sys
#np.set_printoptions(threshold=sys.maxsize)
#print(half_img.reshape(screen_h//2,screen_w,4)[:,:,:-1])
final_img = half_img.reshape(screen_h,screen_w,4)[:,:,:-1]
fig,ax = plt.subplots()
ax.imshow(final_img)
end = time.time()
print("matplotlib imshow time:",end-start)
start = end
dpi=900
ax.grid(False)
ax.set_axis_off()
fig.savefig('mandelbrot_{}x{}_{}dpi'.format(screen_w,screen_h,dpi),dpi=dpi,bbox_inches="tight")

end = time.time()
print("matplotlib save time:",end-start)
start = end
#plt.show()
