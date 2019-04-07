import numpy as np
import matplotlib.pyplot as plt
import itertools as it
import matplotlib.cm as cmx
import matplotlib.colors as colors
import multiprocessing as mp

limit=150
N=5000
dpi=800

fig, ax = plt.subplots()
ax.grid(False)
ax.set_axis_off()
normalize = colors.Normalize(0,limit)
cm = plt.get_cmap('plasma')
scalarMap = cmx.ScalarMappable(norm=normalize, cmap=cm)

squared_modulus = lambda x : x.real*x.real + x.imag*x.imag

def compute_range(start,end,index,pieces_nb):
    return (start+index*(end-start)/pieces_nb,start+(index+1)*(end-start)/pieces_nb)

def data_gen(queue,index,total):
    startx, endx = compute_range(-1.6,1.5,index,total)
    print("process {} computing from {} to {}".format(index,startx,endx))
    result = np.zeros((N//2,N//total,3))
    count = 0
    for u,v in it.product(np.linspace(startx,endx,N//total),np.linspace(0,1.6,N//2)):
        if count%(N//total*N//2//30) == 0:
            print("process {}: {:.2f}%".format(index,100*count/(N//total*N//2)))
        z0=complex(0)
        for k in range(limit):
            if squared_modulus(z0) <= 4:
                z0=z0**2+complex(u,v)
            else:
                break
        if k == limit-1:
            result[count%(N//2)][count//(N//2)] = [0,0,0]
        else:
            result[count%(N//2)][count//(N//2)] = scalarMap.to_rgba(k)[:-1]
        count+=1
    queue.put((index,result))
    print("process {} finished".format(index))

m = mp.Manager()
queue=m.Queue()
pool=mp.Pool()
pool.starmap(data_gen, zip(it.repeat(queue),range(0,mp.cpu_count()),it.repeat(mp.cpu_count())))
pool.close()
tmp = []
while not queue.empty():
    tmp.append(queue.get())
del queue
tmp=sorted(tmp)
half_final_image = tmp[0][1]
for img_piece in tmp[1:]:
    _, img_piece = img_piece
    half_final_image = np.concatenate((half_final_image,img_piece),1)
del tmp
final_image = np.concatenate((np.flip(half_final_image,0),half_final_image),0)
del half_final_image
plt.imshow(final_image)
fig.savefig('mandelbrot_{}dpi_{}matrix'.format(dpi,N),dpi=dpi)
#plt.show()
