import numpy as np
import matplotlib.pyplot as plt
import itertools as it
import matplotlib.cm as cmx
import multiprocessing as mp

limit=500
N=10**4
assert(mp.cpu_count()<=mp.cpu_count())
assert(N <= (2**32-1)/(2*mp.cpu_count()))

fig, ax = plt.subplots()
ax.grid()
A = mp.Array('d',2*N*mp.cpu_count(),lock=False)

squared_modulus = lambda x : x.real*x.real + x.imag*x.imag
def sub_data_gen(index):
    np.random.seed()
    for i in range(N):
        u=np.random.uniform(*ax.get_xlim())
        v=np.random.uniform(*ax.get_ylim())
        z0=complex(0)
        for k in range(limit):
            if squared_modulus(z0) <= 4:
                z0=z0**2+complex(u,v)
            else:
                break
        if k == limit-1:
            A[2*N*index+2*i]=u
            A[2*N*index+2*i+1]=v

            

def data_gen():
    while True:
        pool=mp.Pool()
        pool.map(sub_data_gen, range(0,mp.cpu_count()))
        pool.close()
        pool.join()
        yield zip(*filter(lambda x : x[0] != 0 and x[1] != 0,zip(*[iter(A)]*2)))

def init():
    ax.set_ylim(-2,2)
    ax.set_xlim(-2,2)

def run(data):
    u,v = data
    return ax.scatter(u,v,marker='.',s=.1,c='black'), ax.scatter(u,[-x for x in v],marker='.',s=.1,c='black')

import matplotlib.animation as animation
ani = animation.FuncAnimation(fig, run, data_gen, blit=False, interval = 1, repeat = False, init_func=init)
plt.show()
