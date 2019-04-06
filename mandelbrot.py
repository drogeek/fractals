import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import matplotlib.cm as cmx
import matplotlib.colors as colors

limit=20

xdata, ydata = [], []
fig, ax = plt.subplots()
ax.grid()
normalize = colors.Normalize(0,limit)
cm = plt.get_cmap('plasma')
scalarMap = cmx.ScalarMappable(norm=normalize, cmap=cm)

squared_modulus = lambda x : x.real*x.real + x.imag*x.imag
def data_gen():
    while True:
        u=np.random.uniform(*ax.get_xlim())
        v=np.random.uniform(*ax.get_ylim())
        z0=complex(0)
        for k in range(limit):
            if squared_modulus(z0) <= 4:
                z0=z0**2+complex(u,v)
            else:
                break
        if k == limit-1:
            color = -1
        else:
            color = k
        yield u,v,color


def init():
    ax.set_ylim(-2,2)
    ax.set_xlim(-2,2)

def run(data):
    u,v,c = data
    if c==-1:
        result=ax.scatter(u,v,s=1,c='black')
        result2=ax.scatter(u,-v,s=1,c='black')
    else:
        result=ax.scatter(u,v,s=1,c=scalarMap.to_rgba(c))
        result2=ax.scatter(u,-v,s=1,c=scalarMap.to_rgba(c))
    return result,result2

import matplotlib.animation as animation
ani = animation.FuncAnimation(fig, run, data_gen, blit=False, interval = 1, repeat = False, init_func=init)
plt.show()
