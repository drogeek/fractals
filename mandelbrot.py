import numpy as np
import matplotlib.pyplot as plt
from itertools import product

N=10**4
limit=10**2

xdata, ydata = [], []
fig, ax = plt.subplots()
ax.grid()
ln, = plt.plot([],[],',')

def data_gen():
    while True:
        u=np.random.uniform(*ax.get_xlim())
        v=np.random.uniform(*ax.get_ylim())
        z0=complex(0)
        for k in range(limit):
            if abs(z0) <= 2:
                z0=z0**2+complex(u,v)
            else:
                break
        if k == limit-1:
            yield u,v

def init():
    ax.set_ylim(-2,2)
    ax.set_xlim(-2,2)
    del xdata[:]
    del ydata[:]
    ln.set_data(xdata,ydata)
    return ln,

def run(data):
    u,v = data
    xdata.append(u)
    ydata.append(v)
    xdata.append(u)
    ydata.append(-v)
    ln.set_data(xdata,ydata)
    return ln,

import matplotlib.animation as animation
ani = animation.FuncAnimation(fig, run, data_gen, blit=False, interval = 1, repeat = False, init_func=init)
plt.show()
