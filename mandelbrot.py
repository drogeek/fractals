import numpy as np
import matplotlib.pyplot as plt
from itertools import product

N=10**4
limit=10**2

real_part=np.linspace(-2,2,N)
imag_part=np.linspace(0,2,N/2)
result = []
fig,ax = plt.subplots()
line, = ax.plot([],[])
ax.grid()
xdata, ydata = [], []

def data_gen(t=0):
    for u,v in product(real_part,imag_part):
        z0=complex(t)
        for k in range(limit):
            if abs(z0) <= 2:
                z0=z0**2+complex(u,v)
            else:
                break
        if k == limit-1:
            yield u,v

def init():
    ax.set_ylim(-0.1,0.1)
    ax.set_xlim(-2,-1.8)
    del xdata[:]
    del ydata[:]
    line.set_data(xdata,ydata)
    return line,

def run(data):
    u,v = data
    xdata.append(u)
    xdata.append(u)
    ydata.append(v)
    ydata.append(-v)
    line.set_data(xdata,ydata)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    if u >= xmax:
        if xmax < 0:
            ax.set_xlim(xmin,xmax/1.5)
        else:
            ax.set_xlim(xmin,xmax*1.5)
        ax.figure.canvas.draw()
    if v >= ymax:
        ax.set_ylim(ymin-0.1,ymax+0.1)
        ax.figure.canvas.draw()
    line.set_data(xdata,ydata)
    return line,

import matplotlib.animation as animation
ani = animation.FuncAnimation(fig, run, data_gen, blit=False, interval = 1, repeat = False, init_func=init)
plt.show()
