import numpy as np
import matplotlib.pyplot as plt
from itertools import product

N=10**3
limit=10**3

real_part=np.linspace(-2,2,N)
imag_part=np.linspace(0,2,N/2)
result = []
for u,v in product(real_part,imag_part):
    z0=complex(0)
    for k in range(limit):
        if abs(z0) <= 2:
            z0=z0**2+complex(u,v)
        else:
            break
    if k == limit-1:
        result.append((u,v))
        result.append((u,-v))

print('preparing to plot')
plt.plot(*zip(*result),',')
plt.show()
