# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 15:29:40 2016

@author: Ziwen
"""
import numpy as np
import math
import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))

r = np.linspace(0, 5, 100)
d = [1,2,5,10,20]
rstar = [x**0.5 for x in d]
i = 0

for x in d:
    
    S = 2 * (math.pi**((x)/2 )) / math.gamma((x)/2)
    y = (S * r**(x-1) / ((2 * math.pi)**(x/2))) * (math.e**(-(r**2)/2))
    plt.subplot(331+i)
    plt.plot(r,y)
    plt.title("d = %d" % x)
    plt.xlabel("r")
    i+=1

plt.subplot(336)

i = 0
y = [0, 0, 0, 0, 0]
for x in d:
    S = 2 * (math.pi**((x)/2 )) / math.gamma((x)/2)
    y[i] = (S * rstar[i]**(x-1) / ((2 * math.pi)**(x/2))) * (math.e**(-(rstar[i]**2)/2))
    plt.text(rstar[i],y[i],'%.2f, %.2f' % (rstar[i], y[i]),horizontalalignment ="right")
    plt.xlabel("r *")    
    i += 1

plt.plot(rstar,y,'o')
plt.xlim(0,5)
plt.ylim(0.4, 0.60)
plt.show()