# generate random walks

import numpy as np
import matplotlib.pyplot as plt

def datagen(num_points, epsilon, p_0=1):

    x = np.zeros(num_points)
    x[0] = p_0
    for i in range(num_points-1):
        x[i+1] = x[i] * (1+ np.random.randn() * epsilon)

    return x




#fig, ax = plt.subplots()
#
#
#for i in range(5):
#    synth = datagen(80000, 0.0025, 1)
#    ax.plot(np.arange(np.size(synth)), synth)
#
#plt.show()
