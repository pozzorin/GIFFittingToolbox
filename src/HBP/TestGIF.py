import sys
sys.path.append('../')
sys.path.append('/Users/christianpozzorini/Documents/workspace/GIFToolbox/src/')

import matplotlib.pylab as plb
import numpy as np

from GIF import *
from GIF_HT import *
from Filter_Rect_LogSpaced import *
from Filter_Exps import *

seed = 1
dt = 0.1

T = 100.0
I0_max = 0.5
I0_all = np.random.rand(50)*I0_max

I = []

for I0 in I0_all :
    I.append(I0*np.ones(int(T/dt)))


I = np.concatenate(I)

myGIF = GIF_HT(dt, 1, 1)

myGIF.DV = 0.5

myGIF.eta = Filter_Exps()
myGIF.eta.setFilter_Timescales([10.0, 50.0, 250.0])
myGIF.eta.setFilter_Coefficients([0.2, 0.05, 0.025])

myGIF.gamma = Filter_Exps()
myGIF.gamma.setFilter_Timescales([5.0, 200.0, 250.0])
myGIF.gamma.setFilter_Coefficients([15.0, 3.0, 1.0])

V0 = -70.0


(time, V, eta_sum, V_T, spks) = myGIF.simulate_seed(I, V0, seed)


plb.figure(figsize=(16,8))
plt.subplot(2,1,1)
plt.plot(time, I, 'gray')
plt.plot(time, eta_sum, 'red')
plt.subplot(2,1,2)
plt.plot(time, V, 'black')
plt.plot(time, V_T, 'red')
plt.show()

dic =  myGIF.getResultDictionary()

print dic['model']

np.savetxt('/Users/christianpozzorini/Desktop/GIFtest_I.txt', I, delimiter=',')  
np.savetxt('/Users/christianpozzorini/Desktop/GIFtest_V.txt', V, delimiter=',')
np.savetxt('/Users/christianpozzorini/Desktop/GIFtest_s.txt', spks, delimiter=',')