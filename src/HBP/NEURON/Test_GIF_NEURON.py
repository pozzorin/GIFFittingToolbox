"""
Create a GIF model and simulate response to current using a fixed random seed.
The output of this script can be used as a test for other implementations of the GIF model.
"""
import sys
sys.path.append('../')
sys.path.append('../../')

import matplotlib.pylab as plb
import numpy as np

from GIF import *
from GIF_HT import *
from GIF_NEURON import *

from Filter_Rect_LogSpaced import *
from Filter_Exps import *

from SpikeTrainComparator import *

b1 = '#1F78B4' #377EB8
b2 = '#A6CEE3'
g1 = '#33A02C' #4DAF4A
g2 = '#B2DF8A'
r1 = '#E31A1C' #E41A1C
r2 = '#FB9A99'
o1 = '#FF7F00' #FF7F00
o2 = '#FDBF6F'
p1 = '#6A3D9A' #984EA3
p2 = '#CAB2D6'

ye1 = '#FFFF33'
br1 = '#A65628'
br2 = '#D3865B'
pi1 = '#F781BF'
gr1 = '#999999'
k1 = '#000000'
pet1 = '#99D6BA'

# PARAMETERS
####################################################
dt = 0.025    # ms, timestep for num simulations.
seed = 1    # seed used for random number generator

V0 = -70.0  # mV, initial condition for voltage


# CREATE INPUT CURRENT
####################################################

np.random.seed(seed)
T = 100.0
I0_max = 0.5
I0_all = np.random.rand(50)*I0_max

I = []

for I0 in I0_all :
    I.append(I0*np.ones(int(T/dt)))

I = np.concatenate(I)

T = len(I)*dt

# DEFINE NEURON GIF MODEL (EXP BASED KERNELS)
####################################################

hGIF = GIF_NEURON(dt)

hGIF.eta = Filter_Exps()
hGIF.eta.setFilter_Timescales([10.0, 50.0, 250.0])
hGIF.eta.setFilter_Coefficients([0.2, 0.05, 0.025])

hGIF.gamma = Filter_Exps()
hGIF.gamma.setFilter_Timescales([5.0, 200.0, 250.0])
hGIF.gamma.setFilter_Coefficients([15.0, 3.0, 1.0])

# Simulate model response
(htime, hV, heta_sum, hV_T, hspks, hp_dontspike, hurand, hi) = hGIF.simulate_seed(I, V0, seed, passive_axon=False)


# DEFINE NEW GIF MODEL (EXP BASED KERNELS)
####################################################

myGIF = GIF_HT(dt, 1, 1)

myGIF.eta = Filter_Exps()
myGIF.eta.setFilter_Timescales([10.0, 50.0, 250.0])
myGIF.eta.setFilter_Coefficients([0.2, 0.05, 0.025])

myGIF.gamma = Filter_Exps()
myGIF.gamma.setFilter_Timescales([5.0, 200.0, 250.0])
myGIF.gamma.setFilter_Coefficients([15.0, 3.0, 1.0])

# Simulate model response
(time, V, eta_sum, V_T, spks, rnd, l) = myGIF.simulate_seed(I, V0, u_vec=hurand)


# COMPARE MODELS
#prediction = SpikeTrainComparator(T_test, all_spks_times_testset, all_spks_times_prediction)


# PLOT MODEL RESPNOSES
####################################################

plb.figure('GIF', figsize=(16,8))
plt.subplot(3,1,1)
plt.plot(time, I, '-', color=gr1, label="Input current (C++)", linewidth=2)
plt.plot(htime, hi, '--', color=k1, label="Input current (NEURON)", linewidth=2)

plt.plot(time, eta_sum, '-', color=r1, label="AHP current (C++)", linewidth=2)
plt.plot(htime, heta_sum, '--', color=b1, label="AHP current (NEURON)", linewidth=2)
#plt.xlim(0,1000)
plt.ylabel("nA")
plt.legend()

plt.subplot(3,1,2)
plt.plot(time, V, '-', color=gr1, label="v(C++)", linewidth=2)
plt.plot(htime, hV, '--', color=k1, label="v(NEURON)", linewidth=2)

plt.plot(time, V_T, '-', color=r1, label="Threshold (C++)", linewidth=2)
plt.plot(htime, hV_T, '--', color=b1, label="Threshold (NEURON)", linewidth=2)
#plt.xlim(0,1000)
plt.ylabel("mV")
plt.legend()

plt.subplot(3,1,3)
#plt.plot(time, hp_dontspike)
error = V-hV
plt.plot(time, error, color=r1, label="v(C++) - v(NEURON)", linewidth=2)
#plt.xlim(0,1000)
plt.ylim(-0.1,0.1)
plt.ylabel("mV")
plt.legend()

plt.savefig('./Comparison_NEURON.pdf', dpi=300)
plt.show()


# PRINT MODEL PARAMETERS
####################################################

dic =  myGIF.getResultDictionary()
print dic['model']
