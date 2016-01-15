import matplotlib.pyplot as plt
import numpy as np

from SpikeTrainComparator import *


s_reference = np.random.rand(100)*10000.0       # generate reference spike train at 10 Hz  
    
data_stochasticity = 10.0                       # ms, standard deviation of the gaussian jitters

# consider different level of stochasticity in the predicted spke train and evaluate Md* for each of them
model_stochasticity_all = [ x*data_stochasticity for x in np.arange(5)/2.0  ]

Md_all = []
model_stochasticity_all_plot = []
nb_rep = 20

for model_stochasticity in model_stochasticity_all :

    for x in np.arange(nb_rep) :
        
        s_data = []
        s_model = []
    
        print "Model stochasticity: ", model_stochasticity
    
        for i in np.arange(3) :
            
            s_data.append(s_reference + np.random.randn(len(s_reference))*data_stochasticity)      
            
        for i in np.arange(1000) :
                    
            s_model.append(s_reference + np.random.randn(len(s_reference))*model_stochasticity)
            
            
        stc = SpikeTrainComparator(11000.0, s_data, s_model)
        
        # to visualize the raster uncomment the following line
        #stc.plotRaster(10.0, 0.1)
        
        Md = stc.computeMD_Kistler(4.0, 1.0)
        
        Md_all.append(Md)
        model_stochasticity_all_plot.append(model_stochasticity)
    
plt.figure()
plt.ylabel('Md*')
plt.plot(model_stochasticity_all_plot, Md_all, '.')
plt.title("Reference spike trains have been jittered with a Gaussian of stdev of 10 ms")
plt.xlabel("Jitter in predicted spikes (ms)")
plt.ylim([0,np.max(Md_all)])
plt.xlim([-1,21])
plt.show()