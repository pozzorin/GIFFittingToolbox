import sys
sys.path.append('../')

import matplotlib.pyplot as plt
import copy

from Experiment import *
from AEC_Badel import *

from GIF import *
from gGIF import *

from Filter_Rect_LogSpaced import *
from Filter_Rect_LinSpaced import *

import Tools



"""
This script fit 2  models on the same experimental data:

- model 1: GIF  : a standard GIF described in Pozzorini et al. PLOS Comp. Biol. 2015
- model 2: gGIF : same GIF as described in Pozzorini et al. PLOS Comp. Biol. 2015 exepct for
                  the fact that the spike-triggered current is transformed into a spike-triggered
                  conductance (as discussed in Mensi et al. 2016).
"""

############################################################################################################
# STEP 1: LOAD EXPERIMENTAL DATA
############################################################################################################

myExp = Experiment('Experiment 1', 0.1)

PATH = '../../data/gif_test/'

# Load AEC data
myExp.setAECTrace(PATH + 'Cell3_Ger1Elec_ch2_1007.ibw', 1.0, PATH + 'Cell3_Ger1Elec_ch3_1007.ibw', 1.0, 10000.0, FILETYPE='Igor')

# Load training set data
myExp.addTrainingSetTrace(PATH + 'Cell3_Ger1Training_ch2_1008.ibw', 1.0, PATH + 'Cell3_Ger1Training_ch3_1008.ibw', 1.0, 120000.0, FILETYPE='Igor')
myExp.trainingset_traces[0].setROI([[0,100000.0]])

# Load test set data
myExp.addTestSetTrace(PATH + 'Cell3_Ger1Test_ch2_1009.ibw', 1.0, PATH + 'Cell3_Ger1Test_ch3_1009.ibw', 1.0, 20000.0, FILETYPE='Igor')
myExp.addTestSetTrace(PATH + 'Cell3_Ger1Test_ch2_1010.ibw', 1.0, PATH + 'Cell3_Ger1Test_ch3_1010.ibw', 1.0, 20000.0, FILETYPE='Igor')
myExp.addTestSetTrace(PATH + 'Cell3_Ger1Test_ch2_1011.ibw', 1.0, PATH + 'Cell3_Ger1Test_ch3_1011.ibw', 1.0, 20000.0, FILETYPE='Igor')
myExp.addTestSetTrace(PATH + 'Cell3_Ger1Test_ch2_1012.ibw', 1.0, PATH + 'Cell3_Ger1Test_ch3_1012.ibw', 1.0, 20000.0, FILETYPE='Igor')
myExp.addTestSetTrace(PATH + 'Cell3_Ger1Test_ch2_1013.ibw', 1.0, PATH + 'Cell3_Ger1Test_ch3_1013.ibw', 1.0, 20000.0, FILETYPE='Igor')
myExp.addTestSetTrace(PATH + 'Cell3_Ger1Test_ch2_1014.ibw', 1.0, PATH + 'Cell3_Ger1Test_ch3_1014.ibw', 1.0, 20000.0, FILETYPE='Igor')
myExp.addTestSetTrace(PATH + 'Cell3_Ger1Test_ch2_1015.ibw', 1.0, PATH + 'Cell3_Ger1Test_ch3_1015.ibw', 1.0, 20000.0, FILETYPE='Igor')
myExp.addTestSetTrace(PATH + 'Cell3_Ger1Test_ch2_1016.ibw', 1.0, PATH + 'Cell3_Ger1Test_ch3_1016.ibw', 1.0, 20000.0, FILETYPE='Igor')
myExp.addTestSetTrace(PATH + 'Cell3_Ger1Test_ch2_1017.ibw', 1.0, PATH + 'Cell3_Ger1Test_ch3_1017.ibw', 1.0, 20000.0, FILETYPE='Igor')


############################################################################################################
# STEP 2: ACTIVE ELECTRODE COMPENSATION
############################################################################################################

# Create new object to perform AEC
myAEC = AEC_Badel(myExp.dt)

# Define metaparametres
myAEC.K_opt.setMetaParameters(length=150.0, binsize_lb=myExp.dt, binsize_ub=2.0, slope=30.0, clamp_period=1.0)
myAEC.p_expFitRange = [3.0,150.0]  
myAEC.p_nbRep = 15     

# Assign myAEC to myExp and compensate the voltage recordings
myExp.setAEC(myAEC)  
myExp.performAEC()  

############################################################################################################
# STEP 3A: FIT GIF MODEL TO DATA
############################################################################################################

# Create a new object GIF 
myGIF = GIF(0.1)

# Define parameters
myGIF.Tref = 4.0  

myGIF.eta = Filter_Rect_LogSpaced()
myGIF.eta.setMetaParameters(length=500.0, binsize_lb=2.0, binsize_ub=1000.0, slope=4.5)

myGIF.gamma = Filter_Rect_LogSpaced()
myGIF.gamma.setMetaParameters(length=500.0, binsize_lb=5.0, binsize_ub=1000.0, slope=5.0)

# Perform the fit
myGIF.fit(myExp, DT_beforeSpike=5.0)

# Plot the model parameters
myGIF.printParameters()
myGIF.plotParameters()   


############################################################################################################
# STEP 3B: FIT gGIF MODEL TO DATA
############################################################################################################


mygGIF = gGIF(0.1)

# Set absolute refractory period
mygGIF.Tref  = myGIF.Tref 

# Define metaparameters used for the fit
mygGIF.eta   = copy.deepcopy(myGIF.eta)
mygGIF.gamma = copy.deepcopy(myGIF.gamma) 

Ek_all = np.linspace(-90, -40, 20)          # set of values tested for the reversal potential associated with
                                            # the spike triggered current (this parameter is extracted from data using 
                                            # a brute force strategy as described in Mensi et al. 2016).



# Perform the fit
mygGIF.fit(myExp, Ek_all, DT_beforeSpike=5.0, do_plot=True)
        
# Plot the model parameters
mygGIF.printParameters()
mygGIF.plotParameters()   


###################################################################################################
# STEP 4: EVALUATE MODEL PERFORMANCES ON THE TEST SET DATA
###################################################################################################

models = [myGIF, mygGIF]

for i in np.arange(len(models)) :

    model = models[i]

    # predict spike times in test set
    prediction = myExp.predictSpikes(model, nb_rep=500)
    
    # compute Md*
    
    Md = prediction.computeMD_Kistler(4.0, 0.1) 
       
    # plot raster   
    
    prediction.plotRaster(delta=1000.0) 

