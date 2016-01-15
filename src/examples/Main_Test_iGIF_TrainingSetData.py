import sys
sys.path.append('../')

import matplotlib.pyplot as plt
import numpy as np
import copy

import scipy
from scipy import io
import cPickle as pkl

from Experiment import *

from GIF import *
from iGIF_NP import *
from iGIF_Na import *
from AEC_Badel import *

from Filter_Rect_LinSpaced import *
from Filter_Rect_LogSpaced import *
from Filter_Exps import *


"""
This script load some experimental data set acquired according to the experimental protocol
discussed in Pozzorini et al. PLOS Comp. Biol. 2015 and fit three different models:

- GIF      : standard GIF as in Pozzorini et al. 2015

- iGIF_NP  : inactivating GIF in which the threshold-voltage coupling is nonparametric 
             (ie, nonlinearity is expanded as a sum of rect functions). This model is the same
             as the iGIF_NP introduced in Mensi et al. PLOS Comp. Biol. 2016, except for the fact that
             the spike-triggered adaptation is current-based and not conductance-based. 
             
- iGIF_Na  : inactivating GIF in which the threshold-voltage coupling is modeled using the nonlinear
             function derived analytically from an HH model in Platkiewicz J and Brette R, PLOS CB 2011.
             The model is the same as the iGIF_Na introduced in Mensi et al. PLOS Comp. Biol. 2016, except for the fact that
             the spike-triggered adaptation is current-based and not conductance-based. 

These three models are fitted to training set data (as described in Pozzorini et al. 2015 for more info). 
The performance of the models is assessed on a test set (as described in Pozzorini et al. 2015) by computing
the spike train similarity measure Md*. The test dataset consists of 9 injections of a frozen noise signal
genrated according to an Ornstein-Uhlenbeck process whose standard deviation was modulated with a sin function.

A similar script (./src/Main_Test_iGIF_FIData.py) is provided that fit the model on the FI data set (as described in Mensi et al. 2016) and test
it on the same test set used here.
"""

#################################################################################################
# SETP 1: LOAD EXPERIMENTAL TRACES
#################################################################################################

# The experimental data being loaded have been acquired according to the protocol
# introduced in Pozzorini et al. PLOS Comp. Biol. 2015

experiment = Experiment('Fit to training set data', 0.1)


PATH = '../../data/fi/'

# Load AEC data
experiment.setAECTrace(PATH + 'Cell3_Ger1Elec_ch2_1007.ibw', 1.0, PATH + 'Cell3_Ger1Elec_ch3_1007.ibw', 1.0, 10000.0, FILETYPE='Igor')

# Load training set data
experiment.addTrainingSetTrace(PATH + 'Cell3_Ger1Training_ch2_1008.ibw', 1.0, PATH + 'Cell3_Ger1Training_ch3_1008.ibw', 1.0, 120000.0, FILETYPE='Igor')

# Load test set data
experiment.addTestSetTrace(PATH + 'Cell3_Ger1Test_ch2_1009.ibw', 1.0, PATH + 'Cell3_Ger1Test_ch3_1009.ibw', 1.0, 20000.0, FILETYPE='Igor')
experiment.addTestSetTrace(PATH + 'Cell3_Ger1Test_ch2_1010.ibw', 1.0, PATH + 'Cell3_Ger1Test_ch3_1010.ibw', 1.0, 20000.0, FILETYPE='Igor')
experiment.addTestSetTrace(PATH + 'Cell3_Ger1Test_ch2_1011.ibw', 1.0, PATH + 'Cell3_Ger1Test_ch3_1011.ibw', 1.0, 20000.0, FILETYPE='Igor')
experiment.addTestSetTrace(PATH + 'Cell3_Ger1Test_ch2_1012.ibw', 1.0, PATH + 'Cell3_Ger1Test_ch3_1012.ibw', 1.0, 20000.0, FILETYPE='Igor')
experiment.addTestSetTrace(PATH + 'Cell3_Ger1Test_ch2_1013.ibw', 1.0, PATH + 'Cell3_Ger1Test_ch3_1013.ibw', 1.0, 20000.0, FILETYPE='Igor')
experiment.addTestSetTrace(PATH + 'Cell3_Ger1Test_ch2_1014.ibw', 1.0, PATH + 'Cell3_Ger1Test_ch3_1014.ibw', 1.0, 20000.0, FILETYPE='Igor')
experiment.addTestSetTrace(PATH + 'Cell3_Ger1Test_ch2_1015.ibw', 1.0, PATH + 'Cell3_Ger1Test_ch3_1015.ibw', 1.0, 20000.0, FILETYPE='Igor')
experiment.addTestSetTrace(PATH + 'Cell3_Ger1Test_ch2_1016.ibw', 1.0, PATH + 'Cell3_Ger1Test_ch3_1016.ibw', 1.0, 20000.0, FILETYPE='Igor')
experiment.addTestSetTrace(PATH + 'Cell3_Ger1Test_ch2_1017.ibw', 1.0, PATH + 'Cell3_Ger1Test_ch3_1017.ibw', 1.0, 20000.0, FILETYPE='Igor')

           
#################################################################################################
# STEP 2: PERFORM ACTIVE ELECTRODE COMPENSATION
#################################################################################################

# Create new object to perform AEC
myAEC = AEC_Badel(experiment.dt)

# Define metaparametres
myAEC.K_opt.setMetaParameters(length=150.0, binsize_lb=experiment.dt, binsize_ub=2.0, slope=30.0, clamp_period=1.0)
myAEC.p_expFitRange = [3.0,150.0]  
myAEC.p_nbRep = 15     

# Assign myAEC to myExp and compensate the voltage recordings
experiment.setAEC(myAEC)  
experiment.performAEC()  

# Plot AEC filters (Kopt and Ke)
#myAEC.plotKopt()
#myAEC.plotKe()


#################################################################################################
# STEP 3A: FIT GIF MODEL (Pozzorini et al. 2015)
#################################################################################################

# More details on how to fit a simple GIF model to data can be found here: Main_TestGIF.py

GIF_fit       = GIF(experiment.dt)
    
GIF_fit.Tref  = 4.0                         
    
GIF_fit.eta   = Filter_Rect_LogSpaced() 
GIF_fit.eta.setMetaParameters(length=4000.0, binsize_lb=1.0, binsize_ub=1000.0, slope=7.0)

GIF_fit.gamma = Filter_Rect_LogSpaced() 
GIF_fit.gamma.setMetaParameters(length=4000.0, binsize_lb=5.0, binsize_ub=1000.0, slope=7.0)

GIF_fit.fit(experiment, DT_beforeSpike = 5.0)

GIF_fit.plotParameters()   


#################################################################################################
# STEP 3B: FIT iGIF_NP (Mensi et al. 2016 with current-based spike-triggered adaptation)
#################################################################################################

# Note that in the iGIF_NP model introduced in Mensi et al. 2016, the adaptation current is
# conductance-based (i.e., eta is a spike-triggered conductance).

# Define metaparameters used during the fit   

theta_inf_nbbins  = 8                            # Number of rect functions used to define the nonlinear coupling between
                                                 # membrane potential and firing threshold (note that the positioning of the rect function
                                                 # is computed automatically based on the voltage distribution).

theta_tau_all     = np.linspace(2.0, 15.0, 12)    # ms, set of timescales tau_thetea that will be explored during the fit
                                                 # tau_theta is the timescale of the threshold-voltage coupling


# Create the new model used for the fit

iGIF_NP_fit = iGIF_NP(experiment.dt)
    
iGIF_NP_fit.Tref  = GIF_fit.Tref                 # use the same absolute refractory period as in GIF_fit
iGIF_NP_fit.eta   = copy.deepcopy(GIF_fit.eta)   # use the same basis function as in GIF_fit for eta (filer coeff will be refitted)
iGIF_NP_fit.gamma = copy.deepcopy(GIF_fit.gamma) # use the same basis function as in GIF_fit for gamma (filer coeff will be refitted)
   
   
# Perform the fit

iGIF_NP_fit.fit(experiment, theta_inf_nbbins=theta_inf_nbbins, theta_tau_all=theta_tau_all, DT_beforeSpike=5.0)


# Plot optimal parameters

iGIF_NP_fit.plotParameters()  


###################################################################################################
# STEP 3C: FIT iGIF_Na (Mensi et al. 2016 with current-based spike-triggered adaptation)
###################################################################################################

# Note that in the iGIF_Na model introduced in Mensi et al. 2016, the adaptation current is
# conductance-based (i.e., eta is a spike-triggered conductance).

# Define metaparameters used during the fit

ki_bounds_Na      = [0.5, 6.0]                  # mV, interval over which the optimal parameter k_i (ie, Na inactivation slope) is searched 
ki_BRUTEFORCE_RESOLUTION = 6                    # number of parameters k_i considered in the fit (lin-spaced over ki_bounds_Na)

Vi_bounds_Na      = [-50.0, -35.0]              # mV, interval over which the optimal parameter V_i (ie, Na half inactivation voltage) is searched 
Vi_BRUTEFORCE_RESOLUTION = 6                    # number of parameters V_i considered in the fit (lin-spaced over Vi_bounds_Na)


# Create new iGIF_Na model that will be used for the fit

iGIF_Na_fit       = iGIF_Na(experiment.dt)                    
iGIF_Na_fit.Tref  = GIF_fit.Tref                 # use the same absolute refractory period as in GIF_fit
iGIF_Na_fit.eta   = copy.deepcopy(GIF_fit.eta)   # use the same basis function as in GIF_fit for eta (filer coeff will be refitted)
iGIF_Na_fit.gamma = copy.deepcopy(GIF_fit.gamma) # use the same basis function as in GIF_fit for gamma (filer coeff will be refitted)

# Compute set of values that will be tested for ki and Vi (these parameters are extracted using a brute force approach, as described in Mensi et al. 2016 PLOS Comp. Biol. 2016) 

ki_all = np.linspace(ki_bounds_Na[0], ki_bounds_Na[-1], ki_BRUTEFORCE_RESOLUTION)
Vi_all = np.linspace(Vi_bounds_Na[0], Vi_bounds_Na[-1], Vi_BRUTEFORCE_RESOLUTION)

# Fit the model

# Note that the second parameter provided by the input is the timescale theta_tau, ie the timescale of the threshold-votlage coupling.
# This parameter is not extracted form the data, but is assumed)
iGIF_Na_fit.fit(experiment, iGIF_NP_fit.theta_tau, ki_all, Vi_all, DT_beforeSpike=5.0, do_plot=True)

# Plot optimal parameters

iGIF_Na_fit.printParameters()

  
###################################################################################################
# STEP 4: EVALUATE MODEL PERFORMANCES ON THE TEST SET DATA
###################################################################################################

models = [GIF_fit, iGIF_NP_fit, iGIF_Na_fit]
labels = ['GIF', 'iGIF_NP', 'iGIF_Na']

for i in np.arange(len(models)) :

    model = models[i]

    # predict spike times in test set
    prediction = experiment.predictSpikes(model, nb_rep=500)
    
    print "\n Model: ", labels[i]
    
    # compute Md*
    
    Md = prediction.computeMD_Kistler(4.0, 0.1) 
       
    #prediction.plotRaster(delta=1000.0) 


###################################################################################################
# STEP 5: COMPARE OPTIMAL PARAMETERS OF iGIF_NP AND iGIF_Na
###################################################################################################

iGIF.compareModels([iGIF_NP_fit, iGIF_Na_fit], labels=['iGIF_NP', 'iGIF_Na'])
