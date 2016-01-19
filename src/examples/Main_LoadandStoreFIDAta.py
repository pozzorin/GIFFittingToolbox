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

from Filter_Rect_LinSpaced import *
from Filter_Rect_LogSpaced import *


"""
This script load some experimental data set acquired according to the experimental protocol
discussed in Mensi et al. PLOS Comp. Biol. 2016 (FI curves) and fit three different models:

- GIF      : standard GIF as in Pozzorini et al. 2015

- iGIF_NP  : inactivating GIF in which the threshold-voltage coupling is nonparametric 
             (ie, nonlinearity is expanded as a sum of rect functions). This model is the same
             as the iGIF_NP introduced in Mensi et al. PLOS Comp. Biol. 2016, except for the fact that
             the spike-triggered adaptation is current-based and not conductance-based. 
             
- iGIF_Na  : inactivating GIF in which the threshold-voltage coupling is modeled using the nonlinear
             function derived analytically from an HH model in Platkiewicz J and Brette R, PLOS CB 2011.
             The model is the same as the iGIF_Na introduced in Mensi et al. PLOS Comp. Biol. 2016, except for the fact that
             the spike-triggered adaptation is current-based and not conductance-based. 

The performance of the models is assessed on a test set (as described in Pozzorini et al. 2015) by computing
the spike train similarity measure Md*. The test dataset consists of 9 injections of a frozen noise signal
genrated according to an Ornstein-Uhlenbeck process whose standard deviation was modulated with a sin function.

A similar script (./src/Main_Test_iGIF_TrainingSetDAta.py) is provided that fit the model on the Training Dataset
(as described in Mensi et al. 2016). The training data set is a long injection of a current drawn from the same 
stochastic process as the test set.
"""

#################################################################################################
# SETP 1: LOAD EXPERIMENTAL TRACES
#################################################################################################

# Define which data to load and use for the fit.
#
# The experimental data consists of 3 long injections (i.e. 3 repetitions of the experiment).
# Each of this injection consists of 4 * 8 = 32 steps of current.
# Each step of current is generated using an Ornstein-Uhlenbeck process with mean_I and sigma_I.
# - 8 different values of mean_I were considered.
# - 4 different values of sigma_I were considered.
# In each repetition, the order of the steps was randomly shuffled and a new realization of the OU process was used.
# More details on the experimental protocol can be found in Mensi et al.PLOS Comp. Biol. 2016.
#
# Data are stored at the following path:
#
PATH = '../../mydata/fi/'
# 
# in a .mat file called FI_DATA_170413A3.mat and have already been preprocessed with AEC
# using the method discussed in Mensi et al. PLOS Comp. Biol. 2016.
#
# Test set data are stored in separate .ibw files and have not yet been preprocessed with AEC.
# Since in this script the test dataset is only used to compute Md*, AEC is not needed and raw data
# can be directly used.


# Select the data from the FI protocol that will be used for the fit:
#
mm      = [ 0,1,2,3,4,5,6,7 ]  # select which mean input mu_I (between 0-7: from 0 nA to mu_max, cell dependent)
ss      = [ 1,2,3 ]            # select which standard deviation sigma_I (between (0-3: from 0nA to 150 pA)
rr      = [ 1 ]                # select which repetitions (between 0-2)


# Load experimental data and create an object experiment

data = io.loadmat(PATH + 'FI_DATA_170413A3.mat')
data = data['Data']
dt   = data['dt'][0][0][0][0]

experiment = Experiment('Fit to FI data', dt)

# loop on standard deviations sigma_I (ie, amplitude of input fluctuations)
for s in ss :   
      
    # loop on repetitions (at each repetition, a new realization of the OU process was used, 
    # so these are not frozen noise repetitions)  
    for r in rr :
           
        # loop over means mu_I (ie., DC component of the current)
        for m in mm :
            
            # Load the input current and the voltage recording associated with a repetition 
            # and a particular mean and standard deviation (mu_I, sigma_I)
            
            V_tmp = data['V_traces'][0][0][s][r][m][-55000:]
            I_tmp = data['I_traces'][0][0][s][r][m][-55000:]                        
            
            tr = experiment.addTrainingSetTrace(V_tmp, 10**-3, I_tmp, 10**-9, 5500.0, FILETYPE='Array')

      
# Load test set data to evaluate the model performance
"""
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
"""
experiment.save('/Users/christianpozzorini/Desktop')