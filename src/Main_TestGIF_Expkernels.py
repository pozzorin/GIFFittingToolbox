"""
THIS SCRIPT LOADS SOME EXPERIMENTAL DATA AND FIT TWO VERSIONS OF THE GIF MODEL:

- model 1 myGIF_rect : a standard GIF in which eta and gamma are expanded in a set of rectangular basis functions
- model 2 myGIF_exp  : a GIF model in which eta and gamma are expanded in a set of exponential functions (with given timescales)

"""

import matplotlib.pyplot as plt

from Experiment import *
from AEC_Badel import *

from GIF import *

from Filter_Rect_LogSpaced import *
from Filter_Rect_LinSpaced import *

from Filter_Exps import *

import Tools


PATH = '../data/gif_test/'


############################################################################################################
# STEP 1: LOAD EXPERIMENTAL DATA
############################################################################################################

myExp = Experiment('Experiment 1', 0.1)

# Load AEC data
myExp.setAECTrace(PATH + 'Cell3_Ger1Elec_ch2_1007.ibw', 1.0, PATH + 'Cell3_Ger1Elec_ch3_1007.ibw', 1.0, 10000.0, FILETYPE='Igor')

# Load training set data
myExp.addTrainingSetTrace(PATH + 'Cell3_Ger1Training_ch2_1008.ibw', 1.0, PATH + 'Cell3_Ger1Training_ch3_1008.ibw', 1.0, 120000.0, FILETYPE='Igor')

# Specify the region of the training set that will be used to fit the models (here first 10 seconds)
myExp.trainingset_traces[0].setROI([[0,10000.0]])


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
# STEP 2A: ACTIVE ELECTRODE COMPENSATION
############################################################################################################

# Create new object to perform AEC
myAEC = AEC_Badel(myExp.dt)

# Define metaparametres
myAEC.K_opt.setMetaParameters(length=150.0, binsize_lb=myExp.dt, binsize_ub=2.0, slope=30.0, clamp_period=1.0)
myAEC.p_expFitRange = [3.0,150.0]  
myAEC.p_nbRep = 5     

# Assign myAEC to myExp and compensate the voltage recordings
myExp.setAEC(myAEC)  
myExp.performAEC()  


############################################################################################################
# STEP 2B: TO NOT PERFORM ACTIVE ELECTRODE COMPENSATION DO NOT RUN STEP 2A AND EXECUTE STEP 2B INSTEAD
############################################################################################################

# to do


############################################################################################################
# STEP 3A: FIT GIF WITH RECT BASIS FUNCTIONS TO DATA
############################################################################################################

# Create a new object GIF 
myGIF_rect = GIF(0.1)

# Define parameters
myGIF_rect.Tref = 4.0  

myGIF_rect.eta = Filter_Rect_LogSpaced()
myGIF_rect.eta.setMetaParameters(length=5000.0, binsize_lb=2.0, binsize_ub=1000.0, slope=4.5)

myGIF_rect.gamma = Filter_Rect_LogSpaced()
myGIF_rect.gamma.setMetaParameters(length=5000.0, binsize_lb=5.0, binsize_ub=1000.0, slope=5.0)

# Perform the fit
myGIF_rect.fit(myExp, DT_beforeSpike=5.0)


############################################################################################################
# STEP 3B: FIT GIF WITH EXP BASIS FUNCTIONS TO DATA
############################################################################################################

# Create a new object GIF 
myGIF_exp = GIF(0.1)

# Define parameters
myGIF_exp.Tref = 4.0  

myGIF_exp.eta = Filter_Exps()
myGIF_exp.eta.setFilter_Timescales([1.0, 5.0, 30.0, 70.0, 100.0, 500.0])
 
myGIF_exp.gamma = Filter_Exps()
myGIF_exp.gamma.setFilter_Timescales([1.0, 5.0, 30.0, 70.0, 100.0, 500.0])

# Perform the fit
myGIF_exp.fit(myExp, DT_beforeSpike=5.0)


############################################################################################################
# STEP 4: COMPARE MODLES BY PREDICTING SPIKES IN TEST SET
############################################################################################################

# Use the myGIF model to predict the spiking data of the test data set in myExp
myPredictionGIF_rect = myExp.predictSpikes(myGIF_rect, nb_rep=500)
myPredictionGIF_exp  = myExp.predictSpikes(myGIF_exp, nb_rep=500)


print "Model performance (Md*):"
print "GIF rect: ", myPredictionGIF_rect.computeMD_Kistler(4.0, 0.1) 
print "GIF exp: ",  myPredictionGIF_exp.computeMD_Kistler(4.0, 0.1)     


############################################################################################################
# STEP 5: COMPARE MODEL PARAMTERS
############################################################################################################

GIF.compareModels([myGIF_rect, myGIF_exp], labels=['GIF rect', 'GIF exp'])





