import sys
sys.path.append('../')
sys.path.append('/Users/christianpozzorini/Documents/workspace/GIFToolbox/src/')

import matplotlib.pyplot as plt
import numpy as np
import hickle
import h5py
import glob
import re
import cPickle as pickle
import time

from Experiment import *
from GIF import *
from GIF_HT import *
from Filter_Rect_LogSpaced import *
from Filter_Exps import *


##########################################################################################
# METAPARAMETERS - MODEL
##########################################################################################

DT_beforespike = 5.0            # ms, metaparameter for fitting subthreshold dynamics: defines the region before spike
                                # which is excluded from the liner regression 
T_ref = 3.0                     # ms, absolute refractory period for threshold detection
tau_gamma = [10.0, 50.0, 250.0] # ms, timescales used to descripe the spike triggered threshold movement
print_and_plot = False          # plot and print results while executing

eta_tau_max = 1000.0            # ms, longest timescale allowed for the adaptation process


##########################################################################################
# METAPARAMETERS - DATA
##########################################################################################

base_dir = "/Users/christianpozzorini/Desktop/soma_corr_fil_somacurr_test/"
depols = ["K4p5", "K5p0", "K6p5"]
sweeps_train = ["Ca1p25_1","Ca1p25_2", "Ca1p25_3"]
sweeps_validation = ["Ca1p25_4"]


dt = 0.1                        # ms, experimental sampling frequency

spk_detection_Vth = -20.0       # mV, threshold on voltage to extract spike times from voltage traces                  
spk_detection_Tref = 2.0        # ms, absolute refractory period used for extracting spike times from voltage traces   

data_T   = 20000                # ms, length of each experimental trace
trainingset_ROI = [1000, 20000] # ms, temporal interval of each trace used to fit the model
testset_ROI = [1000, 20000]     # ms, temporal interval of each trace used to evaluate the model perfomrance



##########################################################################################
# MAIN FUNCTION FIT
##########################################################################################



def fitAndValidateGIFModel(gid, depols, data):     
    
    tic = time.time()
    
    nbOfSpikes_TrainingSet = data.getTrainingSetNbOfSpikes()
    print "Number of spikes in training set (ROI): ", int(nbOfSpikes_TrainingSet)
    
    if nbOfSpikes_TrainingSet >= 5 :
    
        #####################################################################################################################
        # FIT SUBTHRESHOLD ACTIVITY WITH RECT-BASED GIF MODEL: EXTRACT OPTIMAL TIMESCALES FOR SPIKE-TRIGGERED CURRENT
        #####################################################################################################################
        myGIF_rect = GIF_HT(dt, depols, gid)
        
        myGIF_rect.Tref = T_ref 
        myGIF_rect.eta = Filter_Rect_LogSpaced()
        myGIF_rect.eta.setMetaParameters(length=500.0, binsize_lb=2.0, binsize_ub=100.0, slope=4.5)
        myGIF_rect.fitVoltageReset(data, myGIF_rect.Tref, do_plot=False)
        myGIF_rect.fitSubthresholdDynamics(data, DT_beforeSpike=DT_beforespike)
        
        # Fit sum of 3 exps on spike-triggered current (timescales slower than 500ms are excluded)
        myGIF_rect.eta.fitSumOfExponentials(3, [ 1.0, 0.5, 0.1], tau_gamma, ROI=None, dt=0.1) 
        print "Optimal timescales: ", myGIF_rect.eta.tau0
        
        tau_opt = [t for t in myGIF_rect.eta.tau0 if t < eta_tau_max]
    
        
        #####################################################################################################################
        # FIT GIF
        #####################################################################################################################
        
        myGIF = GIF_HT(dt, depols, gid)
        
        myGIF.Tref = T_ref
        myGIF.eta = Filter_Exps()
        myGIF.eta.setFilter_Timescales(tau_opt)
        myGIF.gamma = Filter_Exps()
        myGIF.gamma.setFilter_Timescales(tau_gamma)
                
        myGIF.fit(data, DT_beforeSpike=DT_beforespike)
        
        
        #####################################################################################################################
        # EVALUATE PERFORMANCES
        #####################################################################################################################
    
        myGIF.computePerf(data, DT_beforeSpike=5.0)
        myGIF.computePerfDropDueToExp(myGIF_rect)
        myGIF.cputime = CPU_time = time.time() - tic
    
        return myGIF

    else :
        
        print "Less than 5 spikes in Training Set. Cannot fit GIF model to data."
        
        myGIF = GIF_HT(dt, depols, gid)
        myGIF.eta = Filter_Exps()
        myGIF.gamma = Filter_Exps()
        
        myGIF.fit_problem = True
        myGIF.fit_problem_which.append("7")        
        
        return myGIF





##########################################################################################
# FIT DATA
##########################################################################################

results = []

# compute number of cells to be fitted
path = depols[0] + "/" + sweeps_train[0] +  "/h5_ready/"
filepaths = glob.glob(base_dir + path + "a*.h5")
nb_cells = len(filepaths)


problems_cnt = 0
L = []
Vexp = []

for cell_id in np.arange(0,nb_cells) :
    
    print "\n\n----------------------------------------------"
    print "CELL NUMBER: ", cell_id
    print "----------------------------------------------\n"
    
    myExp = Experiment("Experiment name", dt)
    
    for depol in depols:
        
        for sweep in sweeps_train + sweeps_validation:
    
            path = depol + "/" + sweep +  "/h5_ready/"
            filepaths = glob.glob(base_dir + path + "a*.h5")
        
            for filepath in filepaths[cell_id:cell_id+1]:                           #remove [:1] to loop over all cells

                # load current and voltage

                with h5py.File(filepath,'r') as f:
                    
                    gid = re.match(".*\/a(\d*).h5", filepath).group(1) # get gid from filename
    
                    #myExp.name = "GID: " + gid
    
                    i_hold = np.array(f["a" + str(gid) + "/IClamp_soma/data"])
                    i_syn = np.array(f["a" + str(gid) + "/ISyn_soma/data"])
    
                    # combine both currents, synaptic current has to be inverted!!!
                    i = i_hold - i_syn
                    i = np.array(i)
                    i = i.flatten()
                                        
                    v = np.array(f["a" + str(gid) + "/V_soma/data"])
                    v = np.array(v)
                    v= v.flatten()                    
                    
                    # load data to Experiment
                                        
                    if sweep in sweeps_train :
    
                        tmp_trace = myExp.addTrainingSetTrace(v, 10**-3, i, 10**-9, data_T, FILETYPE='Array')
                        tmp_trace.setROI([trainingset_ROI])
                            
                    elif sweep in sweeps_validation :
                        
                        tmp_trace = myExp.addTestSetTrace(v, 10**-3, i, 10**-9, data_T, FILETYPE='Array')
                        tmp_trace.setROI([trainingset_ROI])


    # fit GIF model to data
    myExp.detectSpikes(threshold=spk_detection_Vth, ref=spk_detection_Tref) 
    myGIF = fitAndValidateGIFModel(gid, depols, myExp)
    myGIF.printSummaryData()
    
    # store results
    res_dic = myGIF.getResultDictionary()    
    results.append(res_dic)
    
    
pickle.dump(results, open( "/Users/christianpozzorini/Desktop/HBP_modelparam.pkl", "wb" ) )



