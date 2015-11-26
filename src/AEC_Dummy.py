import numpy as np

from AEC import *
from Experiment import *


class AEC_Dummy(AEC) :

    """
    Assume that the recorded voltage is not biased. 
    The membrane potential V is therefore assumed to be equal to V_rec. 
    """
    
    def performAEC(self, experiment):

        print "\nDO NOT PERFORM ACTIVE ELECTRODE COMPENSATION..."

        if experiment.AEC_trace != 0 :
            
            # Copy recorded voltage V_rec into membrane potential V.
            experiment.AEC_trace.V        = experiment.AEC_trace.V_rec 
            experiment.AEC_trace.AEC_flag = True    
         
        for tr in experiment.trainingset_traces :
            tr.V        = tr.V_rec 
            tr.AEC_flag = True 
            tr.detectSpikes()   
                       
        for tr in experiment.testset_traces :
            tr.V        = tr.V_rec 
            tr.AEC_flag = True   
            tr.detectSpikes()   
                 
        print "Done!"
    
    
    def plot(self):
           
        print "Dummy AEC does not use filters. Nothing to plot."
        