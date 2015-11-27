import matplotlib.pyplot as plt
import numpy as np

from Filter_Rect_LogSpaced import *
from scipy.signal import fftconvolve

from Filter import *


class Filter_Rect_LogSpaced_AEC(Filter_Rect_LogSpaced) :

    """
    This class define a function of time expanded using log-spaced rectangular basis functions.    
    Using the metaparameter p_clamp_period, one can force the rectangular basis functions covering
    the first p_clamp_period ms to have a to have a specific size binsize_lb. 
    Log-spacing only starts after p_clamp_period.
    """
    
    def __init__(self, length=1000.0, binsize_lb=2.0, binsize_ub=1000.0, slope=7.0, clamp_period=1.0):
        
        # Metaparameters
        
        self.p_clamp_period = clamp_period
         
        Filter_Rect_LogSpaced.__init__(self, length=length, binsize_lb=binsize_lb, binsize_ub=binsize_ub, slope=slope)
        
          
        # Initialize    
            
        self.computeBins()                   # using meta parameters self.metaparam_subthreshold define bins and support.
        
        self.setFilter_toZero()              # initialize filter to 0
     

    ################################################################
    # OVERVRITE METHODS OF Filter_Rect_LogSpaced
    ################################################################
    
    def setMetaParameters(self, length=1000.0, binsize_lb=2.0, binsize_ub=1000.0, slope=7.0, clamp_period=10.0):
        
        # Set metaparameters inherited from  Filter_Rect_LogSpaced 
        
        super(Filter_Rect_LogSpaced_AEC, self).setMetaParameters(length=length, binsize_lb=binsize_lb, binsize_ub=binsize_ub, slope=slope)
        
        
        # Set paramters which are specific to this class
        
        self.p_clamp_period   = clamp_period
                
        self.computeBins()
            
        self.setFilter_toZero()

    
    def computeBins(self) :
        
        """
        This function compute bins and support given metaparameters.
        """
        
        self.bins = []
        self.bins.append(0)
        
        total_length = 0
                        
        
        for i in np.arange(int (self.p_clamp_period/self.p_binsize_lb ) ) :
            total_length = total_length + self.p_binsize_lb
            self.bins.append( total_length )
        
        cnt = 1
 
        while (total_length <= self.p_length) :  
            tmp = min( self.p_binsize_lb*np.exp(cnt/self.p_slope), self.p_binsize_ub )
            total_length = total_length + tmp
            self.bins.append( total_length )
    
            cnt+=1
    
        self.bins = np.array(self.bins)
        
        self.computeSupport()
        
        self.filter_coeffNb = len(self.bins)-1
        

        
        
        