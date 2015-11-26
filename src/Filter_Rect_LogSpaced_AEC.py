import matplotlib.pyplot as plt
import numpy as np

from Filter_Rect_LogSpaced import *
from scipy.signal import fftconvolve

from Filter import *


class Filter_Rect_LogSpaced_AEC(Filter_Rect_LogSpaced) :

    """
    This class define a function of time expanded using log-spaced rectangular basis functions.
    A filter f(t) is defined in the form f(t) = sum_j b_j*rect_j(t),
    where b_j is a set of coefficient and rect_j is a set of rectangular basis functions.
    The width of the rectangular basis functions increase exponentially (log-spacing).
    This class is used to define both the spike-triggered current eta(t) and the spike-triggered
    movement of the firing threshold gamma(t).
    """
    
    
    def __init__(self, length=1000.0, binsize_lb=2.0, binsize_ub=1000.0, slope=7.0, clamp_period=1.0):
        
        self.p_clamp_period = clamp_period
         
        Filter_Rect_LogSpaced.__init__(self, length=length, binsize_lb=binsize_lb, binsize_ub=binsize_ub, slope=slope)
        
          
        # Initialize        
        self.computeBins()                   # using meta parameters self.metaparam_subthreshold define bins and support.
        self.setFilter_toZero()              # initialize filter to 0
     
         
    ########################################################################################
    # AUXILIARY METHODS USED BY THIS PARTICULAR IMPLEMENTATION OF FILTER
    ########################################################################################
    
    def computeBins(self) :
        
        """
        This function compute bins and support given the metaparameters.
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
        self.support = np.array( [ (self.bins[i]+self.bins[i+1])/2 for i in range(len(self.bins)-1) ])
        self.bins_l = len(self.bins)-1



    def setMetaParameters(self, length=1000.0, binsize_lb=2.0, binsize_ub=1000.0, slope=7.0, clamp_period=10.0):
       
        super(Filter_Rect_LogSpaced_AEC, self).setMetaParameters(length=length, binsize_lb=binsize_lb, binsize_ub=binsize_ub, slope=slope)
        self.p_clamp_period   = clamp_period
                
        self.computeBins()
        self.setFilter_toZero()
        

        
        
        