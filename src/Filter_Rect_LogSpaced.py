import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import fftconvolve
import Tools

from Filter import *
from Filter_Rect import *

class Filter_Rect_LogSpaced(Filter_Rect) :

    """
    This class defines a temporal filter defined as a linear combination of log-spaced rectangular basis functions.
    """

    def __init__(self, length=1000.0, binsize_lb=2.0, binsize_ub=1000.0, slope=7.0):
        
        Filter_Rect.__init__(self)
        
        # Metaparamters
        
        self.p_length     = length           # ms, filter length
        
        self.p_binsize_lb = binsize_lb       # ms, min size for bin
        
        self.p_binsize_ub = binsize_ub       # ms, max size for bin
        
        self.p_slope      = slope            # exponent for log-scaling  
        
        
        # Initialize      
          
        self.computeBins()                   # using meta parameters self.metaparam_subthreshold define bins and support.
        
        self.setFilter_toZero()              # initialize filter to 0
     


    def setMetaParameters(self, length=1000.0, binsize_lb=2.0, binsize_ub=1000.0, slope=7.0):

        """
        Set the parameters defining the rectangular basis functions.
        Each time meta parameters are changeD, the value of the filer is reset to 0.
        """
        
        self.p_length     = length                  # ms, filter length
        
        self.p_binsize_lb = binsize_lb              # ms, min size for bin
        
        self.p_binsize_ub = binsize_ub              # ms, max size for bin
        
        self.p_slope      = slope                   # exponent for log-scale binning  
        
        self.computeBins()
        
        self.setFilter_toZero()                     # initialize filter to 0
                
    
    ################################################################
    # IMPLEMENT ABSTRACT METHODS OF Filter_Rect
    ################################################################
               
    def computeBins(self) :
        
        """
        This function compute log-spaced bins and support given the metaparameters.
        """
        
        self.bins = []
        self.bins.append(0)
        
        cnt = 1
        total_length = 0
        
        while (total_length <= self.p_length) :  
            tmp = min( self.p_binsize_lb*np.exp(cnt/self.p_slope), self.p_binsize_ub )
            total_length = total_length + tmp
            self.bins.append( total_length )
    
            cnt+=1
    
        self.bins = np.array(self.bins)
        
        self.computeSupport()
        
        self.filter_coeffNb = len(self.bins)-1

        
        

    ################################################################
    # IMPLEMENT ABSTRACT METHODS OF Filter
    ################################################################
        
    def convolution_Spiketrain_basisfunctions(self, spks, T, dt):
        
        """
        Filter spike train spks with the set of rectangular basis functions defining the Filter.
        """
        
        T_i     = int(T/dt)
                       
        bins_i = Tools.timeToIndex(self.bins, dt)
        spks_i = Tools.timeToIndex(spks, dt)
 
        nb_bins = self.getNbOfBasisFunctions()
        
        X = np.zeros( (T_i, nb_bins) )
        
        # Fill matrix
        for l in np.arange(nb_bins) :
                        
            tmp = np.zeros( T_i + bins_i[-1] + 1 )
            
            for s in spks_i :
                lb = s + bins_i[l]
                ub = s + bins_i[l+1]
                tmp[lb:ub] += 1
            
            X[:,l] = tmp[:T_i]
        
        
        return X
    
    
    def convolution_ContinuousSignal_basisfunctions(self, I, dt):
        
        """
        Filter continuous input I with the set of rectangular basis functions defining the Filter.
        """
        
        T_i     = len(I)
        
        bins_i  = Tools.timeToIndex(self.bins, dt)                
        bins_l  = self.getNbOfBasisFunctions()
        
        X = np.zeros( (T_i, bins_l) )
        I_tmp = np.array(I,dtype='float64')        
        
        # Fill matrix
        for l in np.arange(bins_l) :
            
            window = np.ones( bins_i[l+1] - bins_i[l])
            window = np.array(window,dtype='float64')  
        
            F_star_I = fftconvolve(window, I_tmp, mode='full')*dt
            F_star_I = F_star_I[: int(len(I))]        
        
            F_star_I_shifted = np.concatenate( ( np.zeros( int(bins_i[l]) ), F_star_I) )
            
            X[:,l] = np.array(F_star_I_shifted[:T_i], dtype='double')
        
        
        return X
        