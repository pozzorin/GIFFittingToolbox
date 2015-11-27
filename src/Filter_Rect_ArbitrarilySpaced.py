import matplotlib.pyplot as plt
import numpy as np

from Filter import *
from Filter_Rect import *
import Tools


class Filter_Rect_ArbitrarilySpaced(Filter_Rect) :

    """
    This class define a function of time expanded using a set of arbitrarily rectangular basis functions.
    A filter f(t) is defined in the form 
    
    f(t) = sum_j b_j*rect_j(t),
    
    where b_j is a set of coefficient and rect_j is a set of rectangular basis functions.
    The width and size of each rectangular basis function is free (it is not restricted to, eg, lin spaced).
    """

    def __init__(self, bins=np.array([0.0,10.0,50.0,100.0,1000.0])) :
        
        Filter_Rect.__init__(self)        
               
        # Initialize    
            
        self.bins = bins 
        
        self.filter_coeffNb = len(bins)-1
        
        self.computeSupport()      
                      
        self.setFilter_toZero()              # initialize filter to 0
 



    def setBasisFunctions(self, bins):

        """
        Set the parameters defining the rectangular basis functions.
        Attention, each time meta parameters are changes, the value of the filer is reset to 0.
        """

        self.bins = np.array(bins)
        
        self.computeSupport()

        self.filter_coeffNb = len(bins)-1
                
        self.setFilter_toZero()
        
        
        
    ################################################################
    # IMPLEMENT ABSTRACT METHODS OF Filter_Rect
    ################################################################

    def computeBins(self):
        
        """
        This filter implementation does not have metaparameters. Filters are direcly set and don't need to be computed.
        """
        
        pass


    ################################################################
    # IMPLEMENT ABSTRACT METHODS OF Filter
    ################################################################
        
    def convolution_Spiketrain_basisfunctions(self, spks, T, dt):
        
        
        T_i     = int(T/dt)
        
        bins_i  = Tools.timeToIndex(self.bins, dt)      
        spks_i  = Tools.timeToIndex(spks, dt)   
        nb_bins = self.getNbOfBasisFunctions()
        
        X = np.zeros( (T_i, nb_bins) )
        
        # Filter the spike train with the first rectangular function (for the other simply shift the solution        
        tmp = np.zeros( T_i + bins_i[-1] + 1)
            
        for s in spks_i :
            lb = s + bins_i[0]
            ub = s + bins_i[1]
            tmp[lb:ub] += 1
          
        tmp = tmp[:T_i]   

        # Fill the matrix by shifting the vector tmp
        for l in np.arange(nb_bins) :
            tmp_shifted = np.concatenate( ( np.zeros( int(bins_i[l]) ), tmp) )
            X[:,l] = tmp_shifted[:T_i]
                    
        return X
    
    
    def convolution_ContinuousSignal_basisfunctions(self, I, dt):
        
        T_i     = len(I)
        
        bins_i  = Tools.timeToIndex(self.bins, dt)   
        bins_l  = self.getNbOfBasisFunctions()
        
        X = np.zeros( (T_i, bins_l) )
        I_tmp = np.array(I,dtype='float64')        

        window = np.ones( bins_i[1] - bins_i[0])
        window = np.array(window,dtype='float64')  
    
        F_star_I = fftconvolve(window, I_tmp, mode='full')*dt
        F_star_I = np.array(F_star_I[:T_i], dtype='double')        
                
        for l in np.arange(bins_l) :
            
            F_star_I_shifted = np.concatenate( ( np.zeros( int(bins_i[l]) ), F_star_I) )
            X[:,l] = np.array(F_star_I_shifted[:T_i], dtype='double')
    
        return X
    



        
        
        