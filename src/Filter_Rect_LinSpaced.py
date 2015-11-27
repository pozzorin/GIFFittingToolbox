import matplotlib.pyplot as plt
import numpy as np

from Filter import *
import Filter_Rect
from Filter_Rect import *
import Tools


class Filter_Rect_LinSpaced(Filter_Rect) :

    """
    This class defines a temporal filter defined as a linear combination of linearly-spaced rectangular basis functions.
    A filter f(t) is defined in the form 
    
    f(t) = sum_j b_j*rect_j(t),
    
    where b_j is a set of coefficient and rect_j is a set of linearly spaced rectangular basis functions,
    meaning that the width of all basis functions is the same.
    """

    def __init__(self, length=1000.0, nbBins=30) :
        
        Filter_Rect.__init__(self)
        
        # Metaparameters
        
        self.p_length       = length         # ms, filter length
        
        self.filter_coeffNb = nbBins         # integer, define the number of rectangular basis functions being used
           
        
        # Initialize   
             
        self.computeBins()                   # using meta parameters self.metaparam_subthreshold define bins and support.
        
        self.setFilter_toZero()              # initialize filter to 0
 


    def setMetaParameters(self, length=1000.0, nbBins=10):

        """
        Set the parameters defining the rectangular basis functions.
        Attention, each time meta parameters are changes, the value of the filer is reset to 0.
        """

        self.p_length = length         
               
        self.filter_coeffNb = nbBins
        
        self.computeBins()
                
        self.setFilter_toZero()

        
    ################################################################
    # IMPLEMENT ABSTRACT METHODS OF Filter_Rect
    ################################################################        

    def computeBins(self) :
        
        """
        This function compute self.bins and self.support given the metaparameters.
        """
                
        self.bins    = np.linspace(0.0, self.p_length, self.filter_coeffNb+1) 
        
        self.computeSupport()
        
        self.filter_coeffNb = len(self.bins)-1




    ################################################################
    # IMPLEMENT ABSTRACT METHODS OF Filter
    ################################################################

    def convolution_Spiketrain_basisfunctions(self, spks, T, dt):
        
        """
        Filter spike train spks with the set of rectangular basis functions defining the Filter.
        Since all the basis functions have the same width calculation can be made efficient by filter just ones and shifting.
        """
        
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
        
        """
        Filter continuous signal I with the set of rectangular basis functions defining the Filter.
        Since all the basis functions have the same width calculation can be made efficient by filter just ones and shifting.
        """
        
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
        