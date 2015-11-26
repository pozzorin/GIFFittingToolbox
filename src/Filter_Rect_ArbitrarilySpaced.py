import matplotlib.pyplot as plt
import numpy as np

from Filter import *
import Tools


class Filter_Rect_ArbitrarilySpaced(Filter) :

    """
    This class define a function of time expanded using a set of arbitrarily rectangular basis functions.
    A filter f(t) is defined in the form f(t) = sum_j b_j*rect_j(t),
    where b_j is a set of coefficient and rect_j is a set of rectangular basis functions.
    The width and size of each rectangular basis function can be arbitrarily fixed.
    """

    def __init__(self, bins=np.array([0.0,10.0,50.0,100.0,1000.0])) :
        
        Filter.__init__(self)
        
        self.p_length     = bins[-1]         # ms, filter length
        self.p_nbBins     = len(bins)        # integer, define the number of bins
        
        # Coefficients b_j that define the shape of the filter f(t)
        self.filter_coeff = np.zeros(1)   # values of bins
        
        # Auxiliary variables that can be computed using the parameters above               
        self.bins    = []                    # ms, vector defining the rectangular basis functions for f(t)
        self.support = []                    # ms, centers of bins used to define the filter 
        
        # Initialize        
        self.bins = bins 
        self.computeSupport()                    
        self.setFilter_toZero()              # initialize filter to 0
 
 
    #############################################################################
    # Set functions
    #############################################################################
    def setFilter_Function(self, f):
        
        """
        Given a function of time f(t), the bins of the filer are initialized accordingly.
        For example, if f(t) is an exponential function, the filter will approximate an exponential using rectangular basis functions
        """
        
        self.filter_coeff = f(self.support)


    def setFilter_Coefficients(self, coeff):
        
        """
        Set the coefficients of the filter (i.e. the values that define the magnitude of each rectangular function)
        """
                
        if len(coeff) == self.p_nbBins :
            self.filter_coeff = coeff
        else :
            print "Error, the number of coefficients do not match the number of basis functions!"
        
        
     #############################################################################
     # Get functions
     #############################################################################

    def getInterpolatedFilter(self, dt) :
            
        """
        Given a particular dt, the function compute and return the support t and f(t).
        """
                
        bins_i = Tools.timeToIndex(self.bins, dt)
        
        if self.p_nbBins == len(self.filter_coeff) :
        
            filter_interpol = np.zeros( (bins_i[-1] - bins_i[0])  )
            
            for i in range(len(self.filter_coeff)) :
                
                lb = int(bins_i[i])
                ub = int(bins_i[i+1])
                filter_interpol[lb:ub] = self.filter_coeff[i]
    
            filter_interpol_support = np.arange(len(filter_interpol))*dt
    
            return (filter_interpol_support, filter_interpol)
    
        else :
            
            print "Error: value of the filter coefficients does not match the number of basis functions!"



    def getNbOfBasisFunctions(self) :
        
        """
        Return the number of rectangular basis functions used to define the filter.
        """
                
        return int(self.p_nbBins)


    def getLength(self):
        
        return self.bins[-1]
    
        
    #############################################################################
    # Functions to compute convolutions
    #############################################################################

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
    

    ########################################################################################
    # AUXILIARY METHODS USED BY THIS PARTICULAR IMPLEMENTATION OF FILTER
    ########################################################################################

    def computeBins(self):
        
        pass

    def computeSupport(self):
        
        self.support = np.array( [ (self.bins[i]+self.bins[i+1])/2 for i in range(len(self.bins)-1) ])

        

    def setBasisFunctions(self, bins):

        """
        Set the parameters defining the rectangular basis functions.
        Attention, each time meta parameters are changes, the value of the filer is reset to 0.
        """

        self.p_length = bins[-1]                
        self.p_nbBins = len(bins)-1
        
        self.bins = np.array(bins)
        self.computeSupport()
        self.setFilter_toZero()
        
        

        
        
        