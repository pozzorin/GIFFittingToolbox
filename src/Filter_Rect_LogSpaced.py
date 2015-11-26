import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import fftconvolve
import Tools
from Filter import *


class Filter_Rect_LogSpaced(Filter) :

    """
    This class defines a function of time expanded using log-spaced rectangular basis functions.
    A filter f(t) is defined in the form f(t) = sum_j b_j*rect_j(t),
    where b_j is a set of coefficient and rect_j is a set of rectangular basis functions.
    The width of the rectangular basis functions increase exponentially (log-spacing).
    This class is used to define both the spike-triggered current eta(t) and the spike-triggered
    movement of the firing threshold gamma(t).
    """


    def __init__(self, length=1000.0, binsize_lb=2.0, binsize_ub=1000.0, slope=7.0):
        
        Filter.__init__(self)
        
        self.p_length     = length           # ms, filter length
        self.p_binsize_lb = binsize_lb       # ms, min size for bin
        self.p_binsize_ub = binsize_ub       # ms, max size for bin
        self.p_slope      = slope            # exponent for log-scale binning  
        
        
        # Auxiliary variables that can be computed using the parameters above        
                
        self.bins    = []                    # ms, vector defining the rectangular basis functions for f(t)
        self.support = []                    # ms, centers of bins used to define the filter
        self.bins_l  = 0                     # nb of bins used to define the filter 
        
        # Initialize        
        self.computeBins()                   # using meta parameters self.metaparam_subthreshold define bins and support.
        self.setFilter_toZero()              # initialize filter to 0
     
     
    #############################################################################
    # Set functions
    #############################################################################
    
    def setFilter_Function(self, f):
        
        """
        Given a function of time f(t), the bins of the filer are initialized accordingly.
        For example, if f(t) is an exponential function, the filter will approximate an exponential using rectangular basis functions
        """
        
        self.computeBins() 
        self.filter_coeff = f(self.support)


    def setFilter_Coefficients(self, coeff):
        
        """
        Set the coefficients of the filter (i.e. the values that define the magnitude of each rectangular function)
        """
        
        self.computeBins() 
        
        if len(coeff) == self.bins_l :
            self.filter_coeff = coeff
        else :
            print "Error, the number of coefficients do not match the number of basis functions!"
        

    def setFilter_toZero(self):
        
        """
        Set the coefficients of the filter to 0
        """
        
        self.computeBins() 
        self.filter_coeff = np.zeros(self.bins_l)   
       
       
    #############################################################################
    # Get functions
    #############################################################################
    
    def getInterpolatedFilter(self, dt) :
            
        """
        Given a particular dt, the function compute and return the support t and f(t).
        """

        self.computeBins() 
                
        bins_i = Tools.timeToIndex(self.bins, dt)
                
        if self.getNbOfBasisFunctions() == len(self.filter_coeff) :
        
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
        
        self.computeBins() 
        
        return int(self.bins_l)
        
        
    def getLength(self):
        
        return self.bins[-1]

        
    #############################################################################
    # Functions to compute convolutions
    #############################################################################

    def convolution_Spiketrain_basisfunctions(self, spks, T, dt):
        
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
    
    
    ########################################################################################
    # AUXILIARY METHODS USED BY THIS PARTICULAR IMPLEMENTATION OF FILTER
    ########################################################################################

    def computeBins(self) :
        
        """
        This function compute bins and support given the metaparameters.
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
        self.support = np.array( [ (self.bins[i]+self.bins[i+1])/2 for i in range(len(self.bins)-1) ])
        self.bins_l = len(self.bins)-1


    def setMetaParameters(self, length=1000.0, binsize_lb=2.0, binsize_ub=1000.0, slope=7.0):

        """
        Set the parameters defining the rectangular basis functions.
        Attention, each time meta parameters are changes, the value of the filer is reset to 0.
        """
        
        self.p_length     = length                  # ms, filter length
        self.p_binsize_lb = binsize_lb              # ms, min size for bin
        self.p_binsize_ub = binsize_ub              # ms, max size for bin
        self.p_slope      = slope                    # exponent for log-scale binning  
        
        self.computeBins()
        self.setFilter_toZero()
        
        
        