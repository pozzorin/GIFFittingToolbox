import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import fftconvolve
import Tools
from Filter import *


class Filter_Rect(Filter) :

    """
    Abstract class for filters defined as linear combinations of rectangular basis functions.
    A filter f(t) is defined in the form 
    
    f(t) = sum_j b_j*rect_j(t),
    
    where b_j is a set of coefficient and rect_j is a set of non-overlapping rectangular basis functions.
    
    This class is abstract because it does not specify the kind of rectangular basis functions used in practice.
    Possible implementations could be e.g. linear spacing, log spacing, arbitrary spacing.
    To implement such filters, inherit from Filter_Rect
    """

    __metaclass__  = abc.ABCMeta
    

    def __init__(self):
        
        Filter.__init__(self)
         
        # Auxiliary variables that can be computed using the parameters above        
                
        self.bins    = []     # ms, vector defining the rectangular basis functions for f(t)
        
        self.support = []     # ms, centers of bins used to define the filter
        
 
 
 
    ############################################################################
    # IMPLEMENT SOME OF THE ABSTRACT METHODS OF FILTER
    ############################################################################
 
    def getLength(self):
        
        """
        Return filter length (in ms).
        """
        
        return self.bins[-1]

     
    def setFilter_Function(self, f):
        
        """
        Given a function of time f(t), the bins of the filer are initialized accordingly.
        For example, if f(t) is an exponential function, the filter will approximate an exponential using rectangular basis functions.
        """
        
        self.computeBins() 
        self.filter_coeff = f(self.support)
 
              
    def computeInterpolatedFilter(self, dt) :
            
        """
        Given a particular dt, the function compute the interpolated filter as well as its temporal support vector.
        """
                
        self.computeBins()
        
        bins_i = Tools.timeToIndex(self.bins, dt)
        
        if self.filter_coeffNb == len(self.filter_coeff) :
        
            filter_interpol = np.zeros( (bins_i[-1] - bins_i[0])  )
            
            for i in range(len(self.filter_coeff)) :
                
                lb = int(bins_i[i])
                ub = int(bins_i[i+1])
                filter_interpol[lb:ub] = self.filter_coeff[i]
    
            filter_interpol_support = np.arange(len(filter_interpol))*dt
    
            self.filtersupport = filter_interpol_support
            self.filter = filter_interpol
        
        else :
            
            print "Error: value of the filter coefficients does not match the number of basis functions!"


    ###################################################################################
    # OTHER FUNCTIONS
    ###################################################################################

    def computeSupport(self):
        
        """
        Based on the rectangular basis functions defined in sefl.bins compute self.support
        (ie, the centers of rectangular basis functions).
        """
        
        self.support = np.array( [ (self.bins[i]+self.bins[i+1])/2 for i in range(len(self.bins)-1) ])


    @abc.abstractmethod    
    def computeBins(self) :
        
        """
        Given metaparametres compute bins associated to the rectangular basis functions.
        """

        
        
        