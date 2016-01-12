import matplotlib.pyplot as plt
import numpy as np
import abc

import copy

import Tools

from scipy.signal import fftconvolve
from scipy.optimize import leastsq


class Filter :

    """
    Abstract class defining an interface for a linear filter defined as a linear combination of basis functions {f_j(t)}.
    
    Filters are useful to define e.g. a spike-triggered current, a spike-triggered threshold movement.
    
    A filter h(t) is defined in the form 
    
        h(t) = sum_j b_j*f_j(t), 
    
    where b_j is a set of coefficient and f_j(t) is a set of rectangular basis functions (see e.g. Eq. 16 in Pozzorini et al. PLOS Comp. Biol. 2015).
    
    Filters can be used to filter a continuous function of time or a spike-train.
    
    Depending on the circumstances different basis functions can be used (e.g., rectangular lin-spaced, rectangular log-spaced).
    Note that this class does provide an implementation of basis functions. To do that, inherit from Filter and implement the abstract methods.   
    """
    
    __metaclass__  = abc.ABCMeta
    

    def __init__(self):
        
        self.filter_coeff    = []              # Values of coefficients b_j which define the amplitude of each basis function f_j.
    
        self.filter_coeffNb  = 0               # Nb of basis functions used to define the filter
    
        self.filter          = 0               # array, interpolated filter
        
        self.filtersupport   = 0               # array, support of interpolated filter (ie, time vector)
        
        
        # Results of multiexponantial fit (these parameters are used to approximate the filter as a sum of exponentials)
        
        self.expfit_falg     = False           # True if the exponential fit has been performed
        
        self.b0              = []              # list, Amplitudes of exponential functions 
        
        self.tau0            = []              # list, Timescales of exponential functions
    


    #####################################################################
    # METHODS
    #####################################################################

    def setFilter_toZero(self):
       
        """
        Set all parameters b_j to zero.
        """    
        
        self.filter_coeff = np.zeros(self.filter_coeffNb)


    def setFilter_Coefficients(self, coeff):
        
        """
        Manually set the coefficients of the filter with coeff (i.e. the values that define the magnitude of each rectangular function).
        """
                
        if len(coeff) == self.filter_coeffNb :
            
            self.filter_coeff = coeff
        
        else :
            
            print "Error, the number of coefficients do not match the number of basis functions!"
       

    def getCoefficients(self) :
        
        """
        Return coefficients b_j that define the amplitude of each basis function.
        """

        return  self.filter_coeff                


    def getNbOfBasisFunctions(self) :
                
        return int(self.filter_coeffNb)


    def getInterpolatedFilter(self, dt) :
        
        """
        Compute and return the interpolated filter as well as its support.
        """
        
        self.computeInterpolatedFilter(dt)
        
        return (self.filtersupport, self.filter)


    def getInterpolatedFilter_expFit(self, dt) :
  
        """
        Return result of multiexponential fit to the interpolated filter.
        """
  
        if self.expfit_falg :
  
            t = np.arange(int(self.getLength()/dt))*dt
            F_exp = Filter.multiExpEval(t, self.b0, self.tau0)
        
            return (t, F_exp)

        else :        
            print "Exp filter has not been performed."
    
      
    def computeIntegral(self, dt):
        
        """
        Compute and return the integral of the interpolated filter.
        """
        
        (t, F) = self.getInterpolatedFilter(dt)   
        return sum(F)*dt   
      
            
    def convolution_ContinuousSignal(self, I, dt):
        
        """
        Compute the return the convolutional integral between the an I and the Filter.
        """
        
        (F_support, F) = self.getInterpolatedFilter(dt) 
    
        # Compute filtered input      
        I_tmp    = np.array(I,dtype='float64')
        F_star_I = fftconvolve(F, I_tmp, mode='full')*dt
        F_star_I = F_star_I[: int(len(I))]        
        
        F_star_I = F_star_I.astype("double")
        
        return F_star_I     


    def convolution_SpikeTrain(self, spks, T, dt):
        
        """
        Compute and return the convolutional integral between a spiking input spks of duration T and the Filter.
        spks  : in ms, spike times
        T     : in ms, duration of the experiment
        dt    : in ms, 
        """
        
        spks_i = Tools.timeToIndex(spks, dt)        
        (t,F) = self.getInterpolatedFilter(dt)
        F_length = len(F)
        
        filtered_spks = np.zeros(int(T/dt) + 5*F_length )
        
        for s in spks_i :
            filtered_spks[s : s + F_length] += F
        
        return filtered_spks[:int(T/dt)]


    def fitSumOfExponentials(self, dim, bs, taus, ROI=None, dt=0.1) :
        
        """
        Fit the interpolated filter self.filter with a sum of exponentails: F_fit(t) = sum_j^N b_j exp(-t/tau_j)
        dim : number N of exponentials
        bs  : list with initial conditions for amplitudes b_j
        taus: ms, list with initial conditions for timescales tau_j
        ROI :[lb, ub], in ms (consistent with units of dt). Specify lowerbound and upperbound (in time) where fit is perfomred.
        dt  : the filer is interpolated and fitted using discretization steps defined in dt.
        """

        (t, F) = self.getInterpolatedFilter(dt)
                
        
        if ROI == None :
            t_fit = t
            F_fit = F
        
        else :
            lb = int(ROI[0]/dt)
            ub = int(ROI[1]/dt)
            
            t_fit = t[ lb : ub ]
            F_fit = F[ lb : ub ]    
            
        p0 = np.concatenate((bs,taus))
        
        plsq = leastsq(Filter.multiExpResiduals, p0, args=(t_fit,F_fit,dim), maxfev=100000,ftol=0.00000001)
        
        p_opt = plsq[0]
        bs_opt = p_opt[:dim]
        taus_opt = p_opt[dim:]
        
        F_exp = Filter.multiExpEval(t, bs_opt, taus_opt)
        
        self.expfit_falg = True
        self.b0          = bs_opt 
        self.tau0        = taus_opt 
                
        return (t, F_exp)
        
      
    def plot(self, dt=0.05):
 
        """
        Plot filter.
        """
 
        (t, F) = self.getInterpolatedFilter(dt)
        
        plt.figure(figsize=(5,5), facecolor='white')
        plt.plot([t[0],t[-1]], [0.0,0.0], ':', color='black')
        plt.plot(t, F, 'black', label='Filter')
        
        
        if self.expfit_falg :
            F_fit = Filter.multiExpEval(t, self.b0, self.tau0)
            plt.plot(t, F_fit, 'red', label='Multiexp fit')
        
                
        plt.xlim([t[0],t[-1]])
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Filter")
        plt.show()        
        
    
    
    
    
    @abc.abstractmethod
    def getLength(self):
        
        """
        Return the duration (in ms) of the filter.
        """
      
         
    @abc.abstractmethod
    def setFilter_Function(self, f):
        
        """
        Initialize the filter coefficients b_j in such a way to approximate a function f provided as input parameter.
        This function should define self.filter_coeffNb.
        """    
    
        
    @abc.abstractmethod
    def computeInterpolatedFilter(self, dt):
        
        """
        Compute the interpolated filter self.filter h(t) = sum_j b_j*f_j(t) with a given time resolution dt as well as its support self.supportfilter (i.e., time vector)
        """        

             
    @abc.abstractmethod                
    def convolution_ContinuousSignal_basisfunctions(self, I, dt):

        """
        Return matrix containing the result of the convolutional integral between a continuous signla I and all basis functions that define the filter.
        """      
           
                  
    @abc.abstractmethod
    def convolution_Spiketrain_basisfunctions(self, spks, T, dt):
        
        """
        Return matrix containing the result of the convolutional integral between a spike train spks and all basis functions that define the filter.
        
        If S(t) is the spike train defined by the spike times in spks, the function should return
        a set of N arrays a1, ..., aN with:
        a_i = int_0^t f_i(s)S(t-s)ds
        """



    @classmethod
    def averageFilters(cls, Fs) :
        
        """
        Class method that compute and return the average filter of a list of filters Fs.
        Return an object of type Filter.   
        """
        
        F_avg = copy.deepcopy(Fs[0])
               
        F_coeff_all = []
        
        all_tau = []
        all_b = []
        
        for F in Fs :
            F_coeff_all.append(F.filter_coeff)
            
            if F.expfit_falg :
                all_tau.append(F.tau0)
                all_b.append(F.b0)
   
   
        F_avg.expfit_falg = False
        F_avg.b0 = []
        F_avg.taus = []
   
        F_avg.filter_coeff = np.mean(F_coeff_all, axis=0)
                
        # If individual filers have been fitted with exponential function, then average the parameters
        if F.expfit_falg :
            F_avg.b0 = np.mean(all_b, axis=0)          
            F_avg.taus = np.mean(all_tau, axis=0)                      
                                 
        return F_avg


    @classmethod
    def plotAverageFilter(cls, Fs, dt=0.05, loglog=False, label_x="Time (ms)", label_y="Filter", plot_expfit=True) :
          
        """
        Class method to average and plot a list of filters Fs.
        """
          
        plt.figure(figsize=(7,6), facecolor='white')
        
        F_interpol_all = []    
        F_exp_all = []

  
        for F in Fs :    
            (t, F_tmp) = F.getInterpolatedFilter(dt) 
            F_interpol_all.append(F_tmp)
            
            if F.expfit_falg :
                (t_exp, F_exp_tmp) = F.getInterpolatedFilter_expFit(dt) 
                F_exp_all.append(F_exp_tmp)
                
            if loglog :
                plt.loglog(t, F_tmp, 'gray')  
            else :
                plt.plot(t, F_tmp, 'gray')    


        F_interpol_mean = np.mean(F_interpol_all, axis=0)      
        
        if loglog :
            plt.loglog(t, F_interpol_mean, 'red', lw=2)   
        else :
            plt.plot(t, F_interpol_mean, 'red', lw=2)     
 
        
        if plot_expfit :
            
            if len(F_exp_all) > 0 :
                
                F_exp_mean = np.mean(F_exp_all, axis=0)  
                plt.plot(t_exp, F_exp_mean, 'black', lw=2, ls='--')         
        
        
        
        plt.plot([t[0],t[-1]], [0.0,0.0], ':', color='black')
        plt.xlim([t[0],t[-1]])
        plt.xlabel(label_x)
        plt.ylabel(label_y)        

 
 
    ######################################################################################
    # FUNCTIONS TO PERFORM EXPONENTIAL FIT
    ######################################################################################

   
    @classmethod
    def multiExpResiduals(cls, p, x, y, d):
        bs = p[0:d]
        taus = p[d:2*d]
            
        return (y - Filter.multiExpEval(x, bs, taus)) 
    
    
    @classmethod
    def multiExpEval(cls, x, bs, taus):
        
        result = np.zeros(len(x))
        L = len(bs)
            
        for i in range(L) :
            result = result + bs[i] *np.exp(-x/taus[i])
            
        return result
    

    
    
    

