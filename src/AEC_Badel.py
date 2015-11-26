import numpy as np
import matplotlib.pyplot as plt
import copy

from AEC import *
from Experiment import *
from Filter_Rect_LinSpaced import *
from Filter_Rect_LogSpaced_AEC import *

from numpy.linalg import *
from random import sample
from time import time


class AEC_Badel(AEC) :
        
    """
    This class implements the Active Electrode Compensation method introduced in Badel et al. J. Neurophys. 2007.
    The mathematical details of the preprocessing technique implemented here can be found in Pozzorini et al. PLOS Comp. Biol. 2015
    Note that variable name are consistent with Pozzorini et al. PLOS Comp. Biol. 2015.
    """
      
    def __init__(self, dt):
        
        """
        Input parameters:
        dt: experimental time step in ms (i.e. 1/sampling frequency).
        """
        
        # Define variables for optimal linear filter K_opt   
        self.K_opt      = Filter_Rect_LogSpaced_AEC(length=150.0, binsize_lb=dt, binsize_ub=5.0, slope=10.0, clamp_period=0.5)
        self.K_opt_all  = []                # List of K_opt, used to store bootstrap repetitions                         

        # Define variables for electrode filter    
        self.K_e        = Filter_Rect_LinSpaced()
        self.K_e_all    = []                # List of K_e, used to store bootstrap repetitions 

        # Meta parameters used in AEC-Step 1 (compute optimal linear filter K_opt)
        self.p_nbRep       = 15             # nb of times the filtres are estimated by resampling from available data
        self.p_pctPoints   = 0.8            # between 0 and 1, fraction of datapoints in subthreshold recording used for each bootstrap repetition 
   
        # Meta parameters used in AEC - Step 2 (estimation of K_e given K_opt)
        self.p_Ke_l        = 7.0            # ms, length of the electrode filter K_e
        self.p_b0          = [10.0]         # MOhm/ms, initial condition for exponential fit on the tail of K_opt (amplitude)
        self.p_tau0        = [20.0]         # ms, initial condition for exponential fit on the tail of K_opt (time scale)  
        self.p_expFitRange = [3.0, 50.0]    # ms, range where to perform exponential fit on the tail of K_opt (first milliseconds)
              
        self.p_derivative_flag = False
        
        
    ##############################################################################################    
    # ABSTRACT METHODS FROM AEC THAT HAVE TO BE IMPLEMENTED
    ##############################################################################################
    
    def performAEC(self, experiment):

        print "\nPERFORM ACTIVE ELECTRODE COMPENSATION (Badel method)..."

        # Estimate electrode filter using the AEC traces of a given Experiment
        self.computeElectrodeFilter(experiment)

        # Compensate voltage traces in a given Experiment    
        self.compensateAllTraces(experiment)
        
        

    def computeElectrodeFilter(self, expr) :
    
        """
        Extract the optimal linter filter K_opt between the AEC input current I and the AEC recorded voltage V_rec (data are stored in Experiment).
        The regression is performed using the temporal derivative of the signals (see Badel et al 2008).

        This function implements AEC Step 2 and Step 3 described in Pozzorini et al.PLOS Comp. Biol. 2015:
        - Step 2: compute optimal linear filter K_opt
        - Step 3: compute electrode filter K_e
        Using the AEC data stored for a given Experiment expr.
        """
        
        print "\nEstimate electrode properties..."
        
        # set experimental sampling frequency
        dt = expr.dt       
        
        # estimate optimal linear filter on I_dot - V_dot
        if self.p_derivative_flag :
        
            # Compute temporal derivative of the signal
            V_dot = np.diff(expr.AEC_trace.V_rec)/dt
            I_dot = np.diff(expr.AEC_trace.I)/dt
            
        # estimate optimal linear filter on I - V
        else :
            
            # remove mean from signals (do not use derivative)                 
            V_dot = expr.AEC_trace.V_rec - np.mean(expr.AEC_trace.V_rec) 
            I_dot = expr.AEC_trace.I - np.mean(expr.AEC_trace.I) 
           
           
        # Get ROI indices for AEC data used to estimate AEC filters
        ROI_selection = expr.AEC_trace.getROI_cutInitialSegments(self.K_opt.getLength())
        ROI_selection = ROI_selection[:-1]
        ROI_selection_l = len(ROI_selection)


        # Perform linear regression described in Eq. 11-13 of Pozzorini et al. PLOS Comp. Biol. 2015
        # and estimate electrode filter based on Eq. 14.

        # Build full X matrix
        X = self.K_opt.convolution_ContinuousSignal_basisfunctions(I_dot, dt)
        nbPoints = int(self.p_pctPoints*ROI_selection_l)
        
        # Estimate electrode filter on multiple repetitions by bootstrapping      
        for rep in np.arange(self.p_nbRep) :
              
            ############################################
            # ESTIMATE OPTIMAL LINEAR FILETR K_opt
            ############################################
    
            # Resample npPoints datapoints from ROI and define X matrix and Y vector for bootstrap regression
            ROI_selection_sampled = sample(ROI_selection, nbPoints)
            Y = np.array(V_dot[ROI_selection_sampled])
            X_tmp = X[ROI_selection_sampled, :]    
                    
            # Compute optimal linear filter K_pot for bootstrap repetition rep   
            XTX = np.dot(np.transpose(X_tmp), X_tmp)
            XTX_inv = inv(XTX)
            XTY = np.dot(np.transpose(X_tmp), Y)
            K_opt_coeff = np.dot(XTX_inv, XTY)
            K_opt_coeff = K_opt_coeff.flatten()     # coefficent of basis functions defining K_opt on bootstrap repetition rep


            ############################################
            # ESTIMATE ELECTRODE FILETR K_e
            ############################################
            
            # Create K_opt filter obtained from single bootstrap repetition
            K_opt_tmp = copy.deepcopy(self.K_opt)
            K_opt_tmp.setFilter_Coefficients(K_opt_coeff)
            
            # Store bootstrap repetiton
            self.K_opt_all.append(K_opt_tmp)
            
            # Fit exponential function on tail of K_opt            
            (t,K_opt_tmp_interpol) = K_opt_tmp.getInterpolatedFilter(dt)
            (K_opt_tmp_expfit_t, K_opt_tmp_expfit) = K_opt_tmp.fitSumOfExponentials(len(self.p_b0), self.p_b0, self.p_tau0, ROI=self.p_expFitRange, dt=dt)

            # Compute electrode filter K_e for bootstrap repetition
            Ke_coeff_tmp = (K_opt_tmp_interpol - K_opt_tmp_expfit)[ : int(self.p_Ke_l/dt) ]        
            Ke_tmp = Filter_Rect_LinSpaced(length=self.p_Ke_l, nbBins=len(Ke_coeff_tmp))
            Ke_tmp.setFilter_Coefficients(Ke_coeff_tmp)         # note that this filter is not defined using basis function expantion
            
            # Fit exponential function on K_e to quantify electrode properties (these values are not used to compensate the recordings)
            (Ke_tmp_expfit_t, Ke_tmp_expfit) = Ke_tmp.fitSumOfExponentials(1, [60.0], [0.5], ROI=[0.0,7.0], dt=dt)

            # Store the bootstrap repetition
            self.K_e_all.append(Ke_tmp)
            print "Repetition ", (rep+1), " R_e (MOhm) = %0.2f, " % (Ke_tmp.computeIntegral(dt))

        # Compute final filter by averaging the filters obtained via bootstrap 
        self.K_opt = Filter.averageFilters(self.K_opt_all)
        self.K_e = Filter.averageFilters(self.K_e_all)   
        
        print "Done!"      


    ##############################################################################################    
    # FUCTIONS TO APPLY AEC TO ALL TRACES IN THE EXPERIMENT
    ##############################################################################################    
    def compensateAllTraces(self, expr) :
        
        """
        Apply AEC to all traces (i.e., AEC traces, traning set traces and test set traces) contained in Experiment expr.
        Traces are compensated according to Eq. 15 in Pozzorini et al. PLOS Comp. Biol. 2015
        """
        
        print "\nCompensate experiment"
        
        print "AEC trace..."
        self.deconvolveTrace(expr.AEC_trace)

        print "Training set..."        
        for tr in expr.trainingset_traces :
            self.deconvolveTrace(tr)
         
        print "Test set..."     
        for tr in expr.testset_traces :
            self.deconvolveTrace(tr)         
        
        print "Done!"
         
         
         
    def deconvolveTrace(self, trace):
        
        """
        Estimate membrane potential V from recorded signal V_rec according to Eq. 15 in Pozzorini et al. PLOS Comp. Biol. 2015
        and compute spiking timing by thresholding on V.
        """
        
        V_e = self.K_e.convolution_ContinuousSignal(trace.I, trace.dt)
        V_aec = trace.V_rec - V_e
        
        trace.V = V_aec
        trace.AEC_flag = True
        
        trace.detectSpikes()
   
    

    #####################################################################################
    # FUNCTIONS FOR PLOTTING
    #####################################################################################
    def plot(self):
           
        """
        Plot K_opt and K_e.
        """
        
        # Plot optimal linear filter K_opt
        Filter.plotAverageFilter(self.K_opt_all, 0.05, loglog=False, label_x='Time (ms)', label_y='Optimal linear filter (MOhm/ms)')
 
        # Plot optimal linear filter K_e        
        Filter.plotAverageFilter(self.K_e_all, 0.05, label_x='Time (ms)', label_y='Electrode filter (MOhm/ms)')
  
        plt.show()
  
  
    def plotKopt(self):
        
        """
        Plot K_opt.
        """
           
        # Plot optimal linear filter K_opt
        Filter.plotAverageFilter(self.K_opt_all, 0.05, loglog=False, label_x='Time (ms)', label_y='Optimal linear filter (MOhm/ms)')
 
        plt.show()
 
    def plotKe(self):
        
        """
        Plot K_e.
        """
           
        # Plot optimal linear filter K_e        
        Filter.plotAverageFilter(self.K_e_all, 0.05, label_x='Time (ms)', label_y='Electrode filter (MOhm/ms)', plot_expfit=False)
       
        plt.show()
