import matplotlib.pyplot as plt
import numpy as np

from scipy import weave
from numpy.linalg import inv

from SpikingModel import *
from iGIF import *
from Filter_Rect_LogSpaced import *

import Tools
from Tools import reprint


class iGIF_Na(iGIF) :

    """
    inactivating Generalized Integrate and Fire model iGIF_Na
    in which the nonlinear coupling between membrane potential and firing threshold
    is defined as in Platkiewicz J and Brette R, PLOS CB 2011.
    
    Mathematically the nonlinear function f(V) defining the coupling is given by:
    
    f(V) = ka * log( 1 + exp( (V-Vi)/ki) )
    
    where:
    
    k_a: defines the slope of sodium channel activation in the Hodgkin-Huxley model (i.e. the slope of m_\infty(V)).
    k_i: defines the slope of sodium channel inactivation in the Hodgkin-Huxley model (i.e. the absolute value of the slope of h_\infty(V)).
    V_i: half-voltage inactivation of sodium channel
    
    This equation can be derived analytically from the HH model assuming that fast sodium channel inactivation is 
    given by an inverted sigmoidal function of the membrane potential.
    
    For more details see Platkiewicz J and Brette R, PLOS CB 2011 or Mensi et al. PLOS CB 2016.
    """

    def __init__(self, dt=0.1):
    
        GIF.__init__(self, dt=dt)          
     
              
        # Initialize parametres for nonlinear coupling
        
        self.theta_tau = 5.0                   # ms, timescale of threshold-voltage coupling
        
        self.theta_ka  = 2.5                   # mV, slope of Na channel activation
        
        self.theta_ki  = 3.0                   # mV, absolute value of the slope of Na channel inactivation
        
        self.theta_Vi  = -55.0                 # mV, half-inactivation voltage for Na channels
         
                    
        # Store parameters related to parameters extraction
                           
        self.fit_all_ki = 0                     # mV, list containing all the values tested during the fit for ki
        
        self.fit_all_Vi = 0                     # mV, list containing all the values tested during the fit for Vi        
        
        self.fit_all_likelihood = 0             # 2D matrix containing all the log-likelihood obtained with different (ki, Vi)

          
            
    def getNonlinearCoupling(self):
        
        """
        Compute and return the nonlinear function f(V) defining the threshold-voltage coupling.
        The function is computed as:
        
        f(V) = ka * log( 1 + exp( (V-Vi)/ki) )
        """
    
        support   = np.linspace(-100,-20.0, 200)
        
        theta_inf = self.Vt_star + self.theta_ka*np.log( 1 + np.exp( (support - self.theta_Vi)/self.theta_ki ) )

        return (support, theta_inf)

    
    ########################################################################################################
    # FUNCTIONS FOR SIMULATIONS
    ########################################################################################################
    def simulateSpikingResponse(self, I, dt):
        
        """
        Simulate the spiking response of the GIF model to an input current I (nA) with time step dt.
        Return a list of spike times (in ms).
        The initial conditions for the simulation is V(0)=El.
        """
        
        self.setDt(dt)
    
        (time, V, eta_sum, V_T, spks_times) = self.simulate(I, self.El)
        
        return spks_times
   
   
    def simulateVoltageResponse(self, I, dt) :

        """
        Simulate the spiking response of the GIF model to an input current I (nA) with time step dt (ms).
        Return a list of spike times (in ms) as well as the dynamics of the subthreshold membrane potential V (mV) and the voltage threshold V_T (mV).
        The initial conditions for the simulation is V(0)=El and VT = VT^* (i.e. the membrane is at rest and threshold is at baseline).
        """

        self.setDt(dt)
    
        (time, V, eta_sum, V_T, spks_times) = self.simulate(I, self.El)
        
        return (spks_times, V, V_T) 
    
    
    
    def simulate(self, I, V0):
 
        """
        Simulate the spiking response of the GIF model to an input current I (nA) with time step dt.
        V0 indicate the initial condition V(0)=V0.
        The function returns:
        - time     : ms, support for V, eta_sum, V_T, spks
        - V        : mV, membrane potential
        - eta_sum  : nA, adaptation current
        - V_T      : mV, firing threshold
        - spks     : ms, list of spike times 
        """
 
        # Input parameters
        p_T         = len(I)
        p_dt        = self.dt
        
        # Model parameters
        p_gl        = self.gl
        p_C         = self.C 
        p_El        = self.El
        p_Vr        = self.Vr
        p_Tref      = self.Tref
        p_Vt_star   = self.Vt_star
        p_DV        = self.DV
        p_lambda0   = self.lambda0
        
        # Model parameters  definin threshold coupling      
        p_theta_ka  = self.theta_ka
        p_theta_ki  = self.theta_ki
        p_theta_Vi  = self.theta_Vi
        p_theta_tau = self.theta_tau
              
        
        # Model kernels   
        (p_eta_support, p_eta) = self.eta.getInterpolatedFilter(self.dt)   
        p_eta       = p_eta.astype('double')
        p_eta_l     = len(p_eta)

        (p_gamma_support, p_gamma) = self.gamma.getInterpolatedFilter(self.dt)   
        p_gamma     = p_gamma.astype('double')
        p_gamma_l   = len(p_gamma)
      
        # Define arrays
        V         = np.array(np.zeros(p_T), dtype="double")
        theta     = np.array(np.zeros(p_T), dtype="double")        
        I         = np.array(I, dtype="double")
        spks      = np.array(np.zeros(p_T), dtype="double")                      
        eta_sum   = np.array(np.zeros(p_T + 2*p_eta_l), dtype="double")
        gamma_sum = np.array(np.zeros(p_T + 2*p_gamma_l), dtype="double")            
 
        # Set initial condition
        V[0] = V0
         
        code =  """
                #include <math.h>
                
                int   T_ind      = int(p_T);                
                float dt         = float(p_dt); 
                
                float gl         = float(p_gl);
                float C          = float(p_C);
                float El         = float(p_El);
                float Vr         = float(p_Vr);
                int   Tref_ind   = int(float(p_Tref)/dt);
                float Vt_star    = float(p_Vt_star);
                float DeltaV     = float(p_DV);
                float lambda0    = float(p_lambda0);
           
                float theta_ka         = float(p_theta_ka);
                float theta_ki         = float(p_theta_ki);
                float theta_Vi         = float(p_theta_Vi);
                float theta_tau        = float(p_theta_tau);
              
                int eta_l        = int(p_eta_l);
                int gamma_l      = int(p_gamma_l);
                                      
                float rand_max  = float(RAND_MAX); 
                float p_dontspike = 0.0 ;
                float lambda = 0.0 ;            
                float r = 0.0;
                
                                                
                for (int t=0; t<T_ind-1; t++) {
    
    
                    // INTEGRATE VOLTAGE
                    V[t+1] = V[t] + dt/C*( -gl*(V[t] - El) + I[t] - eta_sum[t] );
                    
                    // INTEGRATE THETA                    
                    theta[t+1] = theta[t] + dt/theta_tau*(-theta[t] + theta_ka*log(1+exp((V[t]-theta_Vi)/theta_ki))); 
            
               
                    // COMPUTE PROBABILITY OF EMITTING ACTION POTENTIAL
                    lambda = lambda0*exp( (V[t+1]-Vt_star-gamma_sum[t]-theta[t+1])/DeltaV );
                    p_dontspike = exp(-lambda*(dt/1000.0));                                  // since lambda0 is in Hz, dt must also be in Hz (this is why dt/1000.0)
                          
                          
                    // PRODUCE SPIKE STOCHASTICALLY
                    r = rand()/rand_max;
                    if (r > p_dontspike) {
                                        
                        if (t+1 < T_ind-1)                
                            spks[t+1] = 1.0; 
                        
                        t = t + Tref_ind;    
                        
                        if (t+1 < T_ind-1) 
                            V[t+1] = Vr;
                        
                        
                        // UPDATE ADAPTATION PROCESSES     
                        for(int j=0; j<eta_l; j++) 
                            eta_sum[t+1+j] += p_eta[j]; 
                        
                        for(int j=0; j<gamma_l; j++) 
                            gamma_sum[t+1+j] += p_gamma[j] ;  
                        
                    }
               
                }
                
                """
 
        vars = [ 'theta', 'p_theta_ka', 'p_theta_ki', 'p_theta_Vi', 'p_theta_tau', 'p_T','p_dt','p_gl','p_C','p_El','p_Vr','p_Tref','p_Vt_star','p_DV','p_lambda0','V','I','p_eta','p_eta_l','eta_sum','p_gamma','gamma_sum','p_gamma_l','spks' ]
        
        v = weave.inline(code, vars)

        time = np.arange(p_T)*self.dt
        
        eta_sum   = eta_sum[:p_T]     
        V_T = gamma_sum[:p_T] + p_Vt_star + theta[:p_T]
     
        spks = (np.where(spks==1)[0])*self.dt
    
        return (time, V, eta_sum, V_T, spks)

        
         
         
    ######################################################################################################################
    # FUNCTIONS TO FIT DYNAMIC THRESHOLD BY BRUTE FORCE
    ######################################################################################################################
    
    def fit(self, experiment, theta_tau, ki_all, Vi_all, DT_beforeSpike=5.0, do_plot=False):
           
        """
        Fit the model to the training set data in experiment.
        
        Input parameters:
        
        - experiment     : an instance of the class Experiment containing the experimental data that will be used for the fit (only training set data will be used).
        - theta_tau      : ms, timescale of the threshold-voltage coupling (this parameter is not fitted but has to be known). To fit this parameter, fit first a GIF_NP model to the data.   
        - ki_all         : mV, array of values containing the parameters k_i (ie, Na channel inactivation slope) tested during the fit
        - Vi_all         : mV, array of values containing the parameters V_i (ie, Na channel half inactivation voltage) tested during the fit
        - DT_beforeSpike : ms, amount of time removed before each action potential (these data will not be considered when fitting the subthreshold membrane potential dynamics)
        - doPlot         : if True plot the max-likelihood as a function of ki and Vi.
        """
             
        print "\n################################"
        print "# Fit iGIF_Na"
        print "################################\n"
        
        # Three step procedure used for parameters extraction 
        
        self.fitVoltageReset(experiment, self.Tref, do_plot=False)
        
        self.fitSubthresholdDynamics(experiment, DT_beforeSpike=DT_beforeSpike)
        
        self.theta_tau = theta_tau
        
        self.fitStaticThreshold(experiment)
              
        self.fitThresholdDynamics_bruteforce(experiment, ki_all, Vi_all, do_plot=do_plot)
  
        #self.fit_bruteforce_flag = True
        #self.fit_binary_flag     = False
  
  
  
    def fitThresholdDynamics_bruteforce(self, experiment, ki_all, Vi_all, do_plot=False):
        
        # Fit a dynamic threshold using a initial condition the result obtained by fitting a static threshold

        print "Fit dynamic threshold..."
                
        #beta0_dynamicThreshold = np.concatenate( ( [1/self.DV], [-self.Vt_star/self.DV], [0], self.gamma.getCoefficients()/self.DV))        
        beta0_dynamicThreshold = np.concatenate( ( [1/self.DV], [-self.Vt_star/self.DV], [0], np.zeros(self.gamma.getNbOfBasisFunctions())))        
         
             
        all_L       = np.zeros((len(ki_all),len(Vi_all)))
        L_opt       = -10**20
        beta_opt    = 0 
        ki_opt      = 0    
        Vi_opt      = 0 
           
           
        for ki_i in np.arange(len(ki_all)) :
            
            for Vi_i in np.arange(len(Vi_all)) :
            
                ki = ki_all[ki_i]
                Vi = Vi_all[Vi_i]            
            
                print "\nTest parameters: ki = %0.2f mV, Vi = %0.2f mV" % (ki, Vi)        
        
                # Perform fit        
                (beta_tmp, L_tmp) = self.maximizeLikelihood_dynamicThreshold(experiment, ki, Vi, beta0_dynamicThreshold)
                
                all_L[ki_i, Vi_i] = L_tmp
        
                if L_tmp > L_opt :
                    
                    print "NEW OPTIMAL SOLUTION: LL = %0.5f (bit/spike)" % (L_tmp)
                    
                    L_opt    = L_tmp
                    beta_opt = beta_tmp 
                    Vi_opt   = Vi
                    ki_opt   = ki
        
        # Store result
        
        self.DV       = 1.0/beta_opt[0]
        self.Vt_star  = -beta_opt[1]*self.DV 
        self.theta_ka =  -beta_opt[2]*self.DV
        self.gamma.setFilter_Coefficients(-beta_opt[3:]*self.DV)
        self.theta_Vi = Vi_opt        
        self.theta_ki = ki_opt
        
        self.fit_all_ki = ki_all                    
        self.fit_all_Vi = Vi_all                   
        self.fit_all_likelihood = all_L   


        # Plot landscape
        
        if do_plot :
            
            (ki_plot,Vi_plot) = np.meshgrid(Vi_all, ki_all)

            print np.shape(ki_plot)
            print np.shape(Vi_plot)
            print np.shape(all_L)
            
            plt.figure(facecolor='white', figsize=(6,6))
            
            plt.pcolor(Vi_plot, ki_plot, all_L)
            plt.plot(ki_opt, Vi_opt, 'o', mfc='white', mec='black', ms=10)
            
            plt.xlabel('ki (mV)')
            plt.ylabel('Vi (mV)')
            
            plt.xlim([ki_all[0], ki_all[-1]])
            plt.ylim([Vi_all[0], Vi_all[-1]])            
            plt.show()
 

   
    def maximizeLikelihood_dynamicThreshold(self, experiment, ki, Vi, beta0, maxIter=10**3, stopCond=10**-6) :
        
        all_X        = []
        all_X_spikes = []
        all_sum_X_spikes = []
        
        T_tot = 0.0
        N_spikes_tot = 0.0
        
        traces_nb = 0
        
        for tr in experiment.trainingset_traces:
            
            if tr.useTrace :              
                
                traces_nb += 1
                
                # Simulate subthreshold dynamics 
                (time, V_est, eta_sum_est) = self.simulateDeterministic_forceSpikes(tr.I, tr.V[0], tr.getSpikeTimes())
                
                
                # Precomputes matrices to perform gradient ascent on log-likelihood
                (X_tmp, X_spikes_tmp, sum_X_spikes_tmp, N_spikes, T) = self.buildXmatrix_dynamicThreshold(tr, V_est, ki, Vi) 
                    
                T_tot        += T
                N_spikes_tot += N_spikes
                    
                all_X.append(X_tmp)
                all_X_spikes.append(X_spikes_tmp)
                all_sum_X_spikes.append(sum_X_spikes_tmp)
        
        logL_poisson = N_spikes_tot*(np.log(N_spikes_tot/T_tot)-1)
        
        
        # Perform gradient ascent
        
        print "Maximize log-likelihood (bit/spks)..."
                        
        beta = beta0
        old_L = 1
        
        for i in range(maxIter) :
          
            learning_rate = 1.0
            
            if i<=10 :                      # be careful in the first iterations (using a small learning rate in the first step makes the fit more stable)
                learning_rate = 0.1
            
            
            L=0; G=0; H=0;  
                
            for trace_i in np.arange(traces_nb):
                (L_tmp,G_tmp,H_tmp) = self.computeLikelihoodGradientHessian(beta, all_X[trace_i], all_X_spikes[trace_i], all_sum_X_spikes[trace_i])
                L+=L_tmp; G+=G_tmp; H+=H_tmp;
            
            
            beta = beta - learning_rate*np.dot(inv(H),G)
                
                
            if (i>0 and abs((L-old_L)/old_L) < stopCond) :              # If converged
                
                print "\nConverged after %d iterations!\n" % (i+1)
                break
            
            old_L = L
            
            # Compute normalized likelihood (for print)
            # The likelihood is normalized with respect to a poisson process and units are in bit/spks
            L_norm = (L-logL_poisson)/np.log(2)/N_spikes_tot
            reprint(L_norm)
            
        
        if (i==maxIter - 1) :                                           # If too many iterations
            print "\nNot converged after %d iterations.\n" % (maxIter)
        
        
        return (beta, L_norm)
              

        
    def buildXmatrix_dynamicThreshold(self, tr, V_est, ki, Vi) :

        """
        Use this function to fit a model in which the firing threshold dynamics is defined as:
        V_T(t) = Vt_star + sum_i gamma(t-\hat t_i) (i.e., model with spike-triggered movement of the threshold)
        """
           
        # Get indices be removing absolute refractory periods (-self.dt is to not include the time of spike)       
        selection = tr.getROI_FarFromSpikes(-tr.dt, self.Tref)
        T_l_selection  = len(selection)

            
        # Get spike indices in coordinates of selection   
        spk_train = tr.getSpikeTrain()
        spks_i_afterselection = np.where(spk_train[selection]==1)[0]


        # Compute average firing rate used in the fit   
        T_l = T_l_selection*tr.dt/1000.0                # Total duration of trace used for fit (in s)
        N_spikes = len(spks_i_afterselection)           # Nb of spikes in the trace used for fit
        
        
        # Define X matrix
        X       = np.zeros((T_l_selection, 3))
        X[:,0]  = V_est[selection]
        X[:,1]  = np.ones(T_l_selection)
        
        X_theta = self.exponentialFiltering_Brette_ref(V_est, tr.getSpikeIndices(), ki, Vi)
        X[:,2]  = X_theta[selection]  
      
           
        # Compute and fill the remaining columns associated with the spike-triggered current gamma              
        X_gamma = self.gamma.convolution_Spiketrain_basisfunctions(tr.getSpikeTimes() + self.Tref, tr.T, tr.dt)
        X = np.concatenate( (X, X_gamma[selection,:]), axis=1 )
  
        
        # Precompute other quantities
        X_spikes = X[spks_i_afterselection,:]
        sum_X_spikes = np.sum( X_spikes, axis=0)
        
        return (X, X_spikes, sum_X_spikes,  N_spikes, T_l)
 

    def exponentialFiltering_Brette_ref(self, V, spks_ind, ki, Vi):

        """
        Auxiliary function used to compute the matrix Y used in maximum likelihood.
        This function compute a set of integrals:
        
        theta_i(t) = \int_0^T 1\tau_theta exp(-s/tau_theta) f{ V(t-s) }ds
        
        wheref(V) = log( 1 + exp( (V-Vi)/ki) )
        
        After each spike in spks_ind theta_i(t) is reset to 0 mV and the integration restarts.
        
        The function returns a matrix where each line is given by theta_i(t).
        
        Input parameters:
        
        - V : numpy array containing the voltage trace (in mV)
        - spks_ind   : list of spike times in ms (used to reset)
        - theta_tau  : ms, timescale used in the intergration.
        
        """
        
        # Input parameters
        p_T         = len(V)
        p_dt        = self.dt
             
         
        # Model parameters  definin threshold coupling      
        p_theta_tau = self.theta_tau
        p_Tref      = self.Tref
        p_theta_ki  = ki
        p_theta_Vi  = Vi


        # Define arrays
        V         = np.array(V, dtype="double")
        theta     = np.array(np.zeros(p_T), dtype="double")        
        
        spks      = np.array(spks_ind, dtype='double')
        p_spks_L  = len(spks)
         
    
        code =  """
                #include <math.h>
                
                int   T_ind      = int(p_T);                
                float dt         = float(p_dt); 
                
                int   Tref_ind   = int(float(p_Tref)/dt);
                float theta_ki         = float(p_theta_ki);
                float theta_Vi         = float(p_theta_Vi);
                float theta_tau        = float(p_theta_tau);
              
                float theta_taufactor = (1.0-dt/theta_tau);                 
                
                int spks_L     = int(p_spks_L);  
                int spks_cnt   = 0;
                int next_spike = int(spks[0]);
         
                                                
                for (int t=0; t<T_ind-1; t++) {
                        
                    // INTEGRATE THETA 
                                       
                    theta[t+1] = theta[t] + dt/theta_tau*(-theta[t] + log(1+exp((V[t]-theta_Vi)/theta_ki))); 
            
             
                    // MANAGE RESET        
                    
                    if ( t+1 >= next_spike ) {                                        
                   
                        if(spks_cnt < spks_L) {
                            spks_cnt  += 1;
                            next_spike = int(spks[spks_cnt]);
                        }
                        else {
                            next_spike = T_ind+1;
                        }
                        
                        
                        if ( t + Tref_ind < T_ind-1 ) { 
                            theta[t + Tref_ind] = 0.0;                                      
                        }   
                          
                        t = t + Tref_ind; 
                                 
                    }  
                            
                }
                
                """
 
        vars = [ 'spks', 'p_spks_L', 'theta', 'p_theta_ki', 'p_theta_Vi', 'p_theta_tau', 'p_T','p_dt','p_Tref','V' ]
        
        v = weave.inline(code, vars)
        
        
        return theta

  
    ########################################################################################################
    # PLOT AND PRINT FUNCTIONS
    ########################################################################################################     
  
    def printParameters(self):

        print "\n-------------------------"        
        print "iGIF_Na model parameters:"
        print "-------------------------"
        print "tau_m (ms):\t%0.3f"  % (self.C/self.gl)
        print "R (MOhm):\t%0.3f"    % (1.0/self.gl)
        print "C (nF):\t\t%0.3f"    % (self.C)
        print "gl (nS):\t%0.3f"     % (self.gl)
        print "El (mV):\t%0.3f"     % (self.El)
        print "Tref (ms):\t%0.3f"   % (self.Tref)
        print "Vr (mV):\t%0.3f"     % (self.Vr)     
        print "Vt* (mV):\t%0.3f"    % (self.Vt_star)    
        print "DV (mV):\t%0.3f"     % (self.DV)  
        print "tau_theta (ms):\t%0.3f"   % (self.theta_tau)
        print "ka (mV):\t%0.3f"    % (self.theta_ka)     
        print "ki (mV):\t%0.3f"    % (self.theta_ki)    
        print "Vi (mV):\t%0.3f"    % (self.theta_Vi) 
        print "ka/ki (mV):\t%0.3f" % (self.theta_ka/self.theta_ki)                 
        print "-------------------------\n"
    
    
    def plotParameters(self) :
        
        super(iGIF_Na, self).plotParameters()

        plt.subplot(1,4,4)
                 
        (ki_plot,Vi_plot) = np.meshgrid(self.fit_all_Vi, self.fit_all_ki)

        plt.pcolor(Vi_plot, ki_plot, self.fit_all_likelihood)
        plt.plot(self.theta_ki, self.theta_Vi, 'o', mfc='white', mec='black', ms=10)
                    
        plt.xlim([self.fit_all_ki[0], self.fit_all_ki[-1]])
        plt.ylim([self.fit_all_Vi[0], self.fit_all_Vi[-1]])            
        plt.xlabel('ki (mV)')
        plt.ylabel('Vi (mV)') 
            
     
    @classmethod
    def plotAverageModel(cls, iGIFs):

        """
        Averae and plot the parameters of a list of iGIF_Na models.
        Input paramters:
        - iGIFs : list of iGFI objects
        """

        GIF.plotAverageModel(iGIFs)
        
        iGIF.plotAverageModel(iGIFs)

        # ki
        #######################################################################################################
        plt.subplot(4,6,12+5)
 
        p_all = []
        for myiGIF in iGIFs :
                
            p = myiGIF.theta_ka
            p_all.append(p)
            
        plt.hist(p_all, histtype='bar', color='red', ec='white', lw=2)
        plt.xlabel('ka (mV)')
        Tools.removeAxis(plt.gca(), ['top', 'left', 'right'])
        plt.yticks([])     

        # ki
        #######################################################################################################
        plt.subplot(4,6,18+5)
 
        p_all = []
        for myiGIF in iGIFs :
                
            p = myiGIF.theta_ki
            p_all.append(p)
            
        plt.hist(p_all, histtype='bar', color='red', ec='white', lw=2)
        plt.xlabel('ki (mV)')
        Tools.removeAxis(plt.gca(), ['top', 'left', 'right'])
        plt.yticks([])     

        # Vi
        #######################################################################################################
        plt.subplot(4,6,12+6)
 
        p_all = []
        for myiGIF in iGIFs :
                
            p = myiGIF.theta_Vi
            p_all.append(p)
            
        plt.hist(p_all, histtype='bar', color='red', ec='white', lw=2)
        plt.xlabel('Vi (mV)')
        Tools.removeAxis(plt.gca(), ['top', 'left', 'right'])
        plt.yticks([])     
        