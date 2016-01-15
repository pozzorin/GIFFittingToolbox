import matplotlib.pyplot as plt
import numpy as np

from scipy import weave
from scipy.weave import converters
from numpy.linalg import inv

from SpikingModel import *
from iGIF import *

from Filter_Rect_LogSpaced import *

import Tools
from Tools import reprint



class iGIF_NP(iGIF) :

    """
    inactivating Generalized Integrate and Fire model 
    in which the nonlinear coupling between membrane potential and firing threshold
    is defined as a linear combination of rectangular basis functions.
    
    Mathematically the nonlinear function f(V) defining the coupling is given by:
    
    f(V) = sum_j b_j * g_j(V)
    
    where:
    
    b_j: are parameters
    g_j(V) : are rectangular functions of V
    """

    def __init__(self, dt=0.1):
        
        GIF.__init__(self, dt=dt)          
               
           
        # Initialize threshold-voltage coupling
        
        self.theta_tau  = 5.0                          # ms, timescale of threshold-voltage coupling
        
        self.theta_bins = np.linspace(-50, -10, 11)    # mV, nodes of rectangular basis functions g_j(V) used to define f(V)
        
        self.theta_i    = np.linspace(  0, 30.0, 10)   # mV, coefficients b_j associated with the rectangular basis functions above (these parameters define the functional shape of the threshodl-voltage coupling )
  
        
        self.fit_flag = False
        
        self.fit_all_tau_theta = 0                     # list containing all the tau_theta (i.e. the timescale of the threshold-voltage coupling) tested during the fit
        
        self.fit_all_likelihood = 0                    # list containing all the log-likelihoods obtained with different tau_theta
                                                       # (the optimal timescale tau_theta is the one that maximize the model likelihood)


                
                
    def getNonlinearCoupling(self):

        """
        Compute and return the nonlinear coupling f(V), as well as its support, according to the rectangular basis functions and its coefficients.
        """

        support = np.linspace(-100, 0.0, 1000)
        dV = support[1]-support[0]
        
        theta_inf = np.ones(len(support))*self.Vt_star
        
        for i in np.arange(len(self.theta_i)-1) :
            
            lb_i = np.where(support >= self.theta_bins[i])[0][0]
            ub_i = np.where(support >= self.theta_bins[i+1])[0][0]     
            
            theta_inf[lb_i:ub_i] = self.theta_i[i] + self.Vt_star
        
            
        theta_inf[ub_i:] = self.theta_i[-1] + self.Vt_star
           
        return (support, theta_inf)
       
        

    
    ########################################################################################################
    # FUNCTIONS FOR SIMULATIONS
    ########################################################################################################
    def simulateSpikingResponse(self, I, dt):
        
        """
        Simulate the spiking response of the GIF model to an input current I (nA) with time step dt (ms).
        Return a list of spike times (in ms).
        The initial conditions for the simulation is V(0)=El and VT = VT^* (i.e. the membrane is at rest and threshold is at baseline).
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
        V0 (mV) indicate the initial condition V(0)=V0.
        
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
        p_theta_tau = self.theta_tau
        p_theta_bins = self.theta_bins
        p_theta_bins = p_theta_bins.astype("double")
        p_theta_i    = self.theta_i       
        p_theta_i    = p_theta_i.astype("double")
              
                
        # Model kernels   
        (p_eta_support, p_eta) = self.eta.getInterpolatedFilter(self.dt)   
        p_eta       = p_eta.astype('double')
        p_eta_l     = len(p_eta)

        (p_gamma_support, p_gamma) = self.gamma.getInterpolatedFilter(self.dt)   
        p_gamma     = p_gamma.astype('double')
        p_gamma_l   = len(p_gamma)
      
        # Define arrays
        V = np.array(np.zeros(p_T), dtype="double")
        I = np.array(I, dtype="double")

        theta_trace = np.array(np.zeros(p_T), dtype="double")        
        R     = len(self.theta_bins)-1                 # subthreshold coupling theta
        theta = np.zeros((p_T,R))
        theta = theta.astype("double")

        
        spks = np.array(np.zeros(p_T), dtype="double")                      
        eta_sum = np.array(np.zeros(p_T + 2*p_eta_l), dtype="double")
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
                float theta_tau  = float(p_theta_tau);

                int eta_l        = int(p_eta_l);
                int gamma_l      = int(p_gamma_l);
                                            
                float rand_max  = float(RAND_MAX); 
                float p_dontspike = 0.0 ;
                float lambda = 0.0 ;            
                float rr = 0.0;

                float theta_taufactor = (1.0-dt/theta_tau);                 
                                                
                for (int t=0; t<T_ind-1; t++) {
    
    
                    // INTEGRATE VOLTAGE
                    V[t+1] = V[t] + dt/C*( -gl*(V[t] - El) + I[t] - eta_sum[t] );
               
               
                    // INTEGRATION THRESHOLD DYNAMICS                
                    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                    for (int r=0; r<R; r++) { 
                
                        theta[t+1,r] = theta_taufactor*theta[t,r];                           // everybody decay
                        
                        if ( V[t] >= p_theta_bins[r] && V[t] < p_theta_bins[r+1] ) {         // identify who integrates
                            theta[t+1,r] += dt/theta_tau;
                        }
                    }
                    
                    float theta_tot = 0.0;
                    for (int r=0; r<R; r++) { 
                        theta_tot += p_theta_i[r]*theta[t+1,r];
                    }                
                    
                    theta_trace[t+1] = theta_tot;
                    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
               
               
    
                    // COMPUTE PROBABILITY OF EMITTING ACTION POTENTIAL
                    lambda = lambda0*exp( (V[t+1]-Vt_star-gamma_sum[t+1]-theta_trace[t+1])/DeltaV );
                    p_dontspike = exp(-lambda*(dt/1000.0));                                  // since lambda0 is in Hz, dt must also be in Hz (this is why dt/1000.0)
                          
                          
                    // PRODUCE SPIKE STOCHASTICALLY
                    rr = rand()/rand_max;
                    if (rr > p_dontspike) {
                                        
                        if (t+1 < T_ind-1)                
                            spks[t+1] = 1.0; 
                        
                        t = t + Tref_ind;    
                        
                        if (t+1 < T_ind-1){ 
                            V[t+1] = Vr;
                            
                            for (int r=0; r<R; r++) 
                                theta[t+1,r] = 0.0;
                        }
                        
                        // UPDATE ADAPTATION PROCESSES     
                        for(int j=0; j<eta_l; j++) 
                            eta_sum[t+1+j] += p_eta[j]; 
                        
                        for(int j=0; j<gamma_l; j++) 
                            gamma_sum[t+1+j] += p_gamma[j] ;  
                        
                    }
               
                }
                
                """
 
        vars = [ 'theta_trace', 'theta', 'R', 'p_theta_tau', 'p_theta_bins', 'p_theta_i', 'p_T','p_dt','p_gl','p_C','p_El','p_Vr','p_Tref','p_Vt_star','p_DV','p_lambda0','V','I','p_eta','p_eta_l','eta_sum','p_gamma','gamma_sum','p_gamma_l','spks' ]
        
        v = weave.inline(code, vars)

        time      = np.arange(p_T)*self.dt
        eta_sum   = eta_sum[:p_T]     
        V_T       = gamma_sum[:p_T] + p_Vt_star + theta_trace[:p_T]
        spks      = (np.where(spks==1)[0])*self.dt
    
        return (time, V, eta_sum, V_T, spks)

               
     
    def fit(self, experiment, DT_beforeSpike = 5.0, theta_inf_nbbins=5, theta_tau_all=np.linspace(1.0, 10.0, 5), last_bin_constrained=False, do_plot=False):
        
        """
        Fit the iGIF_NP model on experimental data (details of the mehtod can be found in Mensi et al. 2016).
        The experimental data are stored in the object experiment (the fit is performed on the training set traces).
        
        Input parameters:
        
        - experiment       : object Experiment containing the experimental data to be fitted.
        
        - DT_beforeSpike   : ms, amount of data removed before each spike to perform the linear regression on the voltage derivative.
        
        - theta_inf_nbbins : integer, number of rectangular basis functions used to define the nonlinear coupling f(V).
                             The actual rectangular basis functions will be computed automatically based on the data (as explained in Mensi et al. 2016).
        
        - theta_tau_all    : list of float, timescales of the threshold-voltage coupling tau_theta tested during the fit (the one of those giving the max likelihood solution is reteined).
        
        - last_bin_constrained : {True, False}, set this to True in order to guarantee that the rectangular basis functions defining f(V) only starts above the voltage reset.
        
        - do_plot          : if True, a plot is made which shows the max likelihood as a function of the timescale tau_theta.
        
        The parameter DT_beforeSpike (in ms) defines the region that is cut before each spike when fitting the subthreshold dynamics of the membrane potential.
        Only training set traces in experiment are used to perform the fit.
        """
        
        # Three step procedure used for parameters extraction 
        
        print "\n################################"
        print "# Fit iGIF_NP"
        print "################################\n"
        
        self.fitVoltageReset(experiment, self.Tref, do_plot=False)
        
        self.fitSubthresholdDynamics(experiment, DT_beforeSpike=DT_beforeSpike)
        
        self.defineBinningForThetaInf(experiment, theta_inf_nbbins, last_bin_constrained=last_bin_constrained) 
        
        self.fitStaticThreshold(experiment)
        
        self.fitThresholdDynamics(experiment, theta_tau_all, do_plot=do_plot)

        self.fit_flag = True
  

        
        
    ########################################################################################################
    # FUNCTIONS RELATED TO FIT FIRING THRESHOLD PARAMETERS (step 3)
    ########################################################################################################
    def defineBinningForThetaInf(self, experiment, theta_inf_nbbins, last_bin_constrained=True) :
    
        """
        Simulate by forcing spikes, and based on voltage distribution, define binning to extract nonlinear coupling. 
        """
        
        # Precompute all the matrices used in the gradient ascent
        
        all_V_spikes = []        

        for tr in experiment.trainingset_traces:
            
            if tr.useTrace :              
                                
                # Simulate subthreshold dynamics 
                (time, V_est, eta_sum_est) = self.simulateDeterministic_forceSpikes(tr.I, tr.V[0], tr.getSpikeTimes())
                  
                all_V_spikes.append(V_est[tr.getSpikeIndices()])
 
           
        all_V_spikes = np.concatenate(all_V_spikes)
        
        V_min = np.min(all_V_spikes)
        V_max = np.max(all_V_spikes)
        
        # Do not allow to have a free bin at voltage reset (this should avoid a bad interaction between gamma and theta_inf)
        
        if last_bin_constrained :
            if V_min < self.Vr + 0.5 :
                V_min = self.Vr + 0.5
        

        print "\nDefine binning to extract theta_inf (V)..."
        print "Interval: %0.1f - %0.1f " % (V_min, V_max)
        
        self.theta_bins = np.linspace(V_min, V_max, theta_inf_nbbins+1)
        self.theta_bins[-1] += 100.0
        self.theta_i = np.zeros(theta_inf_nbbins)
        
        print "Bins (mV): ", self.theta_bins
        
        
    
    ########################################################################################################
    # FUNCTIONS TO FIT DYNAMIC THRESHLD
    ########################################################################################################
    
    def fitThresholdDynamics(self, experiment, theta_tau_all, do_plot=False):
                        
        self.setDt(experiment.dt)
        
        # Fit a dynamic threshold using a initial condition the result obtained by fitting a static threshold
        
        print "Fit dynamic threshold..."
        
        # Perform fit        
        beta0_dynamicThreshold = np.concatenate( ( [1/self.DV], [-self.Vt_star/self.DV], self.gamma.getCoefficients()/self.DV, self.theta_i))        
        (beta_opt, theta_tau_opt) = self.maximizeLikelihood_dynamicThreshold(experiment, beta0_dynamicThreshold, theta_tau_all, do_plot=do_plot)
        
        # Store result
        self.DV      = 1.0/beta_opt[0]
        self.Vt_star = -beta_opt[1]*self.DV 
        self.gamma.setFilter_Coefficients(-beta_opt[2:2+self.gamma.getNbOfBasisFunctions()]*self.DV)
        self.theta_i = -beta_opt[2+self.gamma.getNbOfBasisFunctions():]*self.DV
        self.theta_tau = theta_tau_opt
        
        self.printParameters()
        
      
      
         
    def maximizeLikelihood_dynamicThreshold(self, experiment, beta0, theta_tau_all, maxIter=10**3, stopCond=10**-6, do_plot=False) :
    
        beta_all = []
        L_all = []
    
        for theta_tau in theta_tau_all :
    
            print "\nTest tau_theta = %0.1f ms... \n" % (theta_tau)

            # Precompute all the matrices used in the gradient ascent
            
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
                    (X_tmp, X_spikes_tmp, sum_X_spikes_tmp, N_spikes, T) = self.buildXmatrix_dynamicThreshold(tr, V_est, theta_tau) 
                        
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
    
            L_all.append(L_norm)
            beta_all.append(beta)
        
        ind_opt = np.argmax(L_all)
        
        
        theta_tau_opt = theta_tau_all[ind_opt]
        beta_opt      = beta_all[ind_opt]
        L_norm_opt    = L_all[ind_opt]
        
        print "\n Optimal timescale: %0.2f ms" % (theta_tau_opt)
        print "Log-likelihood: %0.2f bit/spike" % (L_norm_opt)
            
        self.fit_all_tau_theta = theta_tau_all                     
        self.fit_all_likelihood = L_all                    
    
        
        if do_plot :
            
            plt.figure(figsize=(6,6), facecolor='white')
            plt.plot(theta_tau_all, L_all, '.-', color='black')
            plt.plot([theta_tau_opt], [L_norm_opt], '.', color='red')            
            plt.xlabel('Threshold coupling timescale (ms)')
            plt.ylabel('Log-likelihood (bit/spike)')
            plt.show()
    
    
        return (beta_opt, theta_tau_opt)
       
   
        
    def buildXmatrix_dynamicThreshold(self, tr, V_est, theta_tau) :

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
        X       = np.zeros((T_l_selection, 2))
        X[:,0]  = V_est[selection]
        X[:,1]  = np.ones(T_l_selection)
           
        # Compute and fill the remaining columns associated with the spike-triggered current gamma              
        X_gamma = self.gamma.convolution_Spiketrain_basisfunctions(tr.getSpikeTimes() + self.Tref, tr.T, tr.dt)
        X = np.concatenate( (X, X_gamma[selection,:]), axis=1 )
  
        # Fill columns related with nonlinera coupling
        X_theta = self.exponentialFiltering_ref(V_est, tr.getSpikeIndices(), theta_tau)
        X = np.concatenate( (X, X_theta[selection,:]), axis=1 )  
  
  
        # Precompute other quantities
        X_spikes = X[spks_i_afterselection,:]
        sum_X_spikes = np.sum( X_spikes, axis=0)
        
        return (X, X_spikes, sum_X_spikes,  N_spikes, T_l)


     
    def exponentialFiltering_ref(self, V, spks_ind, theta_tau) :
        
        """
        Auxiliary function used to compute the matrix Y used in maximum likelihood.
        This function compute a set of integrals:
        
        theta_i(t) = \int_0^T 1\tau_theta exp(-s/tau_theta) g_j{ V(t-s) }ds
        
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
        p_Tref      = self.Tref
        
        # Model parameters  definin threshold coupling      
        p_theta_tau = theta_tau
        p_theta_bins = self.theta_bins
        p_theta_bins = p_theta_bins.astype("double")

        # Define arrays
        V = np.array(V, dtype="double")

        R      = len(self.theta_bins)-1                 # subthreshold coupling theta
        theta  = np.zeros((p_T,R))
        theta  = theta.astype("double")

        spks   = np.array(spks_ind, dtype='double')
        p_spks_L = len(spks)
        
        
        code =  """
                #include <math.h>
                
                int   T_ind      = int(p_T);                
                float dt         = float(p_dt); 
                int   Tref_ind   = int(float(p_Tref)/dt);     
                float theta_tau  = float(p_theta_tau);

                float theta_taufactor = (1.0-dt/theta_tau);                 
                
                int spks_L     = int(p_spks_L);  
                int spks_cnt   = 0;
                int next_spike = int(spks(0));
                                             
                for (int t=0; t<T_ind-1; t++) {
    
    
                    // INTEGRATION THRESHOLD DYNAMICS      
                              
                    for (int r=0; r<R; r++) { 
                
                        theta(t+1,r) = theta_taufactor*theta(t,r);                           // everybody decay
                        
                        if ( V(t) >= p_theta_bins(r) && V(t) < p_theta_bins(r+1) ) {         // identify who integrates
                            theta(t+1,r) += dt/theta_tau;
                        }
                    }
       

                    // MANAGE RESET        
                    
                    if ( t+1 >= next_spike ) {                                        
                   
                        if(spks_cnt < spks_L) {
                            spks_cnt  += 1;
                            next_spike = int(spks(spks_cnt));
                        }
                        else {
                            next_spike = T_ind+1;
                        }
                        
                        
                        if ( t + Tref_ind < T_ind-1 ) { 
                            for (int r=0; r<R; r++)  
                                theta(t + Tref_ind ,r) = 0.0;                                // reset         
                        }   
                          
                        t = t + Tref_ind; 
                                 
                    }
                          
                }
                
                """
 
        vars = [ 'spks', 'p_spks_L', 'theta', 'R', 'p_theta_tau', 'p_theta_bins', 'p_T','p_dt','p_Tref','V' ]
        
        v = weave.inline(code, vars, type_converters=converters.blitz)
            
        return theta
     
         
    ########################################################################################################
    # PLOT AND PRINT FUNCTIONS
    ########################################################################################################     

    def printParameters(self):

        print "\n-------------------------"        
        print "iGIF_NP model parameters:"
        print "-------------------------"
        print "tau_m (ms):\t%0.3f"  % (self.C/self.gl)
        print "R (MOhm):\t%0.6f"    % (1.0/self.gl)
        print "C (nF):\t\t%0.3f"    % (self.C)
        print "gl (nS):\t%0.3f"     % (self.gl)
        print "El (mV):\t%0.3f"     % (self.El)
        print "Tref (ms):\t%0.3f"   % (self.Tref)
        print "Vr (mV):\t%0.3f"     % (self.Vr)     
        print "Vt* (mV):\t%0.3f"    % (self.Vt_star)    
        print "DV (mV):\t%0.3f"     % (self.DV)  
        print "tau_theta (ms):\t%0.3f"     % (self.theta_tau)        
        print "-------------------------\n"
                  
                      
    def plotParameters(self) :
        
        super(iGIF_NP, self).plotParameters()

        if self.fit_flag :
            
            plt.subplot(1,4,4)
            plt.plot(self.fit_all_tau_theta, self.fit_all_likelihood, '.-', color='black')
            plt.plot([self.theta_tau], [np.max(self.fit_all_likelihood)], '.', color='red')            
            plt.xlabel('Threshold coupling timescale (ms)')
            plt.ylabel('Max log-likelihood (bit/spike)')


        plt.subplots_adjust(left=0.07, bottom=0.2, right=0.98, top=0.90, wspace=0.35, hspace=0.10)
        
        plt.show()