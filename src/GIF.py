import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

from scipy import weave
from numpy.linalg import inv

from ThresholdModel import *
from Filter_Rect_LogSpaced import *

from Tools import reprint


class GIF(ThresholdModel) :

    """
    Generalized Integrate and Fire model
    Spike are produced stochastically with firing intensity:
    
    lambda(t) = lambda0 * exp( (V(t)-V_T(t))/DV ),
    
    
    where the membrane potential dynamics is given by:
    
    C dV/dt = -gl(V-El) + I - sum_j eta(t-\hat t_j)
    
    
    and the firing threshold V_T is given by:
    
    V_T = Vt_star + sum_j gamma(t-\hat t_j)
    
    
    and \hat t_j denote the spike times.
    """

    def __init__(self, dt=0.1):
                   
        self.dt = dt                    # dt used in simulations (eta and gamma are interpolated according to this value)
  
        # Define model parameters
        
        self.gl      = 1.0/100.0        # nS, leak conductance
        self.C       = 20.0*self.gl     # nF, capacitance
        self.El      = -65.0            # mV, reversal potential
        
        self.Vr      = -50.0            # mV, voltage reset
        self.Tref    = 4.0              # ms, absolute refractory period
        
        self.Vt_star = -48.0            # mV, steady state voltage threshold VT*
        self.DV      = 0.5              # mV, threshold sharpness
        self.lambda0 = 1.0              # by default this parameter is always set to 1.0 Hz
        
        
        self.eta     = Filter_Rect_LogSpaced()    # nA, spike-triggered current (must be instance of class Filter)
        self.gamma   = Filter_Rect_LogSpaced()    # mV, spike-triggered movement of the firing threshold (must be instance of class Filter)
        
        
        # Varialbes relatd to fit
        
        self.avg_spike_shape = 0
        self.avg_spike_shape_support = 0
        
        
        # Initialize the spike-triggered current eta with an exponential function        
        
        def expfunction_eta(x):
            return 0.2*np.exp(-x/100.0)
        
        self.eta.setFilter_Function(expfunction_eta)


        # Initialize the spike-triggered current gamma with an exponential function        
        
        def expfunction_gamma(x):
            return 10.0*np.exp(-x/100.0)
        
        self.gamma.setFilter_Function(expfunction_gamma)        
        
              
            
    ########################################################################################################
    # SET DT FOR NUMERICAL SIMULATIONS (KERNELS ARE REINTERPOLATED EACH TIME DT IS CHANGED)
    ########################################################################################################    
    def setDt(self, dt):

        """
        Define the time step used for numerical simulations. The filters eta and gamma are interpolated accordingly.
        """
        
        self.dt = dt

    
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
           
                int eta_l        = int(p_eta_l);
                int gamma_l      = int(p_gamma_l);
                
                                                  
                float rand_max  = float(RAND_MAX); 
                float p_dontspike = 0.0 ;
                float lambda = 0.0 ;            
                float r = 0.0;
                
                                                
                for (int t=0; t<T_ind-1; t++) {
    
    
                    // INTEGRATE VOLTAGE
                    V[t+1] = V[t] + dt/C*( -gl*(V[t] - El) + I[t] - eta_sum[t] );
               
               
                    // COMPUTE PROBABILITY OF EMITTING ACTION POTENTIAL
                    lambda = lambda0*exp( (V[t+1]-Vt_star-gamma_sum[t])/DeltaV );
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
 
        vars = [ 'p_T','p_dt','p_gl','p_C','p_El','p_Vr','p_Tref','p_Vt_star','p_DV','p_lambda0','V','I','p_eta','p_eta_l','eta_sum','p_gamma','gamma_sum','p_gamma_l','spks' ]
        
        v = weave.inline(code, vars)

        time = np.arange(p_T)*self.dt
        
        eta_sum   = eta_sum[:p_T]     
        V_T = gamma_sum[:p_T] + p_Vt_star
     
        spks = (np.where(spks==1)[0])*self.dt
    
        return (time, V, eta_sum, V_T, spks)

        
    def simulateDeterministic_forceSpikes(self, I, V0, spks):
        
        """
        Simulate the subthresohld response of the GIF model to an input current I (nA) with time step dt.
        Adaptation currents are enforced at times specified in the list spks (in ms) given as an argument to the function.
        V0 indicate the initial condition V(0)=V0.
        The function returns:
        - time     : ms, support for V, eta_sum, V_T, spks
        - V        : mV, membrane potential
        - eta_sum  : nA, adaptation current
        """
 
        # Input parameters
        p_T          = len(I)
        p_dt         = self.dt
          
          
        # Model parameters
        p_gl        = self.gl
        p_C         = self.C 
        p_El        = self.El
        p_Vr        = self.Vr
        p_Tref      = self.Tref
        p_Tref_i    = int(self.Tref/self.dt)
    
    
        # Model kernel      
        (p_eta_support, p_eta) = self.eta.getInterpolatedFilter(self.dt)   
        p_eta       = p_eta.astype('double')
        p_eta_l     = len(p_eta)


        # Define arrays
        V        = np.array(np.zeros(p_T), dtype="double")
        I        = np.array(I, dtype="double")
        spks     = np.array(spks, dtype="double")                      
        spks_i   = Tools.timeToIndex(spks, self.dt)


        # Compute adaptation current (sum of eta triggered at spike times in spks) 
        eta_sum  = np.array(np.zeros(p_T + 1.1*p_eta_l + p_Tref_i), dtype="double")   
        
        for s in spks_i :
            eta_sum[s + 1 + p_Tref_i  : s + 1 + p_Tref_i + p_eta_l] += p_eta
        
        eta_sum  = eta_sum[:p_T]  
   
   
        # Set initial condition
        V[0] = V0
        
    
        code = """ 
                #include <math.h>
                
                int   T_ind      = int(p_T);                
                float dt         = float(p_dt); 
                
                float gl         = float(p_gl);
                float C          = float(p_C);
                float El         = float(p_El);
                float Vr         = float(p_Vr);
                int   Tref_ind   = int(float(p_Tref)/dt);


                int next_spike = spks_i[0] + Tref_ind;
                int spks_cnt = 0;
 
                                                                       
                for (int t=0; t<T_ind-1; t++) {
    
    
                    // INTEGRATE VOLTAGE
                    V[t+1] = V[t] + dt/C*( -gl*(V[t] - El) + I[t] - eta_sum[t] );
               
               
                    if ( t == next_spike ) {
                        spks_cnt = spks_cnt + 1;
                        next_spike = spks_i[spks_cnt] + Tref_ind;
                        V[t-1] = 0 ;                  
                        V[t] = Vr ;
                        t=t-1;           
                    }
               
                }
        
                """
 
        vars = [ 'p_T','p_dt','p_gl','p_C','p_El','p_Vr','p_Tref','V','I','eta_sum','spks_i' ]
        
        v = weave.inline(code, vars)

        time = np.arange(p_T)*self.dt
        eta_sum = eta_sum[:p_T]     

        return (time, V, eta_sum)
        
        
     
    def fit(self, experiment, DT_beforeSpike = 5.0):
        
        """
        Fit the GIF model on experimental data.
        The experimental data are stored in the object experiment.
        The parameter DT_beforeSpike (in ms) defines the region that is cut before each spike when fitting the subthreshold dynamics of the membrane potential.
        Only training set traces in experiment are used to perform the fit.
        """
        
        # Three step procedure used for parameters extraction 
        
        print "\n################################"
        print "# Fit GIF"
        print "################################\n"
        
        self.fitVoltageReset(experiment, self.Tref, do_plot=False)
        
        self.fitSubthresholdDynamics(experiment, DT_beforeSpike=DT_beforeSpike)
        
        self.fitStaticThreshold(experiment)

        self.fitThresholdDynamics(experiment)



    ########################################################################################################
    # FIT VOLTAGE RESET GIVEN ABSOLUTE REFRACOTORY PERIOD (step 1)
    ########################################################################################################
    def fitVoltageReset(self, experiment, Tref, do_plot=False):
        
        """
        Tref: ms, absolute refractory period. 
        The voltage reset is estimated by computing the spike-triggered average of the voltage.
        """
        
        print "Estimate voltage reset (Tref = %0.1f ms)..." % (Tref)
        
        # Fix absolute refractory period
        self.dt = experiment.dt
        self.Tref = Tref
        
        all_spike_average = []
        all_spike_nb = 0
        for tr in experiment.trainingset_traces :
            
            if tr.useTrace :
                if len(tr.spks) > 0 :
                    (support, spike_average, spike_nb) = tr.computeAverageSpikeShape()
                    all_spike_average.append(spike_average)
                    all_spike_nb += spike_nb

        spike_average = np.mean(all_spike_average, axis=0)
        
        # Estimate voltage reset
        Tref_ind = np.where(support >= self.Tref)[0][0]
        self.Vr = spike_average[Tref_ind]

        # Save average spike shape
        self.avg_spike_shape = spike_average
        self.avg_spike_shape_support = support
        
        if do_plot :
            plt.figure()
            plt.plot(support, spike_average, 'black')
            plt.plot([support[Tref_ind]], [self.Vr], '.', color='red')            
            plt.show()
        
        print "Done! Vr = %0.2f mV (computed on %d spikes)" % (self.Vr, all_spike_nb)
        


    ########################################################################################################
    # FUNCTIONS RELATED TO FIT OF SUBTHRESHOLD DYNAMICS (step 2)
    ########################################################################################################

    def fitSubthresholdDynamics(self, experiment, DT_beforeSpike=5.0):
                    
        print "\nGIF MODEL - Fit subthreshold dynamics..." 
            
        # Expand eta in basis functions
        self.dt = experiment.dt
        self.eta.computeBins()
        
        # Build X matrix and Y vector to perform linear regression (use all traces in training set)            
        X = []
        Y = []
    
        cnt = 0
        
        for tr in experiment.trainingset_traces :
        
            if tr.useTrace :
        
                cnt += 1
                reprint( "Compute X matrix for repetition %d" % (cnt) )          
                
                (X_tmp, Y_tmp) = self.fitSubthresholdDynamics_Build_Xmatrix_Yvector(tr, DT_beforeSpike=DT_beforeSpike)
     
                X.append(X_tmp)
                Y.append(Y_tmp)
    
    
        # Concatenate matrixes associated with different traces to perform a single multilinear regression
        if cnt == 1:
            X = X[0]
            Y = Y[0]
            
        elif cnt > 1:
            X = np.concatenate(X, axis=0)
            Y = np.concatenate(Y, axis=0)
        
        else :
            print "\nError, at least one training set trace should be selected to perform fit."
        
        
        # Linear Regression
        print "\nPerform linear regression..."
        XTX     = np.dot(np.transpose(X), X)
        XTX_inv = inv(XTX)
        XTY     = np.dot(np.transpose(X), Y)
        b       = np.dot(XTX_inv, XTY)
        b       = b.flatten()
   
   
        # Update and print model parameters  
        self.C  = 1./b[1]
        self.gl = -b[0]*self.C
        self.El = b[2]*self.C/self.gl
        self.eta.setFilter_Coefficients(-b[3:]*self.C)
    
        self.printParameters()   
        
        
        # Compute percentage of variance explained on dV/dt
        
        var_explained_dV = 1.0 - np.mean((Y - np.dot(X,b))**2)/np.var(Y)
        print "Percentage of variance explained (on dV/dt): %0.2f" % (var_explained_dV*100.0)

        
        # Compute percentage of variance explained on V
        
        SSE = 0     # sum of squared errors
        VAR = 0     # variance of data
        
        for tr in experiment.trainingset_traces :
        
            if tr.useTrace :

                # Simulate subthreshold dynamics 
                (time, V_est, eta_sum_est) = self.simulateDeterministic_forceSpikes(tr.I, tr.V[0], tr.getSpikeTimes())
                
                indices_tmp = tr.getROI_FarFromSpikes(0.0, self.Tref)
                
                SSE += sum((V_est[indices_tmp] - tr.V[indices_tmp])**2)
                VAR += len(indices_tmp)*np.var(tr.V[indices_tmp])
                
        var_explained_V = 1.0 - SSE / VAR
        
        print "Percentage of variance explained (on V): %0.2f" % (var_explained_V*100.0)
                    

    def fitSubthresholdDynamics_Build_Xmatrix_Yvector(self, trace, DT_beforeSpike=5.0):
                   
        # Length of the voltage trace       
        Tref_ind = int(self.Tref/trace.dt)
        
        # Select region where to perform linear regression
        selection = trace.getROI_FarFromSpikes(DT_beforeSpike, self.Tref)
        selection_l = len(selection)
        
        # Build X matrix for linear regression
        X = np.zeros( (selection_l, 3) )
        
        # Fill first two columns of X matrix        
        X[:,0] = trace.V[selection]
        X[:,1] = trace.I[selection]
        X[:,2] = np.ones(selection_l) 
        
             
        # Compute and fill the remaining columns associated with the spike-triggered current eta               
        X_eta = self.eta.convolution_Spiketrain_basisfunctions(trace.getSpikeTimes() + self.Tref, trace.T, trace.dt) 
        X = np.concatenate( (X, X_eta[selection,:]), axis=1 )


        # Build Y vector (voltage derivative)    
        
        # COULD BE A BETTER SOLUTION IN CASE OF EXPERIMENTAL DATA (NOT CLEAR WHY)
        #Y = np.array( np.concatenate( ([0], np.diff(trace.V)/trace.dt) ) )[selection]
        
        # CORRECT SOLUTION TO FIT ARTIFICIAL DATA
        Y = np.array( np.concatenate( (np.diff(trace.V)/trace.dt, [0]) ) )[selection]      

        return (X, Y)
        
        
        
    ########################################################################################################
    # FUNCTIONS RELATED TO FIT FIRING THRESHOLD PARAMETERS (step 3)
    ########################################################################################################
        
    
    def fitStaticThreshold(self, experiment):
        
        self.setDt(experiment.dt)
    
        # Start by fitting a constant firing threshold, the result is used as initial condition to fit dynamic threshold
        
        print "\nGIF MODEL - Fit static threshold...\n"
        
        # Define initial conditions (based on the average firing rate in the training set)
        
        nbSpikes = 0
        duration = 0
        
        for tr in experiment.trainingset_traces :
            
            if tr.useTrace :
                
                nbSpikes += tr.getSpikeNb_inROI()
                duration += tr.getTraceLength_inROI()
                
        mean_firingrate = 1000.0*nbSpikes/duration      
        
        self.lambda0 = 1.0
        self.DV = 50.0
        self.Vt_star = -np.log(mean_firingrate)*self.DV


        # Perform fit     
        beta0_staticThreshold = [1/self.DV, -self.Vt_star/self.DV] 
        beta_opt = self.maximizeLikelihood(experiment, beta0_staticThreshold, self.buildXmatrix_staticThreshold) 
            
        # Store result  
        self.DV      = 1.0/beta_opt[0]
        self.Vt_star = -beta_opt[1]*self.DV 
        self.gamma.setFilter_toZero()
        
        self.printParameters()

      
    def fitThresholdDynamics(self, experiment):
                        
        self.setDt(experiment.dt)
    
        # Fit a dynamic threshold using a initial condition the result obtained by fitting a static threshold
        
        print "\nGIF MODEL - Fit dynamic threshold...\n"
        
        # Perform fit        
        beta0_dynamicThreshold = np.concatenate( ( [1/self.DV], [-self.Vt_star/self.DV], self.gamma.getCoefficients()/self.DV))        
        beta_opt = self.maximizeLikelihood(experiment, beta0_dynamicThreshold, self.buildXmatrix_dynamicThreshold)
        
        # Store result
        self.DV      = 1.0/beta_opt[0]
        self.Vt_star = -beta_opt[1]*self.DV 
        self.gamma.setFilter_Coefficients(-beta_opt[2:]*self.DV)

        self.printParameters()
          
      
    def maximizeLikelihood(self, experiment, beta0, buildXmatrix, maxIter=10**3, stopCond=10**-6) :
    
        """
        Maximize likelihood. This function can be used to fit any model of the form lambda=exp(Xbeta).
        Here this function is used to fit both:
        - static threshold
        - dynamic threshold
        The difference between the two functions is in the size of beta0 and the returned beta, as well
        as the function buildXmatrix.
        """
        
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
                (X_tmp, X_spikes_tmp, sum_X_spikes_tmp, N_spikes, T) = buildXmatrix(tr, V_est) 
                    
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


        return beta
     
        
    def computeLikelihoodGradientHessian(self, beta, X, X_spikes, sum_X_spikes) : 
        
        # IMPORTANT: in general we assume that the lambda_0 = 1 Hz
        # The parameter lambda0 is redundant with Vt_star, so only one of those has to be fitted.
        # We genearlly fix lambda_0 adn fit Vt_star
              
        dt = self.dt/1000.0     # put dt in units of seconds (to be consistent with lambda_0)
        
        X_spikesbeta    = np.dot(X_spikes,beta)
        Xbeta           = np.dot(X,beta)
        expXbeta        = np.exp(Xbeta)

        # Compute likelihood (would be nice to improve this to get a normalized likelihood)
        L = sum(X_spikesbeta) - self.lambda0*dt*sum(expXbeta)
                                       
        # Compute gradient
        G = sum_X_spikes - self.lambda0*dt*np.dot(np.transpose(X), expXbeta)
        
        # Compute Hessian
        H = -self.lambda0*dt*np.dot(np.transpose(X)*expXbeta, X)
        
        return (L,G,H)


    def buildXmatrix_staticThreshold(self, tr, V_est) :

        """
        Use this function to fit a model in which the firing threshold dynamics is defined as:
        V_T(t) = Vt_star (i.e., no spike-triggered movement of the firing threshold)
        """        
        # Get indices be removing absolute refractory periods (-self.dt is to not include the time of spike)       
        selection = tr.getROI_FarFromSpikes(-self.dt, self.Tref )
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
        
        X_spikes = X[spks_i_afterselection,:]
                
        sum_X_spikes = np.sum( X_spikes, axis=0)
        
        return (X, X_spikes, sum_X_spikes, N_spikes, T_l)
        
            
    def buildXmatrix_dynamicThreshold(self, tr, V_est) :

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
  
        # Precompute other quantities
        X_spikes = X[spks_i_afterselection,:]
        sum_X_spikes = np.sum( X_spikes, axis=0)
        
                
        return (X, X_spikes, sum_X_spikes,  N_spikes, T_l)
 
 
        
    ########################################################################################################
    # PLOT AND PRINT FUNCTIONS
    ########################################################################################################     
  
        
    def plotParameters(self) :
        
        plt.figure(facecolor='white', figsize=(14,4))
            
        # Plot kappa
        plt.subplot(1,3,1)
        
        K_support = np.linspace(0,150.0, 300)             
        K = 1./self.C*np.exp(-K_support/(self.C/self.gl)) 
            
        plt.plot(K_support, K, color='red', lw=2)
        plt.plot([K_support[0], K_support[-1]], [0,0], ls=':', color='black', lw=2)
            
        plt.xlim([K_support[0], K_support[-1]])    
        plt.xlabel("Time (ms)")
        plt.ylabel("Membrane filter (MOhm/ms)")        
        
        # Plot eta
        plt.subplot(1,3,2)
        
        (eta_support, eta) = self.eta.getInterpolatedFilter(self.dt) 
        
        plt.plot(eta_support, eta, color='red', lw=2)
        plt.plot([eta_support[0], eta_support[-1]], [0,0], ls=':', color='black', lw=2)
            
        plt.xlim([eta_support[0], eta_support[-1]])    
        plt.xlabel("Time (ms)")
        plt.ylabel("Eta (nA)")
        

        # Plot gamma
        plt.subplot(1,3,3)
        
        (gamma_support, gamma) = self.gamma.getInterpolatedFilter(self.dt) 
        
        plt.plot(gamma_support, gamma, color='red', lw=2)
        plt.plot([gamma_support[0], gamma_support[-1]], [0,0], ls=':', color='black', lw=2)
            
        plt.xlim([gamma_support[0], gamma_support[-1]])    
        plt.xlabel("Time (ms)")
        plt.ylabel("Gamma (mV)")
        plt.subplots_adjust(left=0.05, bottom=0.15, right=0.95, top=0.92, wspace=0.35, hspace=0.25)

        plt.show()
      
      
    def printParameters(self):

        print "\n-------------------------"        
        print "GIF model parameters:"
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
        print "-------------------------\n"
                  

    @classmethod
    def compareModels(cls, GIFs, labels=None):

        """
        Given a list of GIF models, GIFs, the function produce a plot in which the model parameters are compared.
        """

        # PRINT PARAMETERS        

        print "\n#####################################"
        print "GIF model comparison"
        print "#####################################\n"
        
        cnt = 0
        for GIF in GIFs :
            
            print "Model: " + labels[cnt]          
            GIF.printParameters()
            cnt+=1

        print "#####################################\n"                
                
        # PLOT PARAMETERS
        plt.figure(facecolor='white', figsize=(9,8)) 
               
        colors = plt.cm.jet( np.linspace(0.7, 1.0, len(GIFs) ) )   
        
        # Membrane filter
        plt.subplot(2,2,1)
            
        cnt = 0
        for GIF in GIFs :
            
            K_support = np.linspace(0,150.0, 1500)             
            K = 1./GIF.C*np.exp(-K_support/(GIF.C/GIF.gl))     
            plt.plot(K_support, K, color=colors[cnt], lw=2)
            cnt += 1
            
        plt.plot([K_support[0], K_support[-1]], [0,0], ls=':', color='black', lw=2, zorder=-1)   
                
        plt.xlim([K_support[0], K_support[-1]])
        plt.xlabel('Time (ms)')
        plt.ylabel('Membrane filter (MOhm/ms)')  


        # Spike triggered current
        plt.subplot(2,2,2)
            
        cnt = 0
        for GIF in GIFs :
            
            if labels == None :
                label_tmp =""
            else :
                label_tmp = labels[cnt]
            
            (eta_support, eta) = GIF.eta.getInterpolatedFilter(0.1)         
            plt.plot(eta_support, eta, color=colors[cnt], lw=2, label=label_tmp)
            cnt += 1
            
        plt.plot([eta_support[0], eta_support[-1]], [0,0], ls=':', color='black', lw=2, zorder=-1)   
        
        if labels != None :
            plt.legend()       
            
        
        plt.xlim([eta_support[0], eta_support[-1]])
        plt.xlabel('Time (ms)')
        plt.ylabel('Eta (nA)')        
        

        # Escape rate
        plt.subplot(2,2,3)
            
        cnt = 0
        for GIF in GIFs :
            
            V_support = np.linspace(GIF.Vt_star-5*GIF.DV,GIF.Vt_star+10*GIF.DV, 1000) 
            escape_rate = GIF.lambda0*np.exp((V_support-GIF.Vt_star)/GIF.DV)                
            plt.plot(V_support, escape_rate, color=colors[cnt], lw=2)
            cnt += 1
          
        plt.ylim([0, 100])    
        plt.plot([V_support[0], V_support[-1]], [0,0], ls=':', color='black', lw=2, zorder=-1)   
    
        plt.xlim([V_support[0], V_support[-1]])
        plt.xlabel('Membrane potential (mV)')
        plt.ylabel('Escape rate (Hz)')  


        # Spike triggered threshold movememnt
        plt.subplot(2,2,4)
            
        cnt = 0
        for GIF in GIFs :
            
            (gamma_support, gamma) = GIF.gamma.getInterpolatedFilter(0.1)         
            plt.plot(gamma_support, gamma, color=colors[cnt], lw=2)
            cnt += 1
            
        plt.plot([gamma_support[0], gamma_support[-1]], [0,0], ls=':', color='black', lw=2, zorder=-1)   
      
        plt.xlim([gamma_support[0]+0.1, gamma_support[-1]])
        plt.xlabel('Time (ms)')
        plt.ylabel('Gamma (mV)')   

        plt.subplots_adjust(left=0.08, bottom=0.10, right=0.95, top=0.93, wspace=0.25, hspace=0.25)
        
        plt.show()
    

       
    @classmethod
    def plotAverageModel(cls, GIFs):

        """
        Average model parameters and plot summary data.
        """
                   
        #######################################################################################################
        # PLOT PARAMETERS
        #######################################################################################################        
        
        fig = plt.figure(facecolor='white', figsize=(16,7))  
        fig.subplots_adjust(left=0.07, bottom=0.08, right=0.95, top=0.90, wspace=0.35, hspace=0.5)   
        rcParams['xtick.direction'] = 'out'
        rcParams['ytick.direction'] = 'out'
       
       
        # MEMBRANE FILTER
        #######################################################################################################
        
        plt.subplot(2,4,1)
                    
        K_all = []
        
        for GIF in GIFs :
                      
            K_support = np.linspace(0,150.0, 300)             
            K = 1./GIF.C*np.exp(-K_support/(GIF.C/GIF.gl))     
            plt.plot(K_support, K, color='0.3', lw=1, zorder=5)
            
            K_all.append(K)

        K_mean = np.mean(K_all, axis=0)
        K_std  = np.std(K_all, axis=0)
        
        plt.fill_between(K_support, K_mean+K_std,y2=K_mean-K_std, color='gray', zorder=0)
        plt.plot(K_support, np.mean(K_all, axis=0), color='red', lw=2, zorder=10)  
        plt.plot([K_support[0], K_support[-1]], [0,0], ls=':', color='black', lw=2, zorder=-1)   
                
        Tools.removeAxis(plt.gca(), ['top', 'right'])
        plt.xlim([K_support[0], K_support[-1]])
        plt.xlabel('Time (ms)')
        plt.ylabel('Membrane filter (MOhm/ms)')  

       
        # SPIKE-TRIGGERED CURRENT
        #######################################################################################################
    
        plt.subplot(2,4,2)
                    
        K_all = []
        
        for GIF in GIFs :
                
            (K_support, K) = GIF.eta.getInterpolatedFilter(0.1)      
   
            plt.plot(K_support, K, color='0.3', lw=1, zorder=5)
            
            K_all.append(K)

        K_mean = np.mean(K_all, axis=0)
        K_std  = np.std(K_all, axis=0)
        
        plt.fill_between(K_support, K_mean+K_std,y2=K_mean-K_std, color='gray', zorder=0)
        plt.plot(K_support, np.mean(K_all, axis=0), color='red', lw=2, zorder=10)  
        plt.plot([K_support[0], K_support[-1]], [0,0], ls=':', color='black', lw=2, zorder=-1)   
                
        Tools.removeAxis(plt.gca(), ['top', 'right'])
        plt.xlim([K_support[0], K_support[-1]/10.0])
        plt.xlabel('Time (ms)')
        plt.ylabel('Spike-triggered current (nA)')  
 
 
        # SPIKE-TRIGGERED MOVEMENT OF THE FIRING THRESHOLD
        #######################################################################################################
    
        plt.subplot(2,4,3)
                    
        K_all = []
        
        for GIF in GIFs :
                
            (K_support, K) = GIF.gamma.getInterpolatedFilter(0.1)      
   
            plt.plot(K_support, K, color='0.3', lw=1, zorder=5)
            
            K_all.append(K)

        K_mean = np.mean(K_all, axis=0)
        K_std  = np.std(K_all, axis=0)
        
        plt.fill_between(K_support, K_mean+K_std,y2=K_mean-K_std, color='gray', zorder=0)
        plt.plot(K_support, np.mean(K_all, axis=0), color='red', lw=2, zorder=10)   
        plt.plot([K_support[0], K_support[-1]], [0,0], ls=':', color='black', lw=2, zorder=-1)   
                
        plt.xlim([K_support[0], K_support[-1]])
        Tools.removeAxis(plt.gca(), ['top', 'right'])
        plt.xlabel('Time (ms)')
        plt.ylabel('Spike-triggered threshold (mV)')  
 
      
         # R
        #######################################################################################################
    
        plt.subplot(4,6,12+1)
 
        p_all = []
        for GIF in GIFs :
                
            p = 1./GIF.gl
            p_all.append(p)
            
        plt.hist(p_all, histtype='bar', color='red', ec='white', lw=2)
        plt.xlabel('R (MOhm)')
        Tools.removeAxis(plt.gca(), ['top', 'left', 'right'])
        plt.yticks([])        
        
        
        # tau_m
        #######################################################################################################
    
        plt.subplot(4,6,18+1)
 
        p_all = []
        for GIF in GIFs :
                
            p = GIF.C/GIF.gl
            p_all.append(p)
            
        plt.hist(p_all, histtype='bar', color='red', ec='white', lw=2)
        plt.xlabel('tau_m (ms)')
        Tools.removeAxis(plt.gca(), ['top', 'left', 'right'])
        plt.yticks([])     
       
   
        # El
        #######################################################################################################
    
        plt.subplot(4,6,12+2)
 
        p_all = []
        for GIF in GIFs :
                
            p = GIF.El
            p_all.append(p)
            
        plt.hist(p_all, histtype='bar', color='red', ec='white', lw=2)
        plt.xlabel('El (mV)')
        Tools.removeAxis(plt.gca(), ['top', 'left', 'right'])
        plt.yticks([])     
       
          
        # V reset
        #######################################################################################################
    
        plt.subplot(4,6,18+2)
 
        p_all = []
        for GIF in GIFs :
                
            p = GIF.Vr
            p_all.append(p)
        
        print "Mean Vr (mV): %0.1f" % (np.mean(p_all))  
        
        plt.hist(p_all, histtype='bar', color='red', ec='white', lw=2)
        plt.xlabel('Vr (mV)')
        Tools.removeAxis(plt.gca(), ['top', 'left', 'right'])
        plt.yticks([])     
        
        
        # Vt*
        #######################################################################################################
    
        plt.subplot(4,6,12+3)
 
        p_all = []
        for GIF in GIFs :
                
            p = GIF.Vt_star
            p_all.append(p)
            
        plt.hist(p_all, histtype='bar', color='red', ec='white', lw=2)
        plt.xlabel('Vt_star (mV)')
        Tools.removeAxis(plt.gca(), ['top', 'left', 'right'])
        plt.yticks([])    
        
        # Vt*
        #######################################################################################################
    
        plt.subplot(4,6,18+3)
 
        p_all = []
        for GIF in GIFs :
                
            p = GIF.DV
            p_all.append(p)
            
        plt.hist(p_all, histtype='bar', color='red', ec='white', lw=2)
        plt.xlabel('DV (mV)')
        Tools.removeAxis(plt.gca(), ['top', 'left', 'right'])
        plt.yticks([])    
