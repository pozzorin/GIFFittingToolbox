import matplotlib.pyplot as plt
import numpy as np

from scipy import weave
from numpy.linalg import inv

from ThresholdModel import *
from Filter_Rect_LogSpaced import *

from GIF import *

from Tools import reprint


class gGIF(GIF) :

    """
    Generalized Integrate and Fire model defined in Pozzorini et al. PLOS Comp. Biol. 2015,
    but where eta is a spike-triggered conductance, rather than a current.
    
    Spike are produced stochastically with firing intensity:
    
    lambda(t) = lambda0 * exp( (V(t)-V_T(t))/DV ),
    
    where the membrane potential dynamics is given by:
    
    C dV/dt = -gl(V-El) + I - sum_j eta(t-\hat t_j)*(V - E_R),
    
    where E_R is the reversal potential associated with the spike-dependent adaptation.
    
    The firing threshold V_T is given by:
    
    V_T = Vt_star + sum_j gamma(t-\hat t_j),
    
    and \hat t_j denote the spike times.    
    """

    def __init__(self, dt=0.1):
          
        GIF.__init__(self, dt=dt)          
    
        self.Ek      = -80.0                  # mV, reversal potential associated with the spike-depedent adaptation.
          
                
        # Internal variables 
        
        self.Ek_all                  = 0      # all parameters Ek systematically tested during parameters extraction
        
        self.variance_explained_all  = 0      # variance explained (on dV/dt) for different values Ek_all
              
              
       
    def simulate(self, I, V0):
 
        """
        Simulate the spiking response of the gGIF model to an input current I (nA) with time step dt.
        
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
        p_Ek        = self.Ek
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
                float Ek         = float(p_Ek);
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
                    V[t+1] = V[t] + dt/C*( -gl*(V[t] - El) + I[t] - eta_sum[t]*(V[t]-Ek) );
               
               
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
 
        vars = [ 'p_T','p_dt','p_gl','p_C','p_Ek','p_El','p_Vr','p_Tref','p_Vt_star','p_DV','p_lambda0','V','I','p_eta','p_eta_l','eta_sum','p_gamma','gamma_sum','p_gamma_l','spks' ]
        
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
 
        print "simulate deterministic gGIF"
 
        # Input parameters
        p_T          = len(I)
        p_dt         = self.dt
          
          
        # Model parameters
        p_gl        = self.gl
        p_C         = self.C 
        p_El        = self.El
        p_Ek        = self.Ek
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
            eta_sum[s  + p_Tref_i  : s  + p_Tref_i + p_eta_l] += p_eta
        
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
                float Ek         = float(p_Ek);
                float Vr         = float(p_Vr);
                int   Tref_ind   = int(float(p_Tref)/dt);


                int next_spike = spks_i[0] + Tref_ind;
                int spks_cnt = 0;
 
                                                                       
                for (int t=0; t<T_ind-1; t++) {
    
    
                    // INTEGRATE VOLTAGE
                    V[t+1] = V[t] + dt/C*( -gl*(V[t] - El) + I[t] - eta_sum[t]*(V[t]-Ek) );
               
               
                    if ( t == next_spike ) {
                        spks_cnt = spks_cnt + 1;
                        next_spike = spks_i[spks_cnt] + Tref_ind;
                        V[t-1] = 0 ;                  
                        V[t] = Vr ;
                        t=t-1;           
                    }
               
                }
        
                """
 
        vars = [ 'p_T','p_dt','p_gl','p_C','p_El','p_Ek','p_Vr','p_Tref','V','I','eta_sum','spks_i' ]
        
        v = weave.inline(code, vars)

        time = np.arange(p_T)*self.dt
        eta_sum = eta_sum[:p_T]     

        return (time, V, eta_sum)
        
        
        
    def fit(self, experiment, Ek_all, DT_beforeSpike = 5.0, do_plot=False):
        
        """
        Fit the gGIF model on experimental data.
        
        The experimental data are stored in the object experiment.
        
        The parameter DT_beforeSpike (in ms) defines the region that is cut before each spike when fitting the subthreshold dynamics of the membrane potential.
        Only training set traces in experiment are used to perform the fit.
        
        Parameters controlling the voltage dynamics are extracted by iterating on the parameter Ek (as described in Mensi et al. PLOS Comp. Biol. 2016):
        for each Ek a linear regression is performed. The optimal Ek is defined as the Ek that minimizes the sum of squared error on the voltage derivative.
        
        Input parameters:
        
        - experiment     : instance of Experiment which contains the experimental data to be fitted
        - Ek_all         : mV, list of values tested for the reversal potential associated with the adaptation current
        - DT_beforeSpike : ms, amount of data discarded from the data when fitting the subthreshold dynamics of the membrane potential.
        - do_plot        : if True, plot the sum of squared error on the voltage derivative as a function of Ek
        """
        
        # Three step procedure used for parameters extraction 
        
        print "\n################################"
        print "# Fit gGIF"
        print "################################\n"
        
        self.fitVoltageReset(experiment, self.Tref, do_plot=False)
        
        self.fitSubthresholdDynamics(experiment, Ek_all, DT_beforeSpike=DT_beforeSpike, do_plot=do_plot)
        
        self.fitStaticThreshold(experiment)

        self.fitThresholdDynamics(experiment)


    ########################################################################################################
    # FUNCTIONS RELATED TO FIT OF SUBTHRESHOLD DYNAMICS (step 2)
    ########################################################################################################

    def fitSubthresholdDynamics(self, experiment, Ek_all, DT_beforeSpike=5.0, do_plot=False):
                    
        print "\ngGIF MODEL - Fit subthreshold dynamics..." 
           
        var_explained_dV_all = []   
        b_all = []
        
        
        for Ek in Ek_all :
        
            print "\nTest Ek = %0.2f mV..." % (Ek)
        
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
                    
                    (X_tmp, Y_tmp) = self.fitSubthresholdDynamics_Build_Xmatrix_Yvector(tr, Ek, DT_beforeSpike=DT_beforeSpike)
         
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
       
      
            # Compute percentage of variance explained on dV/dt
            var_explained_dV = 1.0 - np.mean((Y - np.dot(X,b))**2)/np.var(Y)
            print "Done! Percentage of variance explained (on dV/dt): %0.2f" % (var_explained_dV*100.0)
    
            # Save results    
            var_explained_dV_all.append(var_explained_dV)
            b_all.append(b)
   
        
        # Select best Ek
        self.Ek_all = Ek_all                  
        self.variance_explained_all = var_explained_dV_all
     
        ind_opt          = np.argmax(var_explained_dV_all)
        b                = b_all[ind_opt]
        Ek_opt           = Ek_all[ind_opt]
        var_explained_dV = var_explained_dV_all[ind_opt]        
        
        
        # Update and print model parameters
        self.C  = 1./b[1]
        self.gl = -b[0]*self.C
        self.El = b[2]*self.C/self.gl
        self.Ek = Ek_opt
        self.eta.setFilter_Coefficients(-b[3:]*self.C)

        self.printParameters()                    

        
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
        print "Percentage of variance explained (on dV/dt): %0.2f" % (var_explained_dV*100.0)
    

        if do_plot :
            
            plt.figure(figsize=(8,8), facecolor='white')
            
            plt.plot(self.Ek_all, self.variance_explained_all, '.-', color='black')
            plt.plot([Ek_opt],[var_explained_dV], '.', color='red')
            
            plt.xlabel('Ek (mV)')
            plt.ylabel('Pct. Variance explained on dV/dt (-)')
            
            
            

    def fitSubthresholdDynamics_Build_Xmatrix_Yvector(self, trace, Ek, DT_beforeSpike=5.0):
                   
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
        
        for i in np.arange( np.shape(X_eta)[1] ) :
            X_eta[:,i] = X_eta[:,i]*(trace.V-Ek)
        
        X = np.concatenate( (X, X_eta[selection,:]), axis=1 )


        # Build Y vector (voltage derivative)    

        Y = np.array( np.concatenate( (np.diff(trace.V)/trace.dt, [0]) ) )[selection]      

        return (X, Y)
        
        
    ##############################################################################################################
    # PRINT PARAMETRES        
    ##############################################################################################################
    
    def printParameters(self):

        print "\n-------------------------"        
        print "gGIF model parameters:"
        print "-------------------------"
        print "tau_m (ms):\t%0.3f"  % (self.C/self.gl)
        print "R (MOhm):\t%0.9f"    % (1.0/self.gl)
        print "C (nF):\t\t%0.3f"    % (self.C)
        print "gl (nS):\t%0.3f"     % (self.gl)
        print "El (mV):\t%0.3f"     % (self.El)
        print "Ek (mV):\t%0.3f"     % (self.Ek)
        print "Tref (ms):\t%0.3f"   % (self.Tref)
        print "Vr (mV):\t%0.3f"     % (self.Vr)     
        print "Vt* (mV):\t%0.3f"    % (self.Vt_star)    
        print "DV (mV):\t%0.3f"     % (self.DV)          
        print "-------------------------\n"