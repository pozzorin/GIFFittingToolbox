import sys
sys.path.append('../')
sys.path.append('../../')

import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

from scipy import weave
from numpy.linalg import inv

from Tools import reprint
from numpy import nan, NaN

import math
from GIF import *


class GIF_HT(GIF) :

    """
    GIF model extended to implement features required for high-throughput fitting of HBP cells.
    """

    def __init__(self, dt, depol, gid):

        super(GIF_HT, self).__init__(dt)


        # Information about cell fitted
        ################################################

        self.gid = gid                            # neuron id
        self.depol = depol                        # depolarization levels on which the data were fitted
        self.problems_dic = {
            "1": "Likelihood: Nan encountered during gradient ascent.",
            "2": "Likelihood: Gradient ascent did not converge.",
            "3": "Likelihood: Gradient ascent interrupted due to numerical instability in gradient ascent.",
            "4": "Subthreshold fit: Significant drop due to exp based spike-triggered current.",
            "5": "Threshold fit: could not fit static threshold. Model cannot be used.",
            "6": "Threshold fit: could not fit dynamic threshold. Model has a static threshold.",
            "7": "Data: not enough spikes in training dataset. Could not fit the model.",
            "8": "Likelihood: singular matrix.",
            "9": "Could not fit static threshold." }

        # Fit problems and computing time
        #####################################

        self.fit_problem = False                  # True, if a problem was encountered during the fit, False if the fit is trustable
        self.fit_problem_which = []               # List of integers representing the problems encountered in the fit

        self.cputime = 0                          # s, cpu time required for the fit


        # Fit performance
        #####################################

        self.L_norm_train = 0                     # bits/spks, model likelihood assessed on training set
        self.L_norm_test = 0                      # bits/spks, model likelihood assessed on test set

        self.V_varexp_train = 0.0                 # percentage of variance explained when the model is fitted using exp based eta, on training set
        self.V_varexp_train_change = 0.0          # change in percentage of variance explained when the model is fitted using rect based eta, rather than exp basd

        self.V_varexp_test = 0.0                  # percentage of variance explained when the model is fitted using exp based eta, on training set


        # Model parameters (used internallly)
        #####################################

        self.beta_opt = 0                         # optimal threshold parametes stored in a conveinent way (useful to evaluate likelihood on test set)




    def computePerfDropDueToExp(self, GIF_HT_rect):

        self.V_varexp_train_change = self.V_varexp_train - GIF_HT_rect.V_varexp_train

        if self.V_varexp_train_change  <= -10.0 :

            print

            self.fit_problem = True
            self.fit_problem_which.append("4")




    def simulateDeterministic(self, trace):

        (time, V, eta_sum) = self.simulateDeterministic_forceSpikes(trace.I, trace.V[0], trace.spks*self.dt)

        return V



    def fit(self, experiment, DT_beforeSpike = 5.0):

        """
        Overwrite function of GIF in order to handle problems during the fit.
        """

        # Three step procedure used for parameters extraction

        print "\n################################"
        print "# Fit GIF"
        print "################################\n"

        self.fitVoltageReset(experiment, self.Tref, do_plot=False)

        self.fitSubthresholdDynamics(experiment, DT_beforeSpike=DT_beforeSpike)

        problem = self.fitStaticThreshold(experiment)

        if problem :

            print "Problem: could not fit static threshold."
            self.fit_problem = True
            self.fit_problem_which.append("9")
            self.gamma.setFilter_toZero()
            self.beta_opt = np.concatenate( ( [1/self.DV], [-self.Vt_star/self.DV], self.gamma.getCoefficients()/self.DV))

        else :

            self.fitThresholdDynamics(experiment)


    def fitVoltageReset(self, experiment, Tref, do_plot=False):

        """
        Overwrite function of GIF class to handle numerical problems.
        """

        print "Estimate voltage reset (Tref = %0.1f ms)..." % (Tref)

        # Fix absolute refractory period
        self.dt = experiment.dt
        self.Tref = Tref

        all_spike_average = []
        all_spike_nb = 0

        for tr in experiment.trainingset_traces :

            if tr.useTrace :
                if tr.getSpikeNbInROI() > 0 :
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


    def fitSubthresholdDynamics(self, experiment, DT_beforeSpike=5.0):

        """
        Overvrite function from GIF to deal with numerical instabiliteis and store quality of fit
        """

        print "\nGIF MODEL - Fit subthreshold dynamics..."

        # Expand eta in basis functions
        self.dt = experiment.dt


        # Build X matrix and Y vector to perform linear regression (use all traces in training set)
        # For each training set an X matrix and a Y vector is built.
        ####################################################################################################
        X = []
        Y = []

        cnt = 0

        for tr in experiment.trainingset_traces :

            if tr.useTrace :

                cnt += 1
                print "Compute X matrix for repetition %d" % (cnt)

                # Compute the the X matrix and Y=\dot_V_data vector used to perform the multilinear linear regression (see Eq. 17.18 in Pozzorini et al. PLOS Comp. Biol. 2015)
                (X_tmp, Y_tmp) = self.fitSubthresholdDynamics_Build_Xmatrix_Yvector(tr, DT_beforeSpike=DT_beforeSpike)

                X.append(X_tmp)
                Y.append(Y_tmp)



        # Concatenate matrixes associated with different traces to perform a single multilinear regression
        ####################################################################################################
        if cnt == 1:
            X = X[0]
            Y = Y[0]

        elif cnt > 1:
            X = np.concatenate(X, axis=0)
            Y = np.concatenate(Y, axis=0)

        else :
            print "\nError, at least one training set trace should be selected to perform fit."


        # Perform linear Regression defined in Eq. 17 of Pozzorini et al. PLOS Comp. Biol. 2015
        ####################################################################################################

        print "\nPerform linear regression..."
        XTX     = np.dot(np.transpose(X), X)
        XTX_inv = inv(XTX)
        XTY     = np.dot(np.transpose(X), Y)
        b       = np.dot(XTX_inv, XTY)
        b       = b.flatten()


        # Extract explicit model parameters from regression result b
        ####################################################################################################

        self.C  = 1./b[1]
        self.gl = -b[0]*self.C
        self.El = b[2]*self.C/self.gl
        self.eta.setFilter_Coefficients(-b[3:]*self.C)


        # Compute percentage of variance explained on dV/dt (training set data)
        ####################################################################################################

        var_explained_dV = 1.0 - np.mean((Y - np.dot(X,b))**2)/np.var(Y)
        print "Percentage of variance explained (on dV/dt): %0.2f" % (var_explained_dV*100.0)


        # Compute percentage of variance explained on V (see Eq. 26 in Pozzorini et al. PLOS Comp. Biol. 2105)
        ####################################################################################################

        SSE = 0     # sum of squared errors
        VAR = 0     # variance of data

        for tr in experiment.trainingset_traces :

            if tr.useTrace :

                # Simulate subthreshold dynamics
                (time, V_est, eta_sum_est) = self.simulateDeterministic_forceSpikes(tr.I, tr.V[0], tr.getSpikeTimes())

                indices_tmp = tr.getROI_FarFromSpikes(0.0, self.Tref)

                SSE += sum((V_est[indices_tmp] - tr.V[indices_tmp])**2)
                VAR += len(indices_tmp)*np.var(tr.V[indices_tmp])

        self.V_varexp_train = (1.0 - SSE / VAR)*100.0

        print "Percentage of variance explained (on V): %0.2f" % (self.V_varexp_train)



    def fitStaticThreshold(self, experiment):

        """
        Overwrite fitStaticThreshold of GIF to handle problems.
        """

        print "\nGIF MODEL - Fit static threshold...\n"


        self.setDt(experiment.dt)


        # Define initial conditions (based on the average firing rate in the training set)
        ###############################################################################################

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


        # Perform maximum likelihood fit (Newton method)
        ###############################################################################################

        beta0_staticThreshold = [1/self.DV, -self.Vt_star/self.DV]
        (beta_opt, problem_flag) = self.maximizeLikelihood(experiment, beta0_staticThreshold, self.buildXmatrix_staticThreshold)


        # Store result of constnat threshold fitting
        ###############################################################################################

        if problem_flag :

            print "Problem: could not fit static threshold."
            self.fit_problem = True
            self.fit_problem_which.append("5")


        else :

            self.DV      = 1.0/beta_opt[0]
            self.Vt_star = -beta_opt[1]*self.DV
            self.gamma.setFilter_toZero()

        return problem_flag

    def fitThresholdDynamics(self, experiment):

        """
        Overwrite fitDynamicThreshold of GIF to handle problems.
        """

        print "\nGIF MODEL - Fit dynamic threshold...\n"


        self.setDt(experiment.dt)


        # Perform maximum likelihood fit (Newton method)
        ###############################################################################################

        # Define initial conditions

        beta0_dynamicThreshold = np.concatenate( ( [1/self.DV], [-self.Vt_star/self.DV], self.gamma.getCoefficients()/self.DV))
        (beta_opt, problem_flag) = self.maximizeLikelihood(experiment, beta0_dynamicThreshold, self.buildXmatrix_dynamicThreshold)


        # Store result
        ###############################################################################################

        if problem_flag :

            print "Problem: could not fit dynamic threshold. Saved parameters of static threshold."
            self.fit_problem = True
            self.fit_problem_which.append("6")
            self.gamma.setFilter_toZero()
            self.beta_opt = np.concatenate( ( [1/self.DV], [-self.Vt_star/self.DV], self.gamma.getCoefficients()/self.DV))

        else :

            self.DV      = 1.0/beta_opt[0]
            self.Vt_star = -beta_opt[1]*self.DV
            self.gamma.setFilter_Coefficients(-beta_opt[2:]*self.DV)

        self.printParameters()



    def maximizeLikelihood(self, experiment, beta0, buildXmatrix, maxIter=250, stopCond=10**-6) :

        """
        Overwrite the method of GIF to keep track of the final log-likelihood and interrupt fit in case of
        numerical instabilities.
        """

        likelihoood_problem_flag = False

        # Precompute all the matrices used in the gradient ascent (see Eq. 20 in Pozzorini et al. 2015)
        ################################################################################################

        # here X refer to the matrix made of y vectors defined in Eq. 21 (Pozzorini et al. 2015)
        # since the fit can be perfomed on multiple traces, we need lists
        all_X        = []

        # similar to X but only contains temporal samples where experimental spikes have been observed
        # storing this matrix is useful to improve speed when computing the likelihood as well as its derivative
        all_X_spikes = []

        # sum X_spikes over spikes. Precomputing this quantity improve speed when the gradient is evaluated
        all_sum_X_spikes = []


        # variables used to compute the loglikelihood of a Poisson process spiking at the experimental firing rate
        T_tot = 0.0
        N_spikes_tot = 0.0

        traces_nb = 0

        for tr in experiment.trainingset_traces:

            if tr.useTrace :

                traces_nb += 1

                # Simulate subthreshold dynamics
                (time, V_est, eta_sum_est) = self.simulateDeterministic_forceSpikes(tr.I, tr.V[0], tr.getSpikeTimes())

                # Precomputes matrices to compute gradient ascent on log-likelihood
                # depeinding on the model being fitted (static vs dynamic threshodl) different buildXmatrix functions can be used
                (X_tmp, X_spikes_tmp, sum_X_spikes_tmp, N_spikes, T) = buildXmatrix(tr, V_est)

                T_tot        += T
                N_spikes_tot += N_spikes

                all_X.append(X_tmp)
                all_X_spikes.append(X_spikes_tmp)
                all_sum_X_spikes.append(sum_X_spikes_tmp)

        # Compute log-likelihood of a poisson process (this quantity is used to normalize the model log-likelihood)
        ################################################################################################

        logL_poisson = N_spikes_tot*(np.log(N_spikes_tot/T_tot)-1)


        # Perform gradient ascent
        ################################################################################################

        print "Maximize log-likelihood (bit/spks)..."

        beta = beta0
        old_L = 1

        for i in range(maxIter) :

            learning_rate = 1.0

            # In the first iterations using a small learning rate makes things somehow more stable
            if i<=10 :
                learning_rate = 0.1


            L=0; G=0; H=0;

            for trace_i in np.arange(traces_nb):

                # compute log-likelihood, gradient and hessian on a specific trace (note that the fit is performed on multiple traces)
                (L_tmp,G_tmp,H_tmp) = self.computeLikelihoodGradientHessian(beta, all_X[trace_i], all_X_spikes[trace_i], all_sum_X_spikes[trace_i])

                # note that since differentiation is linear: gradient of sum = sum of gradient ; hessian of sum = sum of hessian
                L+=L_tmp;
                G+=G_tmp;
                H+=H_tmp;


            # Update optimal parametes (ie, implement Newton step) by tacking into account multiple traces
            try :
                beta = beta - learning_rate*np.dot(inv(H),G)
            except np.linalg.linalg.LinAlgError as err:
                print "Problem during gradient ascent. Hessian is singular."
                self.fit_problem = True
                likelihoood_problem_flag = True
                self.fit_problem_which.append("8")
                break

            if (i>0 and (old_L - L) > 10**5) :                          # Likelihood drops drammatically during ascent interrupt
                print "\nLast value of log likelihood (bits/spk): ", L
                print "\nGradient ascent interruped due to numerical instability."
                self.fit_problem = True
                likelihoood_problem_flag = True
                self.fit_problem_which.append("3")
                break

            if (i>0 and abs((L-old_L)/old_L) < stopCond) :              # If converged
                print "\nConverged after %d iterations!\n" % (i+1)
                break

            old_L = L

            # Compute normalized likelihood (for print)
            # The likelihood is normalized with respect to a poisson process and units are in bit/spks
            L_norm = (L-logL_poisson)/np.log(2)/N_spikes_tot
            reprint(L_norm)

            if math.isnan(L_norm):
                print "Problem during gradient ascent. Optimizatino stopped."
                self.fit_problem = True
                likelihoood_problem_flag = True
                self.fit_problem_which.append("1")
                break

        if (i==maxIter - 1) :                                           # If too many iterations

            print "\nNot converged after %d iterations.\n" % (maxIter)
            self.fit_problem = True
            likelihoood_problem_flag = True
            self.fit_problem_which.append("2")

        self.beta_opt = beta

        if likelihoood_problem_flag :
            print "Likelihood sucesfully maximized."
        else :
            self.L_norm_train = L_norm


        return (beta, likelihoood_problem_flag)


    #############################################################################
    # PERF EVAL
    #############################################################################

    def computePerf(self, experiment, DT_beforeSpike=5.0):

        buildXmatrix = self.buildXmatrix_dynamicThreshold

        print "\nGIF MODEL - Evaluate perforance on test set..."

        # Expand eta in basis functions
        self.dt = experiment.dt


        all_X        = []
        all_X_spikes = []
        all_sum_X_spikes = []

        T_tot = 0.0
        N_spikes_tot = 0.0

        traces_nb = 0

        # Compute percentage of variance explained on V (see Eq. 26 in Pozzorini et al. PLOS Comp. Biol. 2105)
        ####################################################################################################

        SSE = 0     # sum of squared errors
        VAR = 0     # variance of data

        for tr in experiment.testset_traces :

            if tr.useTrace :

                traces_nb += 1

                # Simulate subthreshold dynamics
                (time, V_est, eta_sum_est) = self.simulateDeterministic_forceSpikes(tr.I, tr.V[0], tr.getSpikeTimes())

                # Compute pct var expalined
                indices_tmp = tr.getROI_FarFromSpikes(0.0, self.Tref)
                SSE += sum((V_est[indices_tmp] - tr.V[indices_tmp])**2)
                VAR += len(indices_tmp)*np.var(tr.V[indices_tmp])

                # Estimate likelihood
                (X_tmp, X_spikes_tmp, sum_X_spikes_tmp, N_spikes, T) = buildXmatrix(tr, V_est)

                T_tot        += T
                N_spikes_tot += N_spikes

                all_X.append(X_tmp)
                all_X_spikes.append(X_spikes_tmp)
                all_sum_X_spikes.append(sum_X_spikes_tmp)


        # Finalize percentage of variance explained

        self.V_varexp_test = (1.0 - SSE / VAR) * 100.0
        print "Percentage of variance explained (on V): %0.2f" % (self.V_varexp_test)


        # Finalize likelihood
        L=0
        for trace_i in np.arange(traces_nb):

            (L_tmp,G_tmp,H_tmp) = self.computeLikelihoodGradientHessian(self.beta_opt, all_X[trace_i], all_X_spikes[trace_i], all_sum_X_spikes[trace_i])
            L+=L_tmp;

        logL_poisson = N_spikes_tot*(np.log(N_spikes_tot/T_tot)-1)
        L_norm = (L-logL_poisson)/np.log(2)/N_spikes_tot

        self.L_norm_test = L_norm

        print "Log-likelihood (bits/spks): %0.2f" % (self.L_norm_test)



    def computeLikelihood(self, experiment, beta0, buildXmatrix) :

        # variables used to compute the loglikelihood of a Poisson process spiking at the experimental firing rate
        T_tot = 0.0
        N_spikes_tot = 0.0

        traces_nb = 0

        for tr in experiment.testset_traces:

            if tr.useTrace :



                # Simulate subthreshold dynamics
                (time, V_est, eta_sum_est) = self.simulateDeterministic_forceSpikes(tr.I, tr.V[0], tr.getSpikeTimes())

                # Precomputes matrices to compute gradient ascent on log-likelihood
                # depeinding on the model being fitted (static vs dynamic threshodl) different buildXmatrix functions can be used
                (X_tmp, X_spikes_tmp, sum_X_spikes_tmp, N_spikes, T) = buildXmatrix(tr, V_est)

                T_tot        += T
                N_spikes_tot += N_spikes

                all_X.append(X_tmp)
                all_X_spikes.append(X_spikes_tmp)
                all_sum_X_spikes.append(sum_X_spikes_tmp)

        # Compute log-likelihood of a poisson process (this quantity is used to normalize the model log-likelihood)
        ################################################################################################

        logL_poisson = N_spikes_tot*(np.log(N_spikes_tot/T_tot)-1)


        # Perform gradient ascent
        ################################################################################################

        print "Maximize log-likelihood (bit/spks)..."

        beta = beta0
        old_L = 1

        for i in range(maxIter) :

            learning_rate = 1.0

            # In the first iterations using a small learning rate makes things somehow more stable
            if i<=10 :
                learning_rate = 0.1


            L=0; G=0; H=0;

            for trace_i in np.arange(traces_nb):

                # compute log-likelihood, gradient and hessian on a specific trace (note that the fit is performed on multiple traces)
                (L_tmp,G_tmp,H_tmp) = self.computeLikelihoodGradientHessian(beta, all_X[trace_i], all_X_spikes[trace_i], all_sum_X_spikes[trace_i])

                # note that since differentiation is linear: gradient of sum = sum of gradient ; hessian of sum = sum of hessian
                L+=L_tmp;
                G+=G_tmp;
                H+=H_tmp;


            # Update optimal parametes (ie, implement Newton step) by tacking into account multiple traces

            beta = beta - learning_rate*np.dot(inv(H),G)

            if (i>0 and abs((L-old_L)/old_L) < stopCond) :              # If converged
                print "\nConverged after %d iterations!\n" % (i+1)
                break

            old_L = L

            # Compute normalized likelihood (for print)
            # The likelihood is normalized with respect to a poisson process and units are in bit/spks
            L_norm = (L-logL_poisson)/np.log(2)/N_spikes_tot
            reprint(L_norm)

            if math.isnan(L_norm):
                print "Problem during gradient ascent. Optimizatino stopped."
                self.fit_problem = True
                break

        if (i==maxIter - 1) :                                           # If too many iterations

            print "\nNot converged after %d iterations.\n" % (maxIter)
            self.fit_problem = True

        self.L_norm_test = L_norm

        return beta

    def simulate_seed(self, I, V0, seed=1, u_vec=False):

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

        # random values can be given directly
        if hasattr(u_vec, "__len__"): # u_vec is a vector
            urand = u_vec
        else:
            np.random.seed(seed)
            urand = np.random.uniform(0,1,p_T)

        urand = np.array(urand, dtype="double")

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
        rnd = np.array(np.zeros(p_T), dtype="double")
        l = np.array(np.zeros(p_T), dtype="double")
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
                    V[t+1] = V[t] + dt/C*( -gl*(V[t] - El) + I[t] - eta_sum[t]);


                    // COMPUTE PROBABILITY OF EMITTING ACTION POTENTIAL
                    lambda = lambda0*exp( (V[t+1]-Vt_star-gamma_sum[t])/DeltaV );
                    l[t+1] = lambda;
                    p_dontspike = exp(-lambda*(dt/1000.0));                                  // since lambda0 is in Hz, dt must also be in Hz (this is why dt/1000.0)


                    // PRODUCE SPIKE STOCHASTICALLY
                    r = urand[t];
                    rnd[t+1] = r;

                    if (r > p_dontspike) {

                        // printf("t:%d ,r=%f, p_dontspike=%f\\n", t, r, p_dontspike);

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

        vars = [ 'l', 'rnd', 'urand', 'p_T','p_dt','p_gl','p_C','p_El','p_Vr','p_Tref','p_Vt_star','p_DV','p_lambda0','V','I','p_eta','p_eta_l','eta_sum','p_gamma','gamma_sum','p_gamma_l','spks' ]

        v = weave.inline(code, vars)

        time = np.arange(p_T)*self.dt

        eta_sum   = eta_sum[:p_T]
        V_T = gamma_sum[:p_T] + p_Vt_star

        spks = (np.where(spks==1)[0])*self.dt

        return (time, V, eta_sum, V_T, spks, rnd, l)


    def getResultDictionary(self):


        modelparam = {
                'C' : self.C,
                'gl' : self.gl,
                'El': self.El,
                'Vr' : self.Vr,
                'Tref' : self.Tref,
                'Vt_star' : self.Vt_star,
                'DV' : self.DV,
                'lambda0' : self.lambda0,
                'eta' : (self.eta.getNbOfBasisFunctions(), self.eta.taus, self.eta.getCoefficients()),
                'gamma' : (self.gamma.getNbOfBasisFunctions(), self.gamma.taus, self.gamma.getCoefficients())
                }


        res = { 'gid' : self.gid,
                'fit_problem'  : self.fit_problem,
                'fit_problem_which' : self.fit_problem_which,
                'likelihood_trainingset' : self.L_norm_train,
                'likelihood_testset' : self.L_norm_test,
                'pct_var_explained_trainingset' : self.V_varexp_train,
                'pct_var_explained_testset' : self.V_varexp_test,
                'pct_var_explained_changeduetoexpassumption' : self.V_varexp_train_change,
                'cpu_time' : self.cputime,
                'model' : modelparam
               }

        return res




    def printSummaryData(self):

        print "\nSUMMARY DATA - CELL NUMBER: ", self.gid
        print "-------------------------------------\n"

        if self.fit_problem :

            print "Problem occurred during fitting procedure:"
            for problem in self.fit_problem_which :
                print "Error ", problem, " -- ", self.problems_dic[problem]

        else :

            print "No problem encoutered during the fit."


        print "\nLog-likelihood - training set (bits/spks): ", self.L_norm_train
        print "Log-likelihood - test set (bits/spks): ", self.L_norm_test
        print "Variance explained on V - training set (pct): ", self.V_varexp_train
        print "Variance explained on V - test set (pct): ", self.V_varexp_test
        print "Change in variance explained on V due to exp assumption (pct): ", self.V_varexp_train_change
        print "CPU time (s): ", self.cputime

        print "-------------------------------------\n"
