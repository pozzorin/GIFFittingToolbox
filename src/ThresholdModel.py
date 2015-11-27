import matplotlib.pyplot as plt
import numpy as np

from SpikingModel import *

import abc

class ThresholdModel(SpikingModel) :

    """
    Abstract class to define a threshold model.
    A threshold model is a model that explicitly describe the membrane potential V and the firing threshold Vt.
    The GIF model is a Threshold model, the GLM model is not.
    """
    
    __metaclass__  = abc.ABCMeta
    
    
    @abc.abstractmethod
    def simulateVoltageResponse(self, I, dt):
        
        """
        Simulate the model response to an input current I and return:
        spks : list of spike times (in ms)
        V    : voltage trace (in mV)
        Vt   : voltage threshold trace (in mV)
        """
   
    
    
    def computeRateAndThreshold_vs_I(self, mu, sigma, tau, dt, T, ROI, nbRep=10):

        """
        Compute standard FI curve using multiple OU processes with means mu (list), standard deviations sigma (list)
        temporal correlation tau (in ms) and of duration T.
        Use ROI to define the region of interest over which to compute the average firing rate. ROI can be usful when
        one wants to evaluate the transient or the steady state FI.
        This function also compute the theta-I curve, that the average value of the firing threshold as a function of the
        input statistics.
        """

        self.setDt(dt)

        FI_all        = np.zeros((len(sigma),len(mu),nbRep))        
        thetaI_all    = np.zeros((len(sigma),len(mu),nbRep))
        thetaI_VT_all = np.zeros((len(sigma),len(mu),nbRep))
                
        s_cnt = -1
        for s in sigma :
            s_cnt += 1
            
            m_cnt = -1 
            for m in mu :
                
                m_cnt += 1
                
                for r in np.arange(nbRep) :
        
                    I_tmp = Tools.generateOUprocess(T=T, tau=tau, mu=m, sigma=s, dt=dt)
                    
                    (spks_t, V, V_T) = self.simulateVoltageResponse(I_tmp, dt)
                    spks_i = Tools.timeToIndex(spks_t, dt)
        
                    spiks_i_sel = np.where( ( ( spks_t > ROI[0] ) & ( spks_t < ROI[1] ) ) == True)[0] 
                    spiks_i_sel = spks_i[spiks_i_sel]
                    
                    rate = 1000.0*len(spiks_i_sel)/(ROI[1]-ROI[0])
                    FI_all[s_cnt, m_cnt, r] = rate
                                        
                    theta = np.mean(V[spiks_i_sel])
                    thetaI_all[s_cnt, m_cnt, r] = theta
                    
                    theta_VT = np.mean(V_T[spiks_i_sel])
                    thetaI_VT_all[s_cnt, m_cnt, r] = theta_VT
         
                                                            
        return (FI_all, thetaI_all, thetaI_VT_all)




        