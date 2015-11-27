import abc

import numpy as np
import cPickle as pkl

import Tools


class SpikingModel :

    """
    Abstract class defining an interface for a Spiking Neuron Model, a model that produces spikes.
    Inherit from this class to generate a new Spiking Neuron Model.
    To create a new model that explicitly describe the membrane potential and the voltage threshold,
    see the class ThreshodlModel.
    """
    
    __metaclass__  = abc.ABCMeta
    
    
    @abc.abstractmethod
    def simulateSpikingResponse(self, I, dt):
        
        """
        Return an array containing the spike times (in ms) evoked by an input current I(t). 
        Dt define the sampling frequency at which the simulation is performed.
        """
   
   
    def computeFIcurve(self, mu, sigma, tau, dt, T, ROI, nbRep=10):

        """
        Compute standard FI curve using multiple OU processes with means mu (list), standard deviations sigma (list)
        temporal correlation tau (in ms) and of duration T.
        Use ROI to define the region of interest over which to compute the average firing rate. ROI can be usful when
        one wants to evaluate the transient or the steady state FI.
        """

        FI_all = np.zeros((len(sigma),len(mu),nbRep))
        
        s_cnt = -1
        for s in sigma :
            s_cnt += 1
            
            m_cnt = -1 
            for m in mu :
                
                m_cnt += 1
                
                for r in np.arange(nbRep) :
        
                    I_tmp = Tools.generateOUprocess(T=T, tau=tau, mu=m, sigma=s, dt=dt)
    
                    spks = self.simulateSpikingResponse(I_tmp, dt)
        
                    nb_spks = len( np.where( ( ( spks > ROI[0] ) & ( spks < ROI[1] ) ) == True)[0] )
                       
                    rate = 1000.0*float(nb_spks)/(ROI[1]-ROI[0])
                                        
                    FI_all[s_cnt, m_cnt, r] = rate
                    
        return FI_all



    ############################################################################################
    # FUNCTIONS FOR SAVING AND LOADING AN EXPERIMENT
    ############################################################################################
    def save(self, path):
          
        print "Saving: " + path + "..."        
        f = open(path,'w')
        pkl.dump(self, f)
        print "Done!"
        
        
    @classmethod
    def load(cls, path):
        
        print "Load spiking model: " + path + "..."        
      
        f = open(path,'r')
        model = pkl.load(f)
    
        print "Done!" 
           
        return model 