import abc
from Experiment import *


class AEC :

    """
    Abstract class defining providing an interface for Active Electrode Compensation (AEC).
    Active Electrode Compensation is a technique used to estimate the membrane potential V from the signal acquired using the current-clamp technique.
    Several versions of AEC have been proposed in the litterature. See, e.g., Badel et al. 2007 or Brette et al. 2007.
    """
    
    __metaclass__  = abc.ABCMeta
    
    @abc.abstractmethod
    def performAEC(self, experiment):
        
        """
        Preprocess all the traces in Experiment to estimate the true membrane potential V, 
        based on the recorded voltage V_rec and as well as the input current I.
        """
   
    @abc.abstractmethod
    def plot(self):
        
        """
        Plot the filters used to perform AEC (e.g. optimal I-V_rec filter and electrode filter used to compensate the recorded signal).
        """    
        
    
            
