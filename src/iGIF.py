import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

import abc

from scipy import weave
from numpy.linalg import inv

from SpikingModel import *
from GIF import *
from Filter_Rect_LogSpaced import *

import Tools
from Tools import reprint



class iGIF(GIF) :

    """
    Abstract class to define the:
    
    inactivating Generalized Integrate and Fire models
    
    Spike are produced stochastically with firing intensity:
    
    lambda(t) = lambda0 * exp( (V(t)-V_T(t))/DV ),
    
    where the membrane potential dynamics is given (as in Pozzorini et al. PLOS Comp. Biol. 2015) by:
    
    C dV/dt = -gl(V-El) + I - sum_j eta(t-\hat t_j)
    
    This equation differs from the one used in Mensi et al. PLOS Comp. Biol. 2016 only because spike-triggerend adaptation is
    current based and not conductance based.
    
    The firing threshold V_T is given by:
    
    V_T = Vt_star + sum_j gamma(t-\hat t_j) + theta(t)
    
    and \hat t_j denote the spike times and theta(t) is given by:
    
    tau_theta dtheta/dt = -theta + f(V)
    
    Classes that inherit form iGIF must specify the nature of the coupling f(V) (this function can eg be defined as 
    a liner sum of rectangular basis functions to perform a nonparametric fit).
    """
    
    __metaclass__  = abc.ABCMeta


    def __init__(self, dt=0.1):
    
        GIF.__init__(self, dt=dt)          

         
    @abc.abstractmethod  
    def getNonlinearCoupling(self):
        
        """
        This method should compute and return:
        - f(V): function defining the steady state value of theta as a funciton of voltage
        - support : the voltage over which f(V) is defined
        """
        
    
    ########################################################################################################
    # PLOT AND PRINT FUNCTIONS
    ########################################################################################################     
  
    def plotParameters(self) :
        
        """
        Plot parameters of the iGIF model.
        """
        
        fig = plt.figure(facecolor='white', figsize=(16,4))
        
        
        # Plot spike triggered current
        ####################################################################################################
        
        plt.subplot(1,4,1)
        
        (eta_support, eta) = self.eta.getInterpolatedFilter(self.dt) 
        
        plt.plot(eta_support, eta, color='red', lw=2)
        plt.plot([eta_support[0], eta_support[-1]], [0,0], ls=':', color='black', lw=2)
            
        plt.xlim([eta_support[0], eta_support[-1]])    
        plt.xlabel("Time (ms)")
        plt.ylabel("Eta (nA)")
        
        
        # Plot spike triggered movement of the firing threshold
        ####################################################################################################
 
        plt.subplot(1,4,2)
        
        (gamma_support, gamma) = self.gamma.getInterpolatedFilter(self.dt) 
        
        plt.plot(gamma_support, gamma, color='red', lw=2)
        plt.plot([gamma_support[0], gamma_support[-1]], [0,0], ls=':', color='black', lw=2)
            
        plt.xlim([gamma_support[0], gamma_support[-1]])    
        plt.xlabel("Time (ms)")
        plt.ylabel("Gamma (mV)")
        
        
        # Plot nonlinear coupling between firing threshold and membrane potential
        ####################################################################################################
 
        plt.subplot(1,4,3)
        
        (support, theta_inf) = self.getNonlinearCoupling()

        plt.plot(support, support, '--', color='black') 
        plt.plot(support, theta_inf, 'red', lw=2)
        plt.plot([self.Vr], [self.Vt_star], 'o', mew=2, mec='black',  mfc='white', ms=8)
        
        plt.ylim([self.Vt_star-10.0,-20.0])
                
        plt.xlabel("Membrane potential (mV)")
        plt.ylabel("Theta (mV)")

        plt.subplots_adjust(left=0.08, bottom=0.10, right=0.95, top=0.93, wspace=0.25, hspace=0.25)
 
 
    @classmethod
    def compareModels(cls, iGIFs, labels=None):

        """
        Given a list of iGIF models, iGIFs, the function produce a plot in which the model parameters are compared.
        """

        # PRINT PARAMETERS        

        print "\n#####################################"
        print "iGIF model comparison"
        print "#####################################\n"
        
        cnt = 0
        for iGIF in iGIFs :
            
            print "Model: " + labels[cnt]          
            iGIF.printParameters()
            cnt+=1

        print "#####################################\n"                
        
        
                
        #######################################################################################################
        # PLOT PARAMETERS
        #######################################################################################################        
        
        plt.figure(facecolor='white', figsize=(14,8))     
        colors = plt.cm.jet( np.linspace(0.7, 1.0, len(iGIFs) ) )   

       
        # MEMBRANE FILTER
        #######################################################################################################
        
        plt.subplot(2,3,1)
            
        cnt = 0
        for iGIF in iGIFs :
            
            if labels == None :
                label_tmp =""
            else :
                label_tmp = labels[cnt]
                
            K_support = np.linspace(0,150.0, 1500)             
            K = 1./iGIF.C*np.exp(-K_support/(iGIF.C/iGIF.gl))     
            plt.plot(K_support, K, color=colors[cnt], lw=2, label=label_tmp)
            cnt += 1
            
        plt.plot([K_support[0], K_support[-1]], [0,0], ls=':', color='black', lw=2, zorder=-1)   
                
        if labels != None :
            plt.legend()  
            
        plt.xlim([K_support[0], K_support[-1]])
        plt.xlabel('Time (ms)')
        plt.ylabel('Membrane filter (MOhm/ms)')  

   
        # SPIKE TRIGGERED CURRENT
        #######################################################################################################
        
        plt.subplot(2,3,2)
            
        cnt = 0
        for iGIF in iGIFs :
                        
            (eta_support, eta) = iGIF.eta.getInterpolatedFilter(0.1)         
            plt.plot(eta_support, eta, color=colors[cnt], lw=2)
            cnt += 1
            
        plt.plot([eta_support[0], eta_support[-1]], [0,0], ls=':', color='black', lw=2, zorder=-1)   
                                
        #plt.xscale('log', nonposx='clip')
        #plt.yscale('log', nonposy='clip')
        plt.xlim([eta_support[0], eta_support[-1]])
        plt.xlabel('Time (ms)')
        plt.ylabel('Eta (ms)')        
        

        # ESCAPE RATE
        #######################################################################################################
 
        plt.subplot(2,3,3)
            
        cnt = 0
        for iGIF in iGIFs :
            
            V_support = np.linspace(iGIF.Vt_star-5*iGIF.DV,iGIF.Vt_star+10*iGIF.DV, 1000) 
            escape_rate = iGIF.lambda0*np.exp((V_support-iGIF.Vt_star)/iGIF.DV)                
            plt.plot(V_support, escape_rate, color=colors[cnt], lw=2)
            cnt += 1

        plt.plot([V_support[0], V_support[-1]], [0,0], ls=':', color='black', lw=2, zorder=-1)  
                  
        plt.ylim([0, 100])    
        plt.xlim([V_support[0], V_support[-1]])
        plt.xlabel('Membrane potential (mV)')
        plt.ylabel('Escape rate (Hz)')  


        # SPIKE TRIGGERED MOVEMENT OF THE FIRING THRESHOLD
        #######################################################################################################
 
        plt.subplot(2,3,4)
            
        cnt = 0
        for myiGIF in iGIFs :
            
            (gamma_support, gamma) = myiGIF.gamma.getInterpolatedFilter(0.1)         
            plt.plot(gamma_support, gamma, color=colors[cnt], lw=2)
            
            cnt += 1
            
        plt.plot([gamma_support[0], gamma_support[-1]], [0,0], ls=':', color='black', lw=2, zorder=-1)   
      
        #plt.xscale('log', nonposx='clip')
        #plt.yscale('log', nonposy='clip')
        plt.xlim([gamma_support[0]+0.1, gamma_support[-1]])
        plt.xlabel('Time (ms)')
        plt.ylabel('Gamma (mV)')   
                

        # NONLINEAR COUPLING OF THE FIRING THRESHOLD
        #######################################################################################################
  
        plt.subplot(2,3,5)      

        cnt = 0
        for iGIF in iGIFs :
            (V, theta) = iGIF.getNonlinearCoupling()
            plt.plot(V, theta, color=colors[cnt], lw=2)
            plt.plot([iGIF.Vr], [iGIF.Vt_star], 'o', mew=2, mec=colors[cnt],  mfc=colors[cnt], ms=8)
            cnt +=1
        
        plt.plot(V, V, color='black', lw=2, ls='--')  
         
        plt.xlim([-80, -20])  
        plt.ylim([-60, -20])     
        plt.xlabel('Membrane potential (mV)')
        plt.ylabel('Threshold theta (mV)')         
        
        plt.subplots_adjust(left=0.08, bottom=0.10, right=0.95, top=0.93, wspace=0.25, hspace=0.25)
        
        plt.show()
        
             
    @classmethod
    def plotAverageModel(cls, iGIFs):


        """
        Average model parameters and plot summary data.
        """

        GIF.plotAverageModel(iGIFs)


        # NONLINEAR THRESHOLD COUPLING
        #######################################################################################################
        plt.subplot(2,4,4)
                    
        K_all = []
        
        plt.plot([-80, -20],[-80,-20], ls='--', color='black', lw=2, zorder=100)   
        
        for iGIF in iGIFs :
                
            (K_support, K) = iGIF.getNonlinearCoupling()
       
            plt.plot(K_support, K, color='0.3', lw=1, zorder=5)
            
            K_all.append(K)

        K_mean = np.mean(K_all, axis=0)
        K_std  = np.std(K_all, axis=0)
        
        plt.fill_between(K_support, K_mean+K_std,y2=K_mean-K_std, color='gray', zorder=0)
        plt.plot(K_support, np.mean(K_all, axis=0), color='red', lw=2, zorder=10)  
            
           
        plt.xlim([-80,-20])
        plt.ylim([-65,-20])
        Tools.removeAxis(plt.gca(), ['top', 'right'])
        plt.xlabel('Membrane potential (mV)')
        plt.ylabel('Threshold coupling (mV)')  
 

        # tau_theta
        #######################################################################################################
        plt.subplot(4,6,12+4)
 
        p_all = []
        for iGIF in iGIFs :
                
            p = iGIF.theta_tau
            p_all.append(p)
            
        plt.hist(p_all, histtype='bar', color='red', ec='white', lw=2)
        plt.xlabel('tau theta (ms)')        
        Tools.removeAxis(plt.gca(), ['top', 'left', 'right'])
        plt.yticks([])     

        plt.show()
