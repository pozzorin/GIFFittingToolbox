import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import fftconvolve




class SpikeTrainComparator :

    """
    This class contains experimental and predicted spike trains.
    Use this class to visualize and quantify the Md similarity metrix between data and model prediction.
    
    To use the Kistler coincidence widnow K(s,s',D)=rect(s/D)*delta(s') call the function
    computeMD_Kistler(self, delta, dt)
    
    To use the double rectangular coincidence window K(s,s',D)=rect(s/D)*rect(s'/D) call the function
    computeMD_Rect(self, delta, dt)
    """

    def __init__(self, T, spks_data, spks_model):
        
        self.T          = T                 # ms, duration of spike-trains
        
        self.spks_data  = spks_data         # a small set of spike-trains (in ms)
        self.spks_model = spks_model        # a large set of spike-trains (in ms) 
        

    def getAverageFiringRate(self):
        
        spks_cnt_data = 0
                
        for  s in self.spks_data :
            spks_cnt_data += len(s)

        rate_data = float(spks_cnt_data)/(self.T/1000.0)/len(self.spks_data)

        spks_cnt_model = 0 
                   
        for  s in self.spks_model :
            spks_cnt_model += len(s)

        rate_model = float(spks_cnt_model)/(self.T/1000.0)/len(self.spks_model)

        return (rate_data, rate_model)


    #########################################################################
    # MD KISTLER WINDOW
    #########################################################################
    
    def computeMD_Kistler(self, delta, dt) :        
         
        print "Computing Md* - Kistler window (%0.1f ms precision)..." % (delta)
        
        KistlerDotProduct = SpikeTrainComparator.Md_dotProduct_Kistler
        KistlerDotProduct_args = {'delta' : delta }
        
        return self.computeMD(KistlerDotProduct, KistlerDotProduct_args, dt)


    @classmethod        
    def Md_dotProduct_Kistler(cls, s1_train, s2_train, args, dt):
        
        delta = args['delta']
        
        rect_size_i = 2*int(float(delta)/dt)
        rect        = np.ones(rect_size_i)
        
        s1_filtered = fftconvolve(s1_train, rect, mode='same')
        
        dotProduct = np.sum(s1_filtered*s2_train)
   
        return dotProduct


    #########################################################################
    # MD RECT*RECT WINDOW
    #########################################################################

    def computeMD_Rect(self, delta, dt) :        
         
        print "Computing Md* - Rectangular window (%0.1f ms precision)..." % (delta)
        
        RectDotProduct = SpikeTrainComparator.Md_dotProduct_Rect
        RectDotProduct_args = {'delta' : delta }
        
        return self.computeMD(RectDotProduct, RectDotProduct_args, dt)
                


    @classmethod        
    def Md_dotProduct_Rect(cls, s1_train, s2_train, args, dt=0.1):
        
        delta = args['delta']
        
        rect_size_i = 2*int(float(delta)/dt)
        rect        = np.ones(rect_size_i)
        
        s1_filtered = fftconvolve(s1_train, rect, mode='same')
        s2_filtered = fftconvolve(s2_train, rect, mode='same')
                
        dotProduct = np.sum(s1_filtered*s2_filtered)
   
        return dotProduct
         



    def computeMD(self, dotProduct, dotProductArgs, dt) :
              
        T = self.T

        # Compute experimental spike trains (given spike times)  
        all_spike_train_data = []
        all_spike_train_data_nb = len(self.spks_data)
                
        for s in self.spks_data :
        
            spike_train_tmp = SpikeTrainComparator.getSpikeTrain(s, T, dt)
            all_spike_train_data.append(spike_train_tmp)
    
        # Compute average spike-strain for both sets
        spiketrain_data_avg = SpikeTrainComparator.getAverageSpikeTrain(self.spks_data, T, dt) 
        spiketrain_model_avg = SpikeTrainComparator.getAverageSpikeTrain(self.spks_model, T, dt)        

        # Compute dot product <data, model>     
        #dotproduct_dm = SpikeTrainComparator.Md_dotProduct_Kistler(spiketrain_data_avg, spiketrain_model_avg, delta, dt)
        dotproduct_dm = dotProduct(spiketrain_data_avg, spiketrain_model_avg, dotProductArgs, dt=dt)
          
                       
        # Compute dot product <model, model>
        #dotproduct_mm = SpikeTrainComparator.Md_dotProduct_Kistler(spiketrain_model_avg, spiketrain_model_avg, delta, dt)
        dotproduct_mm = dotProduct(spiketrain_model_avg, spiketrain_model_avg, dotProductArgs, dt=dt)
        
        
        # Compute dot product <data, data> using unbiased method 
        tmp = 0   
        for i in range(all_spike_train_data_nb) :
            for j in range(i+1, all_spike_train_data_nb) :    
                tmp += dotProduct(all_spike_train_data[i], all_spike_train_data[j], dotProductArgs, dt=dt)
        
        dotproduct_dd_unbaiased = tmp/ (all_spike_train_data_nb*(all_spike_train_data_nb-1)/2.0)
                
        MDstar = 2.0*dotproduct_dm / (dotproduct_dd_unbaiased + dotproduct_mm)
        
        print "Md* = %0.4f" % (MDstar)
                

        return MDstar

  
    @classmethod
    def getSpikeTrain(cls, s, T, dt):
        
        """
        Given spike times in s, build a spike train of duration T (in ms) and with a resolution of dt.
        """
        
        T_i = int(T/dt)
        
        s_i = np.array(s, dtype='double')
        s_i = s_i/dt
        s_i = np.array(s_i, dtype='int')
         
        spike_train = np.zeros(T_i)
        spike_train[s_i] = 1.0
        
        return np.array(spike_train)
    
    


    @classmethod
    def getAverageSpikeTrain(cls, all_s, T, dt):
        
        """
        Given set of spike trains s (defined as list of spike times), build the mean spike train vector of duration T (in ms) and with a resolution of dt.
        """
        
        T_i = int(T/dt)
        average_spike_train = np.zeros(T_i)
        nbSpikeTrains = len(all_s)
        
        for s in all_s :        
            s_i = np.array(s, dtype='double')
            s_i = s_i/dt
            s_i = np.array(s_i, dtype='int')
        
            average_spike_train[s_i] += 1.0
        
        average_spike_train = average_spike_train / float(nbSpikeTrains)
        
        return np.array(average_spike_train)

      
      
    #######################################################################
    # FUNCTIONS FOR PLOTTING
    #######################################################################
    def plotRaster(self, delta=10.0, dt=0.1):
        
        plt.figure(facecolor='white', figsize=(14,4))
        
        # Plot raster
        plt.subplot(2,1,1)
        
        nb_rep = min(len(self.spks_data), len(self.spks_model) )
        
        cnt = 0
        for spks in self.spks_data[:nb_rep] :
            cnt -= 1      
            plt.plot(spks, cnt*np.ones(len(spks)), '|', color='black', ms=5, mew=2)

        for spks in self.spks_model[:nb_rep] :
            cnt -= 1      
            plt.plot(spks, cnt*np.ones(len(spks)), '|', color='red', ms=5, mew=2)
          
   
        plt.yticks([])  
        
        # Plot PSTH
        plt.subplot(2,1,2)
        rect_width  = delta
        rect_size_i = int(float(rect_width)/dt)
        rect_window = np.ones(rect_size_i)/(rect_width/1000.0)

        spks_avg_data         = SpikeTrainComparator.getAverageSpikeTrain(self.spks_data, self.T, dt)
        spks_avg_data_support = np.arange(len(spks_avg_data))*dt
        spks_avg_data_smooth  = fftconvolve(spks_avg_data, rect_window, mode='same')
           
        spks_avg_model = SpikeTrainComparator.getAverageSpikeTrain(self.spks_model, self.T, dt)
        spks_avg_model_support = np.arange(len(spks_avg_data))*dt             
        spks_avg_model_smooth  = fftconvolve(spks_avg_model, rect_window, mode='same')        
        
        plt.plot(spks_avg_data_support, spks_avg_data_smooth, 'black', label='Data')
        plt.plot(spks_avg_model_support, spks_avg_model_smooth, 'red', label='Model')

        plt.legend()

        plt.xlabel("Time (ms)")
        plt.ylabel('PSTH (Hz)')
        
        # Compute % of variance explained
        SSE = np.mean( (spks_avg_data_smooth-spks_avg_model_smooth)**2 )
        VAR = np.var(spks_avg_data_smooth)
        pct_variance_explained = (1.0 - SSE/VAR)*100.0
        
        print "Percentage of variance explained: %0.1f" % (pct_variance_explained)
        
        plt.show()
        
        
        
        


        
        
        

        