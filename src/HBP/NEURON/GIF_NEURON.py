import sys
sys.path.append('../../')

import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy

from scipy import weave
from numpy.linalg import inv

from Tools import reprint
from numpy import nan, NaN

import math
from GIF import *

class GIF_NEURON(GIF):

    """
    NEURON implementation of GIF model using exponential filters
    """

    def __init__(self, dt):

        super(GIF_NEURON, self).__init__(dt)


    def simulate_seed(self, I, V0, seed):

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
        from neuron import h

        # Input parameters
        T         = len(I)*self.dt

        soma = h.Section(name='soma')

        cm = 1 # uF/cm2
        Area = self.C*1e-3 / cm    # (uF/(uF/cm2)) = cm2 # self.C in nF
        l = numpy.sqrt(Area / numpy.pi) * 1e4   # um

        soma.cm = 1
        soma.L = l      # um
        soma.diam = l   # um
        soma.nseg = 1

        soma.insert('pas')
        g = self.gl*1e-6 / Area     # S/cm2 # self.gl in uS
        soma(0.5).pas.g = g
        soma(0.5).pas.e = self.El

        gif_fun = h.gif(soma(0.5))
        #gif_fun.toggleVerbose()

        gif_fun.Vr        = self.Vr
        gif_fun.Tref      = self.Tref
        gif_fun.Vt_star   = self.Vt_star
        gif_fun.DV        = self.DV
        gif_fun.lambda0   = self.lambda0

        # Model kernels
        if self.eta.filter_coeffNb > 3:
            raise Exception("Number of filter coefficients too large "
                            "only 3 coefficients implemented in NEURON")

        gif_fun.tau_eta1 = self.eta.taus[0]
        gif_fun.tau_eta2 = self.eta.taus[1]
        gif_fun.tau_eta3 = self.eta.taus[2]
        gif_fun.a_eta1 = self.eta.filter_coeff[0]
        gif_fun.a_eta2 = self.eta.filter_coeff[1]
        gif_fun.a_eta3 = self.eta.filter_coeff[2]

        gif_fun.tau_gamma1 = self.gamma.taus[0]
        gif_fun.tau_gamma2 = self.gamma.taus[1]
        gif_fun.tau_gamma3 = self.gamma.taus[2]
        gif_fun.a_gamma1 = self.gamma.filter_coeff[0]
        gif_fun.a_gamma2 = self.gamma.filter_coeff[1]
        gif_fun.a_gamma3 = self.gamma.filter_coeff[2]

        rndd = h.Random(seed)
        #randseed1 = seed * 100000 + 100
        #randseed2 = seed + 250
        #rndd.MCellRan4(randseed1, randseed2)
        #rndd.normal(0, 1)
        rndd.uniform(0, 1)
        #rndd.negexp(1)
        gif_fun.setRNG(rndd)

        # Inject current
        iclamp = h.IClamp(0.5, sec=soma)
        iclamp.dur = T
        #iclamp.del = 0

        #I = numpy.array(I, dtype="double")
        i_vec = h.Vector(I)

        #t = h.Vector(np.arange(T)*self.dt)
        #print np.arange(T)*self.dt
        i_vec.play(iclamp._ref_amp, self.dt)

        rec_t = h.Vector()
        rec_t.record(h._ref_t)

        rec_v = h.Vector()
        rec_v.record(soma(0.5)._ref_v)

        rec_eta = h.Vector()
        rec_eta.record(gif_fun._ref_i_eta)

        rec_gamma = h.Vector()
        rec_gamma.record(gif_fun._ref_gamma_sum)

        rec_pdontspike = h.Vector()
        rec_pdontspike.record(gif_fun._ref_p_dontspike)

        rec_urand = h.Vector()
        rec_urand.record(gif_fun._ref_rand)

        rec_i = h.Vector()
        rec_i.record(iclamp._ref_amp)

        h.load_file("stdrun.hoc")

        h.dt = self.dt
        h.steps_per_ms = 1.0 / self.dt
        h.v_init = V0

        h.celsius = 34
        h.init()
        h.tstop = T
        h.run()

        time = numpy.array(rec_t)
        V = numpy.array(rec_v)

        eta_sum = numpy.array(rec_eta)
        V_T = numpy.array(rec_gamma) + self.Vt_star

        p_dontspike = numpy.array(rec_pdontspike)

        urand = numpy.array(rec_urand)
        i = numpy.array(rec_i)

        #i = numpy.array(i_vec)
        import efel

        trace = {}
        trace['T'] = time
        trace['V'] = V
        trace['stim_start'] = [0]
        trace['stim_end'] = [T]
        traces = [trace]

        efel_result = efel.getFeatureValues(traces, ['peak_time'])[0]
        spks = efel_result['peak_time']

        return (time[:-1], V[:-1], eta_sum[:-1], V_T[:-1], spks, p_dontspike[:-1], urand[:-1], i[:-1])



if __name__ == '__main__':

    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    dt = 0.1
    hGIF = GIF_NEURON(dt)

    from Filter_Exps import *
    hGIF.eta = Filter_Exps()
    hGIF.eta.setFilter_Timescales([10.0, 50.0, 250.0])
    hGIF.eta.setFilter_Coefficients([0.2, 0.05, 0.025])
    hGIF.gamma = Filter_Exps()
    hGIF.gamma.setFilter_Timescales([5.0, 200.0, 250.0])
    hGIF.gamma.setFilter_Coefficients([15.0, 3.0, 1.0])


    # Simulate model response
    seed = 1    # seed used for random number generator
    V0 = -70.0  # mV, initial condition for voltage

    T = 100.0
    I0_max = 0.5
    I0_all = numpy.random.rand(50)*I0_max

    I = []

    for I0 in I0_all :
        I.append(I0*numpy.ones(int(T/dt)))

    I = numpy.concatenate(I)

    (time, V, eta_sum, V_T, spks, p_dontspike, urand, i) = hGIF.simulate_seed(I, V0, seed)

    print len(urand), urand

    plt.figure('hGIF', figsize=(16,8))
    plt.subplot(3,1,1)
    plt.plot(time, i, 'gray')
    plt.plot(time, eta_sum, 'red')

    plt.subplot(3,1,2)
    plt.plot(time, V, 'black')
    plt.plot(time, V_T, 'red')

    plt.subplot(3,1,3)
    plt.plot(time, p_dontspike, 'black')

    plt.show()
