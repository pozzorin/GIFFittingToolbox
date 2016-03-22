"""
Preliminary wrapping of NEURON version of GIF neuron model to work with PyNN.

Author: Andrew Davison
"""

import sys

sys.path.extend(['..',      # src/HBP
                 '../..'])  # src
import numpy
from neuron import h
from NEURON.GIF_NEURON import GIF_NEURON
from Filter_Exps import Filter_Exps
from pyNN.neuron import NativeCellType, simulator, run
from quantities import mV

simulator.load_mechanisms('../NEURON')


class GIFNeuron(GIF_NEURON):

    def __init__(self,
                 dt_interpolation=0.025,
                 tau_eta1=10.0,
                 tau_eta2=50.0,
                 tau_eta3=250.0,
                 a_eta1=0.2,
                 a_eta2=0.05,
                 a_eta3=0.025,
                 tau_gamma1=5.0,
                 tau_gamma2=200.0,
                 tau_gamma3=250.0,
                 a_gamma1=15.0,
                 a_gamma2=3.0,
                 a_gamma3=1.0,
    ):
        super(GIFNeuron, self).__init__(dt_interpolation)
        self.eta = Filter_Exps()
        self.eta.setFilter_Timescales([tau_eta1, tau_eta2, tau_eta3])
        self.eta.setFilter_Coefficients([a_eta1, a_eta2, a_eta3])
        self.gamma = Filter_Exps()
        self.gamma.setFilter_Timescales([tau_gamma1, tau_gamma2, tau_gamma3])
        self.gamma.setFilter_Coefficients([a_gamma1, a_gamma2, a_gamma3])
        self._build(passive_axon=False)

        # needed for PyNN
        self.source_section = self.soma
        self.source = self.gif_fun
        self.rec = h.NetCon(self.source, None)
        self.parameter_names = ('dt_interpolation',
                                'tau_eta1', 'tau_eta2', 'tau_eta3',
                                'a_eta1', 'a_eta2', 'a_eta3',
                                'tau_gamma1', 'tau_gamma2', 'tau_gamma3',
                                'a_gamma1', 'a_gamma2', 'a_gamma3')
        self.traces = {}
        self.recording_time = False

    def memb_init(self):
        for seg in self.soma:
            seg.v = self.v_init
        if hasattr(self, 'axon'):
            for seg in self.axon:
                seg.v = self.v_init

def simulate_seed(population, I, V0, seed, passive_axon=False):

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
    assert passive_axon is False
    self = population[0]._cell

    # Input parameters
    T = len(I)*self.dt

    rndd = h.Random(seed)
    rndd.uniform(0, 1)
    self.gif_fun.setRNG(rndd)

    rec_t = h.Vector()
    rec_t.record(h._ref_t)
    h.celsius = 34

    population.initialize(v=V0)
    run(T)

    time = numpy.array(rec_t)
    data = population.get_data().segments[0]
    signals = {}
    for sig in data.analogsignalarrays:
        signals[sig.name] = sig.magnitude

    V = signals['v']
    eta_sum = signals['i_eta']
    V_T = signals['gamma_sum'] + self.Vt_star
    p_dontspike = signals['p_dontspike']
    urand = signals['rand']
    # time = _time

    i = I

    import efel

    trace = {}
    trace['T'] = time
    trace['V'] = V
    trace['stim_start'] = [0]
    trace['stim_end'] = [T]
    traces = [trace]

    efel_result = efel.getFeatureValues(traces, ['peak_time'])[0]
    spks = efel_result['peak_time']

    return (time[:-1], V[:-1], eta_sum[:-1], V_T[:-1], spks, p_dontspike[:-1], urand[:-1], i)


class GIFNeuronType(NativeCellType):
    default_parameters = {'dt_interpolation': 0.025,
                          'tau_eta1': 10.0, 'tau_eta2': 50.0, 'tau_eta3': 250.0,
                          'a_eta1': 0.2, 'a_eta2': 0.05, 'a_eta3': 0.025,
                          'tau_gamma1': 5.0, 'tau_gamma2': 200.0, 'tau_gamma3': 250.0,
                          'a_gamma1': 15.0, 'a_gamma2': 3.0, 'a_gamma3': 1.0,}
    default_initial_values = {'v': -70.0}
    recordable = ['v', 'i_eta', 'gamma_sum', 'p_dontspike', 'rand']
    units = {'v' : 'mV', 'i_eta': 'nA', 'gamma_sum': 'mV',
             'p_dontspike': 'dimensionless', 'rand': 'dimensionless'}
    receptor_types = []
    model = GIFNeuron
