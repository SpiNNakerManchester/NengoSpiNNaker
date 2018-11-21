import numpy as np

import nengo
import nengo_spinnaker_gfe.nengo_simulator as gfe_nengo
import nengo_spinnaker as mundy_nengo
USE_GFE = True


def create_model():
    num_items = 5

    d_key = 2
    d_value = 4

    record_encoders = True

    rng = np.random.RandomState(seed=7)
    keys = nengo.dists.UniformHypersphere(surface=True).sample(
        num_items, d_key, rng=rng)
    values = nengo.dists.UniformHypersphere(surface=False).sample(
        num_items, d_value, rng=rng)

    intercept = (np.dot(keys, keys.T) - np.eye(num_items)).flatten().max()

    def cycle_array(x, period, dt=0.001):
        """Cycles through the elements"""
        i_every = int(round(period/dt))
        if i_every != period/dt:
            raise ValueError("dt (%s) does not divide period (%s)" % (
                dt, period))

        def f(t):
            i = int(round((t - dt)/dt))  # t starts at dt
            return x[(i/i_every)%len(x)]
        return f

    # Model constants
    n_neurons = 200
    dt = 0.001
    period = 0.3
    T = period*num_items*2

    # Model network
    model = nengo.Network()
    with model:

        # Create the inputs/outputs
        stim_keys = nengo.Node(
            output=cycle_array(keys, period, dt), label="stim_keys")

        # Setup probes
        p_keys = nengo.Probe(stim_keys, synapse=None, label="p_keys")

        probes = [p_keys]
    return model, list(), dict(), probes

if __name__ == '__main__':
    network, function_of_time, function_of_time_time_period, \
    probes = create_model()
    if USE_GFE:
        sim = gfe_nengo.NengoSimulator(network)
    else:
        sim = mundy_nengo.Simulator(network)
    sim.run(0.1)

    for probe in probes:
        print "data for probe {} is {}".format(probe.label, sim.data[probe])



