import numpy as np
import nengo

import nengo_spinnaker_gfe.nengo_simulator as gfe_nengo
import nengo_spinnaker as mundy_nengo
USE_GFE = True

def create_model():

    model = nengo.Network()
    with model:
        stimulus = nengo.Node(lambda t: (np.sin(t), np.cos(t)))
        ens = nengo.Ensemble(n_neurons=1000, dimensions=2)
        nengo.Connection(stimulus, ens)
    return model, list(), dict()

if __name__ == '__main__':
    network, function_of_time, function_of_time_time_period = create_model()
    if USE_GFE:
        sim = gfe_nengo.NengoSimulator(network)
    else:
        sim = mundy_nengo.Simulator(network)
    sim.run(100)
