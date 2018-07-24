import numpy as np

import nengo

import nengo_spinnaker_gfe.nengo_simulator as gfe_nengo
import nengo_spinnaker as mundy_nengo
USE_GFE = True

def create_model():
    dimension = 9

    model = nengo.Network()
    with model:
        for i in range(dimension):
            def waves(t, i=i):
                return np.sin(t + np.arange(i + 1) * 2 * np.pi / (i + 1))
            _ = nengo.Node(waves)
    return model, list(), dict()

if __name__ == '__main__':
    network, function_of_time, function_of_time_time_period = create_model()
    if USE_GFE:
        sim = gfe_nengo.NengoSimulator(network)
    else:
        sim = mundy_nengo.Simulator(network)
    sim.run(100)
