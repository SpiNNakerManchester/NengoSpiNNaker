
import nengo.spa as spa

import nengo_spinnaker_gfe.nengo_simulator as gfe_nengo
import nengo_spinnaker as mundy_nengo
USE_GFE = True


def create_model():
    D = 16

    model = spa.SPA(seed=1)
    with model:
        model.a = spa.State(dimensions=D)
        model.b = spa.State(dimensions=D)
        model.c = spa.State(dimensions=D)
        model.cortical = spa.Cortical(spa.Actions(
            'c = a+b',
            ))

        model.input = spa.Input(
            a='A',
            b=(lambda t: 'C*~A' if (t%0.1 < 0.05) else 'D*~A'),
            )
    return model, list(), dict()

if __name__ == '__main__':
    network, function_of_time, function_of_time_time_period = create_model()
    if USE_GFE:
        sim = gfe_nengo.NengoSimulator(network)
    else:
        sim = mundy_nengo.Simulator(network)
    sim.run(100)
