import run_spaun

import nengo_spinnaker_gfe.nengo_simulator as gfe_nengo
import nengo_spinnaker as mundy_nengo
USE_GFE = True

def create_model():
    args, max_probe_time, _ = run_spaun.set_defaults()
    model, _, _, _, _, _, _, _ = run_spaun.create_spaun_model(
        0, args, max_probe_time)
    return model, list(), dict()

if __name__ == '__main__':
    network, function_of_time, function_of_time_time_period = create_model()
    if USE_GFE:
        sim = gfe_nengo.NengoSimulator(network)
    else:
        sim = mundy_nengo.Simulator(network)
    sim.run(100)
