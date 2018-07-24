import nengo_spinnaker


class NengoCreateHostSimulator(object):

    def __call__(self, host_network, machine_time_step_in_seconds):

        # Build the host simulator
        host_sim = nengo_spinnaker.Simulator(
            host_network, dt=machine_time_step_in_seconds)
        return host_sim
