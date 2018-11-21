from spinn_front_end_common.utilities.database import DatabaseConnection

from nengo_spinnaker_gfe.io_connections.nengo_host_graph_updater import \
    NengoHostGraphUpdater

import nengo
from spinn_utilities.socket_address import SocketAddress


class NengoHostGraphUpdateBuilder(object):

    def __call__(
            self, host_network, time_step, machine_time_step_in_seconds,
            database_socket_addresses, notify_host_name, n_machine_time_steps):

        # Build the host simulator
        host_sim = nengo.Simulator(
            host_network, dt=machine_time_step_in_seconds)

        # wrap the host network into the updater, so that there's
        # functionality to call step
        nengo_host_graph_updater = NengoHostGraphUpdater(
            host_sim, machine_time_step_in_seconds, n_machine_time_steps)

        # create the auto pause and resume connection for start and pause
        auto_pause_and_resume_interface = DatabaseConnection(
            start_resume_callback_function=(
                nengo_host_graph_updater.start_resume),
            stop_pause_callback_function=(
                nengo_host_graph_updater.pause_stop),
            local_port=20000)

        # update the socket address
        socket_address = SocketAddress(
            listen_port=None,
            notify_host_name=notify_host_name,
            notify_port_no=auto_pause_and_resume_interface.local_port)
        database_socket_addresses.add(socket_address)

        # return as needed
        return (auto_pause_and_resume_interface, nengo_host_graph_updater,
                database_socket_addresses)
