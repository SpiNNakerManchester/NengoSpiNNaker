from nengo_spinnaker_gfe import helpful_functions
from nengo_spinnaker_gfe.machine_vertices.\
    sdp_transmitter_machine_vertex import SDPTransmitterMachineVertex
from spinnman.connections import ConnectionListener
from spinnman.connections.udp_packet_connections import SDPConnection
import numpy


class NengoSetUpLiveIO(object):
    """ handles the live output components of the machine graph
    
    """

    __slots__ = [
        "_machine_connections",
        "_connection_listener",
        "_placements"
    ]

    def __init__(self):
        self._machine_connections = list()
        self._connection_listener = list()
        self._placements = None

    def __call__(self, machine_graph, placements, tags):
        self._placements = placements

        for machine_vertex in machine_graph.vertices:
            if isinstance(machine_vertex, SDPTransmitterMachineVertex):
                self._process_sdp_transmitter_core(machine_vertex, tags)

        return self

    def _process_sdp_transmitter_core(self, machine_vertex, tags):
        ip_tags = tags.get_ip_tags_for_vertex(machine_vertex)
        if ip_tags is not None:
            for ip_tag in ip_tags:
                if ip_tag.traffic_identifier == \
                        machine_vertex.IPTAG_TRAFFIC_IDENTIFIER:
                    connection = SDPConnection(
                        local_host=ip_tag.ip_address,
                        local_port=ip_tag.port)
                    self._machine_connections.append(connection)
                    connection_listener = ConnectionListener(connection)
                    self._connection_listener.append(connection_listener)
                    connection_listener.add_callback(self._process_packet)

    def _process_packet(self, sdp_message):
        # Unpack the data, and store it as the input for the vertex
        numpy_array = helpful_functions.convert_s16_15_to_numpy_array(
            numpy.frombuffer(sdp_message.data, dtype=numpy.int32))
        vertex = self._placements.get_vertex_on_processor(
            sdp_message.sdp_header.source_chip_x,
            sdp_message.sdp_header.source_chip_y,
            sdp_message.sdp_header.source_cpu)
        vertex.output = numpy_array

    def stop(self):
        for connection in self._machine_connections:
            connection.close()
        for connection_listener in self._connection_listener:
            connection_listener.close()
