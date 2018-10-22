import threading

import numpy
from collections import defaultdict

from nengo_spinnaker_gfe import constants, helpful_functions
from nengo_spinnaker_gfe.nengo_filters.\
    filter_and_routing_region_generator import \
    FilterAndRoutingRegionGenerator
from pacman.executor.injection_decorator import inject_items
from spinn_utilities.overrides import overrides
from nengo_spinnaker_gfe. \
    abstracts.abstract_nengo_application_vertex import \
    AbstractNengoApplicationVertex
from nengo_spinnaker_gfe.machine_vertices. \
    sdp_transmitter_machine_vertex import \
    SDPTransmitterMachineVertex

from nengo_spinnaker_gfe.nengo_implicit_interfaces.nengo_live_input_interface\
    import NengoLiveInputInterface


class SDPTransmitterApplicationVertex(
        AbstractNengoApplicationVertex, NengoLiveInputInterface):
    """
    LPG equiv vertex (but includes filtering and some routing stuff)
    """

    __slots__ = [
        #
        '_size_in',
        #
        '_vertex',
        #
        '_output',
        #
        '_lock'
    ]

    def __init__(self, size_in, label, rng, seed):
        AbstractNengoApplicationVertex.__init__(
            self, label=label, rng=rng, seed=seed)
        NengoLiveInputInterface.__init__(self)
        self._size_in = size_in
        self._vertex = None
        self._output = numpy.zeros(self._size_out)
        self._lock = threading.Lock()

    @property
    def size_in(self):
        return self._size_in

    @overrides(NengoLiveInputInterface.output)
    def output(self, t):
        """This is a interface used by the nengo_spinnaker_gfe
        """
        with self._lock:
            return self._output

    def set_output(self, new_output):
        with self._lock:
            self._output = new_output

    @inject_items({
        "operator_graph": "NengoOperatorGraph",
        "ip_address": "IPAddress"})
    @overrides(AbstractNengoApplicationVertex.create_machine_vertices,
               additional_arguments=["operator_graph", "ip_address"])
    def create_machine_vertices(
            self, resource_tracker, machine_graph, graph_mapper,
            operator_graph, ip_address):
        """ Create vertices that will simulate the SDPTransmitter.
        
        :param resource_tracker: 
        :param machine_graph: 
        :param graph_mapper: 
        :param operator_graph:
        :param ip_address:
        :return: 
        """
        incoming_standard_edges = \
            helpful_functions.locate_all_incoming_edges_of_type(
                operator_graph, self, constants.INPUT_PORT.STANDARD)

        # generate filters and mc key count
        inputs_n_keys = 0
        input_filters = defaultdict(list)
        for input_edge in incoming_standard_edges:
            FilterAndRoutingRegionGenerator.add_filters(
                input_filters, input_edge,
                operator_graph.get_outgoing_partition_for_edge(input_edge),
                minimise=True)
            inputs_n_keys += 1

        machine_vertex = SDPTransmitterMachineVertex(
            self._size_in, input_filters, inputs_n_keys, ip_address,
            self._label)
        resource_tracker.allocate_resources(machine_vertex.resources_required)
        machine_graph.add_vertex(machine_vertex)
        graph_mapper.add_vertex_mapping(
            machine_vertex=machine_vertex, application_vertex=self)
