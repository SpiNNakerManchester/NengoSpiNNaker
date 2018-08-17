import math

from collections import defaultdict

from nengo_spinnaker_gfe import helpful_functions, constants
from nengo_spinnaker_gfe.nengo_filters.filter_and_routing_region_generator import \
    FilterAndRoutingRegionGenerator
from nengo_spinnaker_gfe.overridden_mapping_algorithms.\
    nengo_partitioner import NengoPartitioner
from pacman.executor.injection_decorator import inject_items
from pacman.model.graphs.common import Slice
from spinn_utilities.overrides import overrides
from nengo_spinnaker_gfe.abstracts.abstract_nengo_application_vertex import \
    AbstractNengoApplicationVertex

from nengo_spinnaker_gfe.machine_vertices.value_sink_machine_vertex\
    import ValueSinkMachineVertex


class ValueSinkApplicationVertex(AbstractNengoApplicationVertex):

    __slots__ = [
        # the number of atoms this vertex is processing
        '_size_in'
    ]

    MAX_WIDTH = 16

    def __init__(self, label, rng, size_in, seed):
        AbstractNengoApplicationVertex.__init__(
            self, label=label, rng=rng, seed=seed)
        self._size_in = size_in

    @property
    def size_in(self):
        return self._size_in

    @inject_items({
        "minimum_buffer_sdram": "MinBufferSize",
        "maximum_sdram_for_buffering": "MaxSinkBuffingSize",
        "using_auto_pause_and_resume": "UsingAutoPauseAndResume",
        "receive_buffer_host": "ReceiveBufferHost",
        "receive_buffer_port": "ReceiveBufferPort",
        "operator_graph": "NengoOperatorGraph"})
    @overrides(
        AbstractNengoApplicationVertex.create_machine_vertices,
        additional_arguments={
            "minimum_buffer_sdram", "maximum_sdram_for_buffering",
            "using_auto_pause_and_resume", "receive_buffer_host",
            "receive_buffer_port", "operator_graph"})
    def create_machine_vertices(
            self, resource_tracker, machine_graph, graph_mapper,
            minimum_buffer_sdram, maximum_sdram_for_buffering,
            using_auto_pause_and_resume, receive_buffer_host,
            receive_buffer_port, operator_graph):
        # Make sufficient vertices to ensure that each has a size_in of less
        # than max_width.

        n_vertices = int(math.ceil((self._size_in // self.MAX_WIDTH)))
        if n_vertices == 0:
            n_vertices = 1

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

        for input_slice in NengoPartitioner.divide_slice(
                Slice(0, self._size_in), n_vertices):
            machine_vertex = ValueSinkMachineVertex(
                input_slice=input_slice,
                minimum_buffer_sdram=minimum_buffer_sdram,
                maximum_sdram_for_buffering=maximum_sdram_for_buffering,
                using_auto_pause_and_resume=using_auto_pause_and_resume,
                receive_buffer_host=receive_buffer_host,
                receive_buffer_port=receive_buffer_port,
                input_filters=input_filters, input_n_keys=inputs_n_keys)
            resource_tracker.allocate_resources(
                machine_vertex.resources_required)
            machine_graph.add_vertex(machine_vertex)
            graph_mapper.add_vertex_mapping(
                machine_vertex=machine_vertex, application_vertex=self)
