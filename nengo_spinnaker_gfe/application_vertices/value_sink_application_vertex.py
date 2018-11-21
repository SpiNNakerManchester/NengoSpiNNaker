import math
import numpy

from collections import defaultdict

from nengo_spinnaker_gfe import helpful_functions, constants
from nengo_spinnaker_gfe.abstracts.abstract_probeable import AbstractProbeable
from nengo_spinnaker_gfe.nengo_filters.\
    filter_and_routing_region_generator import FilterAndRoutingRegionGenerator
from nengo_spinnaker_gfe.overridden_mapping_algorithms.\
    nengo_partitioner import NengoPartitioner
from pacman.executor.injection_decorator import inject_items
from pacman.model.graphs.common import Slice
from spinn_utilities.overrides import overrides
from nengo_spinnaker_gfe.abstracts.abstract_nengo_application_vertex import \
    AbstractNengoApplicationVertex

from nengo_spinnaker_gfe.machine_vertices.value_sink_machine_vertex\
    import ValueSinkMachineVertex


class ValueSinkApplicationVertex(
        AbstractNengoApplicationVertex, AbstractProbeable):

    __slots__ = [
        # the number of atoms this vertex is processing
        '_size_in',
        '_sampling_interval',
        "_probeable_variables",
        "_is_recording_probeable_variable"
    ]

    MAX_WIDTH = 16.0

    def __init__(
            self, label, rng, size_in, seed, sampling_interval,
            simulation_time_in_seconds):
        AbstractNengoApplicationVertex.__init__(
            self, label=label, rng=rng, seed=seed)
        self._size_in = size_in
        self._probeable_variables = [
            constants.DECODER_OUTPUT_FLAG, constants.RECORD_OUTPUT_FLAG]
        self._is_recording_probeable_variable = {
            constants.DECODER_OUTPUT_FLAG: True,
            constants.RECORD_OUTPUT_FLAG: True}

        # Compute the sample period
        if sampling_interval is None:
            self._sampling_interval = 1
        else:
            self._sampling_interval = int(
                numpy.round(sampling_interval / simulation_time_in_seconds))

    @overrides(AbstractProbeable.can_probe_variable)
    def can_probe_variable(self, variable):
        return variable in self._probeable_variables

    @overrides(AbstractProbeable.set_probeable_variable)
    def set_probeable_variable(self, variable):
        raise Exception("these are set by default, so no point setting again")

    @overrides(AbstractProbeable.get_data_for_variable)
    def get_data_for_variable(
            self, variable, run_time, placements, graph_mapper, buffer_manager):

        if ((variable == constants.DECODER_OUTPUT_FLAG) or
                (variable == constants.RECORD_OUTPUT_FLAG)):

            # store app vertex data in a numpy array
            app_data = \
                numpy.zeros((int(run_time), self._size_in), dtype=numpy.float)

            # iterate though the machine verts and acquire data
            machine_vertices = graph_mapper.get_machine_vertices(self)
            for machine_vertex in machine_vertices:
                app_data[:, machine_vertex.input_slice.as_slice] = \
                    machine_vertex.get_data_for_recording_region(
                        run_time=run_time,
                        placement=placements.get_placement_of_vertex(
                            machine_vertex),
                        buffer_manager=buffer_manager,
                        sampling_interval=self._sampling_interval)

            return app_data
        else:
            raise Exception("not recognised probe variable")

    def get_possible_probeable_variables(self):
        return self._probeable_variables

    def is_set_probeable_variable(self, variable):
        if variable in self._probeable_variables:
            return True

    @property
    def size_in(self):
        return self._size_in

    @inject_items({
        "minimum_buffer_sdram": "MinBufferSize",
        "maximum_sdram_for_buffering": "MaxSinkBuffingSize",
        "using_auto_pause_and_resume": "UsingAutoPauseAndResume",
        "receive_buffer_host": "ReceiveBufferHost",
        "receive_buffer_port": "ReceiveBufferPort",
        "operator_graph": "NengoOperatorGraph",
        "time_between_requests": "TimeBetweenRequests",
        "buffer_size_before_receive": "BufferSizeBeforeReceive"})
    @overrides(
        AbstractNengoApplicationVertex.create_machine_vertices,
        additional_arguments={
            "minimum_buffer_sdram", "maximum_sdram_for_buffering",
            "using_auto_pause_and_resume", "receive_buffer_host",
            "receive_buffer_port", "operator_graph", "time_between_requests",
            "buffer_size_before_receive"})
    def create_machine_vertices(
            self, resource_tracker, machine_graph, graph_mapper,
            minimum_buffer_sdram, maximum_sdram_for_buffering,
            using_auto_pause_and_resume, receive_buffer_host,
            receive_buffer_port, operator_graph, time_between_requests,
            buffer_size_before_receive):
        # Make sufficient vertices to ensure that each has a size_in of less
        # than max_width.

        n_vertices = int(math.ceil((self._size_in / self.MAX_WIDTH)))

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
                Slice(0, self._size_in - 1), n_vertices):
            machine_vertex = ValueSinkMachineVertex(
                input_slice=input_slice,
                minimum_buffer_sdram=minimum_buffer_sdram,
                maximum_sdram_for_buffering=maximum_sdram_for_buffering,
                using_auto_pause_and_resume=using_auto_pause_and_resume,
                receive_buffer_host=receive_buffer_host,
                receive_buffer_port=receive_buffer_port,
                input_filters=input_filters, input_n_keys=inputs_n_keys,
                time_between_requests=time_between_requests,
                buffer_size_before_receive=buffer_size_before_receive,
                label="{} for {}".format(input_slice, self._label))
            resource_tracker.allocate_resources(
                machine_vertex.resources_required)
            machine_graph.add_vertex(machine_vertex)
            graph_mapper.add_vertex_mapping(
                machine_vertex=machine_vertex, application_vertex=self)
