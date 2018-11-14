import logging
import math
import numpy
from nengo.utils import numpy as nengo_numpy

from nengo.processes import Process
from nengo_spinnaker_gfe import constants, helpful_functions

from nengo_spinnaker_gfe.machine_vertices.value_source_machine_vertex import \
    ValueSourceMachineVertex
from nengo_spinnaker_gfe.overridden_mapping_algorithms.\
    nengo_partitioner import NengoPartitioner
from pacman.executor.injection_decorator import inject_items
from pacman.model.graphs.common import Slice
from spinn_utilities.log import FormatAdapter
from spinn_utilities.overrides import overrides
from nengo_spinnaker_gfe.abstracts. \
    abstract_nengo_application_vertex import \
    AbstractNengoApplicationVertex

from nengo_spinnaker_gfe.abstracts.abstract_probeable import AbstractProbeable

logger = FormatAdapter(logging.getLogger(__name__))


class ValueSourceApplicationVertex(
        AbstractNengoApplicationVertex, AbstractProbeable):

    __slots__ = [
        #
        '_nengo_output_function',
        #
        '_size_out',
        #
        '_update_period',
        #
        '_recording_of',
        #
        "_output_data"
    ]

    PROBEABLE_OUTPUT = 'output'
    PROBEABLE_ATTRIBUTES = [PROBEABLE_OUTPUT]

    MAX_CHANNELS_PER_MACHINE_VERTEX = 10.0
    SYSTEM_REGION_DATA_ITEMS = 6

    n_value_source_machine_vertices = 0

    def __init__(
            self, label, rng, nengo_output_function, size_out, update_period,
            utilise_extra_core_for_output_types_probe, seed):
        AbstractNengoApplicationVertex.__init__(
            self, label=label, rng=rng, seed=seed)
        self._nengo_output_function = nengo_output_function
        self._size_out = size_out
        self._update_period = update_period
        self._recording_of = dict()
        self._output_data = None

        if not utilise_extra_core_for_output_types_probe:
            for attribute in self.PROBEABLE_ATTRIBUTES:
                self._recording_of[attribute] = False

    @overrides(AbstractProbeable.set_probeable_variable)
    def set_probeable_variable(self, variable):
        if self.can_probe_variable(variable):
            self._recording_of[variable] = not self._recording_of[variable]

    @overrides(AbstractProbeable.can_probe_variable)
    def can_probe_variable(self, variable):
        return variable in self._recording_of

    @overrides(AbstractProbeable.get_data_for_variable)
    def get_data_for_variable(
            self,  variable, run_time, placements, graph_mapper,
            buffer_manager):
        pass

    @overrides(AbstractProbeable.get_possible_probeable_variables)
    def get_possible_probeable_variables(self):
        return self.PROBEABLE_ATTRIBUTES

    @overrides(AbstractProbeable.is_set_probeable_variable)
    def is_set_probeable_variable(self, variable):
        return self._recording_of[variable]

    @property
    def nengo_output_function(self):
        return self._nengo_output_function

    @property
    def size_out(self):
        return self._size_out

    @property
    def update_period(self):
        return self._update_period

    @inject_items({
        "operator_graph": "NengoOperatorGraph",
        "n_machine_time_steps": "TotalMachineTimeSteps",
        "minimum_buffer_sdram": "MinBufferSize",
        "maximum_sdram_for_buffering": "MaxSinkBuffingSize",
        "using_auto_pause_and_resume": "UsingAutoPauseAndResume",
        "receive_buffer_host": "ReceiveBufferHost",
        "receive_buffer_port": "ReceiveBufferPort",
        "current_time_step": "FirstMachineTimeStep",
        "machine_time_step_in_seconds": "MachineTimeStepInSeconds"})
    @overrides(
        AbstractNengoApplicationVertex.create_machine_vertices,
        additional_arguments=[
            "operator_graph", "n_machine_time_steps", "minimum_buffer_sdram",
            "maximum_sdram_for_buffering", "using_auto_pause_and_resume",
            "receive_buffer_host", "receive_buffer_port",
            "machine_time_step_in_seconds", "n_machine_time_steps",
            "current_time_step"])
    def create_machine_vertices(
            self, resource_tracker, machine_graph, graph_mapper, operator_graph,
            n_machine_time_steps, minimum_buffer_sdram,
            machine_time_step_in_seconds, maximum_sdram_for_buffering,
            using_auto_pause_and_resume, receive_buffer_host,
            receive_buffer_port, current_time_step):

        # only generate the output data once.
        if self._output_data is None:
            self._output_data = self._generate_output_data(
                operator_graph, n_machine_time_steps,
                machine_time_step_in_seconds, current_time_step)

        outgoing_partitions = \
            operator_graph.get_outgoing_edge_partitions_starting_at_vertex(self)
        n_machine_verts = int(math.ceil(
            len(outgoing_partitions) / self.MAX_CHANNELS_PER_MACHINE_VERTEX))
        vertex_partition_slices = NengoPartitioner.divide_slice(
            Slice(0, len(outgoing_partitions)), n_machine_verts)

        for vertex_partition_slice in vertex_partition_slices:
            recording_output = False
            if self.PROBEABLE_OUTPUT in self._recording_of.keys():
                recording_output = self._recording_of[self.PROBEABLE_OUTPUT]

            this_cores_matrix = \
                helpful_functions.convert_matrix_to_machine_vertex_level(
                    self._output_data, vertex_partition_slice.as_slice,
                    constants.MATRIX_CONVERSION_PARTITIONING.COLUMNS)

            machine_vertex = ValueSourceMachineVertex(
                vertex_partition_slice,
                self._update_period, minimum_buffer_sdram,
                receive_buffer_host, maximum_sdram_for_buffering,
                using_auto_pause_and_resume, receive_buffer_port,
                recording_output, this_cores_matrix,
                label="{} for {}".format(vertex_partition_slice, self._label))

            # tracker for random back off
            ValueSourceApplicationVertex.n_value_source_machine_vertices += 1

            resource_tracker.allocate_resources(
                machine_vertex.resources_required)
            machine_graph.add_vertex(machine_vertex)
            graph_mapper.add_vertex_mapping(
                machine_vertex=machine_vertex, application_vertex=self)

    def _generate_output_data(
            self, app_graph, n_machine_time_steps, machine_time_step,
            current_time_step):

        if self._update_period is not None:
            max_n = min(
                n_machine_time_steps,
                int(numpy.ceil(self._update_period / machine_time_step)))
        else:
            max_n = n_machine_time_steps

        ts = (numpy.arange(current_time_step, n_machine_time_steps + max_n) *
              machine_time_step)
        if callable(self._nengo_output_function):
            values = numpy.array([self.function(t) for t in ts])
        elif isinstance(self._nengo_output_function, Process):
            values = self._nengo_output_function.run_steps(
                max_n, d=self.size_out, dt=machine_time_step)
        else:
            values = numpy.array([self._nengo_output_function for t in ts])

        # Ensure that the values can be sliced, regardless of how they were
        # generated.
        values = nengo_numpy.array(values, min_dims=2)

        # Compute the output for each connection
        outputs = []
        for outgoing_partition in app_graph.\
                get_outgoing_edge_partitions_starting_at_vertex(self):
            if (outgoing_partition.identifier.source_port ==
                    constants.OUTPUT_PORT.STANDARD):
                output = []
                transmission_parameter = \
                    outgoing_partition.identifier.transmission_parameter
                transform = transmission_parameter.full_transform(
                    slice_in=False, slice_out=False)
                keep = numpy.any(transform != 0.0, axis=1)
                transform = transform[keep]

                # For each f(t) for the next set of simulations we calculate the
                # output at the end of the connection.  To do this we first
                #  apply the pre-slice, then the function and then the
                # post-slice.
                for out_value in values:

                    # Apply the pre-slice
                    out_value = out_value[transmission_parameter.pre_slice]

                    # Apply the function on the connection, if there is one.
                    if transmission_parameter.parameter_function is not None:
                        out_value = numpy.asarray(
                            transmission_parameter.parameter_function(
                                out_value), dtype=float)

                    output.append(numpy.dot(transform, out_value.T))
                outputs.append(numpy.array(output).reshape(max_n, -1))

        # Combine all of the output values to form a large matrix which we can
        # dump into memory.
        output_matrix = numpy.hstack(outputs)
        return output_matrix
