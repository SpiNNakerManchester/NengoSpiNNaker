import logging
import math

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
        '_recording_of'
        #
        '_probeable_variables'
    ]

    PROBEABLE_ATTRIBUTES = ['output']

    MAX_CHANNELS_PER_MACHINE_VERTEX = 10
    SYSTEM_REGION_DATA_ITEMS = 6

    def __init__(
            self, label, rng, nengo_output_function, size_out, update_period,
            utilise_extra_core_for_output_types_probe, seed):
        AbstractNengoApplicationVertex.__init__(
            self, label=label, rng=rng, seed=seed)
        self._nengo_output_function = nengo_output_function
        self._size_out = size_out
        self._update_period = update_period
        self._recording_of = dict()

        self._probeable_variables = dict()
        if not utilise_extra_core_for_output_types_probe:
            for attribute in self.PROBEABLE_ATTRIBUTES:
                self._recording_of[attribute] = False

    def set_probeable_variable(self, variable):
        if self.can_probe_variable(variable):
            self._recording_of[variable] = not self._recording_of[variable]

    def can_probe_variable(self, variable):
        return variable in self._recording_of

    def get_data_for_variable(self, variable):
        pass

    @property
    def nengo_output_function(self):
        return self._nengo_output_function

    @property
    def size_out(self):
        return self._size_out

    @property
    def update_period(self):
        return self._update_period

    @property
    def recording_of(self):
        return self._recording_of

    @inject_items({"operator_graph": "NengoOperatorGraph",
                   "n_machine_time_steps": "TotalMachineTimeSteps"})
    @overrides(
        AbstractNengoApplicationVertex.create_machine_vertices,
        additional_arguments=["operator_graph", "n_machine_time_steps"])
    def create_machine_vertices(
            self, resource_tracker, machine_graph, graph_mapper, operator_graph,
            n_machine_time_steps):
        outgoing_partitions = \
            operator_graph.get_outgoing_edge_partitions_starting_at_vertex(self)
        n_machine_verts = int(math.ceil(
            len(outgoing_partitions) / self.MAX_CHANNELS_PER_MACHINE_VERTEX))
        vertex_partition_slices = NengoPartitioner.divide_slice(
            Slice(0, len(outgoing_partitions)), n_machine_verts)



        for vertex_partition_slice in vertex_partition_slices:



            machine_vertex = ValueSourceMachineVertex(
                vertex_partition_slice, n_machine_time_steps)
            resource_tracker.allocate_resources(
                machine_vertex.resources_required)
            machine_graph.add_vertex(machine_vertex)
            graph_mapper.add_vertex_mapping(
                machine_vertex=machine_vertex, application_vertex=self)
