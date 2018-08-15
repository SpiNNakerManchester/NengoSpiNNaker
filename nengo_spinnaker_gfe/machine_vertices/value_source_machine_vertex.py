from enum import Enum

from pacman.model.graphs.machine import MachineVertex

from spinn_front_end_common.abstract_models import AbstractHasAssociatedBinary
from spinn_front_end_common.abstract_models.impl import \
    MachineDataSpecableVertex
from spinn_front_end_common.utilities.utility_objs import ExecutableType

from spinn_utilities.overrides import overrides


class ValueSourceMachineVertex(
        MachineVertex, MachineDataSpecableVertex, AbstractHasAssociatedBinary):

    __slots__ = [
        "_resources",
        "_outgoing_partition_slice"

    ]

    DATA_REGIONS = Enum(
        value="DATA_REGIONS",
        names=[('SYSTEM', 0),
               ('FILTERS', 1),
               ('FILTER_ROUTING', 2),
               ('RECORDING', 3)])

    def __init__(self, outgoing_partition_slice, resources):
        MachineVertex.__init__(self)
        MachineDataSpecableVertex.__init__(self)
        AbstractHasAssociatedBinary.__init__(self)
        self._outgoing_partition_slice = outgoing_partition_slice
        self._resources = resources

    @overrides(MachineDataSpecableVertex.generate_machine_data_specification)
    def generate_machine_data_specification(
            self, spec, placement, machine_graph, routing_info, iptags,
            reverse_iptags, machine_time_step, time_scale_factor):
        pass

    @overrides(AbstractHasAssociatedBinary.get_binary_start_type)
    def get_binary_start_type(self):
        return ExecutableType.USES_SIMULATION_INTERFACE

    @property
    @overrides(MachineVertex.resources_required)
    def resources_required(self):
        return self._resources

    @overrides(AbstractHasAssociatedBinary.get_binary_file_name)
    def get_binary_file_name(self):
        return "value_source.aplx"
