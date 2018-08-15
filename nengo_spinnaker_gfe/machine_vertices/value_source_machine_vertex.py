from enum import Enum

from nengo_spinnaker_gfe import constants
from nengo_spinnaker_gfe.abstracts.abstract_accepts_multicast_signals import \
    AcceptsMulticastSignals
from pacman.model.graphs.machine import MachineVertex
from pacman.model.resources import ResourceContainer, SDRAMResource

from spinn_front_end_common.abstract_models import AbstractHasAssociatedBinary
from spinn_front_end_common.abstract_models.impl import \
    MachineDataSpecableVertex
from spinn_front_end_common.utilities.utility_objs import ExecutableType

from spinn_utilities.overrides import overrides


class ValueSourceMachineVertex(
        MachineVertex, MachineDataSpecableVertex, AbstractHasAssociatedBinary,
        AcceptsMulticastSignals):

    __slots__ = [
        "_n_machine_time_steps",
        "_outgoing_partition_slice"

    ]

    DATA_REGIONS = Enum(
        value="DATA_REGIONS",
        names=[('SYSTEM', 0),
               ('FILTERS', 1),
               ('FILTER_ROUTING', 2),
               ('RECORDING', 3)])

    def __init__(self, outgoing_partition_slice, n_machine_time_steps):
        MachineVertex.__init__(self)
        MachineDataSpecableVertex.__init__(self)
        AbstractHasAssociatedBinary.__init__(self)
        self._outgoing_partition_slice = outgoing_partition_slice
        self._n_machine_time_steps = n_machine_time_steps

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
        return self.generate_static_resources(
            self._outgoing_partition_slice, self._n_machine_time_steps)

    def generate_static_resources(
            self, outgoing_partition_slice, n_machine_time_steps):
        sdram = (
            # system region
            (self.SYSTEM_REGION_DATA_ITEMS *
             constants.BYTE_TO_WORD_MULTIPLIER) +
            # key region
            (outgoing_partition_slice.n_atoms * constants.BYTES_PER_KEY) +
            # output region
            (outgoing_partition_slice.n_atoms * n_machine_time_steps) *
            constants.BYTE_TO_WORD_MULTIPLIER)

        return ResourceContainer(sdram=SDRAMResource(sdram))

    @overrides(AbstractHasAssociatedBinary.get_binary_file_name)
    def get_binary_file_name(self):
        return "value_source.aplx"

    @overrides(AcceptsMulticastSignals.accepts_multicast_signals)
    def accepts_multicast_signals(self, transmission_params):
        return True
