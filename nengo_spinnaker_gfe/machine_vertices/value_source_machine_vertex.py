from enum import Enum

from nengo_spinnaker_gfe import constants
from nengo_spinnaker_gfe.abstracts.abstract_accepts_multicast_signals import \
    AbstractAcceptsMulticastSignals
from pacman.model.graphs.machine import MachineVertex
from pacman.model.resources import ResourceContainer, SDRAMResource

from spinn_front_end_common.abstract_models import AbstractHasAssociatedBinary
from spinn_front_end_common.abstract_models.impl import \
    MachineDataSpecableVertex
from spinn_front_end_common.interface.buffer_management import \
    recording_utilities
from spinn_front_end_common.utilities.utility_objs import ExecutableType

from spinn_utilities.overrides import overrides


class ValueSourceMachineVertex(
        MachineVertex, MachineDataSpecableVertex, AbstractHasAssociatedBinary,
        AbstractAcceptsMulticastSignals):

    __slots__ = [
        "_n_machine_time_steps",
        "_outgoing_partition_slice",
        "_minimum_buffer_sdram",
        #
        "_maximum_sdram_for_buffering",
        #
        "_using_auto_pause_and_resume",
        #
        "_receive_buffer_host",
        #
        "_receive_buffer_port",

    ]

    DATA_REGIONS = Enum(
        value="DATA_REGIONS",
        names=[('SYSTEM', 0),
               ('FILTERS', 1),
               ('FILTER_ROUTING', 2),
               ('RECORDING', 3)])

    N_RECORDING_REGIONS = 1

    def __init__(
            self, outgoing_partition_slice, n_machine_time_steps,
            minimum_buffer_sdram, receive_buffer_host,
            maximum_sdram_for_buffering, using_auto_pause_and_resume,
            receive_buffer_port):
        MachineVertex.__init__(self)
        MachineDataSpecableVertex.__init__(self)
        AbstractHasAssociatedBinary.__init__(self)
        self._outgoing_partition_slice = outgoing_partition_slice
        self._n_machine_time_steps = n_machine_time_steps
        self._minimum_buffer_sdram = minimum_buffer_sdram
        self._maximum_sdram_for_buffering = maximum_sdram_for_buffering
        self._using_auto_pause_and_resume = using_auto_pause_and_resume
        self._receive_buffer_host = receive_buffer_host
        self._receive_buffer_port = receive_buffer_port

    @overrides(MachineDataSpecableVertex.generate_machine_data_specification)
    def generate_machine_data_specification(
            self, spec, placement, machine_graph, routing_info, iptags,
            reverse_iptags, machine_time_step, time_scale_factor):
        self._reverse_memory_regions(spec)

    def _reverse_memory_regions(self, spec):
        spec.reserve_memory_region(
            self.DATA_REGIONS.SYSTEM.value,
            fec_constants.SYSTEM_BYTES_REQUIREMENT,
            label="system region")
        spec.reserve_memory_region(
            self.DATA_REGIONS.ROUTING.value,
            helpful_functions.sdram_size_in_bytes_for_routing_region(
                self._input_n_keys), label="routing region")
        spec.reserve_memory_region(
            self.DATA_REGIONS.FILTERS.value,
            helpful_functions.sdram_size_in_bytes_for_filter_region(
                self._input_filters), label="filter region")
        spec.reserve_memory_region(
            region=self.DATA_REGIONS.RECORDING.value,
            size=recording_utilities.get_recording_header_size(
                self.N_RECORDING_REGIONS))

    @overrides(AbstractHasAssociatedBinary.get_binary_start_type)
    def get_binary_start_type(self):
        return ExecutableType.USES_SIMULATION_INTERFACE

    @property
    @overrides(MachineVertex.resources_required)
    def resources_required(self):
        return self.generate_static_resources(
            self._outgoing_partition_slice, self._n_machine_time_steps)

    @staticmethod
    def _sdram_size_in_bytes_for_key_region(n_atoms):
        return n_atoms * constants.BYTES_PER_KEY

    @staticmethod
    def _sdram_size_in_bytes_for_output_region(n_atoms, n_machine_time_steps):
        return ((n_atoms * n_machine_time_steps) *
                constants.BYTE_TO_WORD_MULTIPLIER)

    def generate_static_resources(
            self, outgoing_partition_slice, n_machine_time_steps):
        sdram = (
            # system region
            (self.SYSTEM_REGION_DATA_ITEMS *
             constants.BYTE_TO_WORD_MULTIPLIER) +
            # key region
            self._sdram_size_in_bytes_for_key_region(
                outgoing_partition_slice.n_atoms) +
            # output region
            self._sdram_size_in_bytes_for_output_region(
                outgoing_partition_slice.n_atoms, n_machine_time_steps) +
            # recordings
            recording_utilities.get_recording_header_size(
                self.N_RECORDING_REGIONS))

        # the basic sdram
        basic_res = ResourceContainer(sdram=SDRAMResource(sdram))

        # handle buffered recording res
        recording_sizes = recording_utilities.get_recording_region_sizes(
            [self._get_buffered_sdram(self._input_slice, n_machine_time_steps)],
            self._minimum_buffer_sdram, self._maximum_sdram_for_buffering,
            self._using_auto_pause_and_resume)
        basic_res.extend(recording_utilities.get_recording_resources(
            recording_sizes, self._receive_buffer_host,
            self._receive_buffer_port))

        return basic_res

    @overrides(AbstractHasAssociatedBinary.get_binary_file_name)
    def get_binary_file_name(self):
        return "value_source.aplx"

    @overrides(AbstractAcceptsMulticastSignals.accepts_multicast_signals)
    def accepts_multicast_signals(self, transmission_params):
        return True
