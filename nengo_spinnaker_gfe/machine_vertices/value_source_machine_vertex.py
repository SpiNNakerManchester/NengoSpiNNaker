from enum import Enum

from nengo_spinnaker_gfe import constants, helpful_functions
from nengo_spinnaker_gfe.abstracts.abstract_accepts_multicast_signals import \
    AbstractAcceptsMulticastSignals
from nengo_spinnaker_gfe.graph_components.nengo_machine_vertex import \
    NengoMachineVertex
from pacman.executor.injection_decorator import inject_items
from pacman.model.resources import ResourceContainer, SDRAMResource

from spinn_front_end_common.abstract_models import AbstractHasAssociatedBinary
from spinn_front_end_common.abstract_models.impl import \
    MachineDataSpecableVertex
from spinn_front_end_common.interface.buffer_management import \
    recording_utilities
from spinn_front_end_common.interface.simulation import simulation_utilities
from spinn_front_end_common.utilities.utility_objs import ExecutableType
from spinn_front_end_common.utilities import constants as fec_constants

from spinn_utilities.overrides import overrides


class ValueSourceMachineVertex(
        NengoMachineVertex, MachineDataSpecableVertex,
        AbstractHasAssociatedBinary, AbstractAcceptsMulticastSignals):

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
        #
        "_update_period"

    ]

    DATA_REGIONS = Enum(
        value="DATA_REGIONS",
        names=[('SYSTEM', 0),
               ('UPDATE_PERIOD', 1),
               ('FILTERS', 2),
               ('FILTER_ROUTING', 3),
               ('RECORDING', 4)])

    N_RECORDING_REGIONS = 1
    UPDATE_PERIOD_ITEMS = 1

    def __init__(
            self, outgoing_partition_slice, n_machine_time_steps,
            update_period, minimum_buffer_sdram, receive_buffer_host,
            maximum_sdram_for_buffering, using_auto_pause_and_resume,
            receive_buffer_port):
        NengoMachineVertex.__init__(self)
        MachineDataSpecableVertex.__init__(self)
        AbstractHasAssociatedBinary.__init__(self)
        self._outgoing_partition_slice = outgoing_partition_slice
        self._n_machine_time_steps = n_machine_time_steps
        self._minimum_buffer_sdram = minimum_buffer_sdram
        self._maximum_sdram_for_buffering = maximum_sdram_for_buffering
        self._using_auto_pause_and_resume = using_auto_pause_and_resume
        self._receive_buffer_host = receive_buffer_host
        self._receive_buffer_port = receive_buffer_port
        self._update_period = update_period

    @inject_items({"n_machine_time_steps": "TotalMachineTimeSteps"})
    @overrides(MachineDataSpecableVertex.generate_machine_data_specification,
               additional_arguments=["n_machine_time_steps"])
    def generate_machine_data_specification(
            self, spec, placement, machine_graph, routing_info, iptags,
            reverse_iptags, machine_time_step, time_scale_factor,
            n_machine_time_steps):

        # reserve data regions
        self._reverse_memory_regions(spec)

        # add system region
        spec.switch_write_focus(self.DATA_REGIONS.SYSTEM.value)
        spec.write_array(simulation_utilities.get_simulation_header_array(
            self.get_binary_file_name(), machine_time_step,
            time_scale_factor))

        # add recording region
        spec.switch_write_focus(self.DATA_REGIONS.RECORDING.value)
        ip_tags = iptags.get_ip_tags_for_vertex(self)
        recorded_region_sizes = recording_utilities.get_recorded_region_sizes(
            self._get_buffered_sdram(self._input_slice, n_machine_time_steps),
            self._maximum_sdram_for_buffering)
        spec.write_array(recording_utilities.get_recording_header_array(
            recorded_region_sizes, self._time_between_requests,
            self._buffer_size_before_receive, ip_tags))

        # add update period region
        spec.switch_write_focus(self.DATA_REGIONS.UPDATE_PERIOD.value)
        spec.write_value(self._update_period is not None)

        # add filer region
        spec.switch_write_focus(self.DATA_REGIONS.FILTERS.value)
        self._write_filter_region(spec)

        # add routing region
        spec.switch_write_focus(self.DATA_REGIONS.ROUTING.value)
        self._write_routing_region(spec)

        spec.end_specification()

    def _write_filter_region(self, spec):
        pass

    def _write_routing_region(self, spec):
        pass

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
                self.N_RECORDING_REGIONS), label="recording")
        spec.reserve_memory_region(
            region=self.DATA_REGIONS.UPDATE_PERIOD.value,
            size=self._sdram_size_in_bytes_for_update_period_region(),
            label="update period")

    @overrides(AbstractHasAssociatedBinary.get_binary_start_type)
    def get_binary_start_type(self):
        return ExecutableType.USES_SIMULATION_INTERFACE

    @property
    @overrides(NengoMachineVertex.resources_required)
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

    @staticmethod
    def _sdram_size_in_bytes_for_update_period_region():
        return (ValueSourceMachineVertex.UPDATE_PERIOD_ITEMS *
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
                self.N_RECORDING_REGIONS) +
            # update period
            self._sdram_size_in_bytes_for_update_period_region())

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
