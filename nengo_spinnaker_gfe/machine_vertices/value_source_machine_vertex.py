from enum import Enum
import numpy

from data_specification.enums import DataType

from nengo_spinnaker_gfe import constants, helpful_functions
from nengo_spinnaker_gfe.abstracts.abstract_nengo_machine_vertex import \
    AbstractNengoMachineVertex
from nengo_spinnaker_gfe.abstracts.\
    abstract_transmits_multicast_signals import \
    AbstractTransmitsMulticastSignals
from pacman.executor.injection_decorator import inject_items
from pacman.model.resources import ResourceContainer, SDRAMResource
from spinn_front_end_common.abstract_models import AbstractHasAssociatedBinary
from spinn_front_end_common.abstract_models.impl import \
    MachineDataSpecableVertex
from spinn_front_end_common.interface.buffer_management import \
    recording_utilities
from spinn_front_end_common.interface.simulation import simulation_utilities
from spinn_front_end_common.utilities import constants as fec_constants
from spinn_front_end_common.utilities.utility_objs import ExecutableType
from spinn_utilities.overrides import overrides


class ValueSourceMachineVertex(
        AbstractNengoMachineVertex, MachineDataSpecableVertex,
        AbstractHasAssociatedBinary, AbstractTransmitsMulticastSignals):

    __slots__ = [
        #
        "_n_machine_time_steps",
        #
        "_outgoing_partition_slice",
        #
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
        "_update_period",
        #
        "_is_recording_output",
        #
        "_output_data"
    ]

    DATA_REGIONS = Enum(
        value="DATA_REGIONS",
        names=[('SYSTEM', 0),
               ('UPDATE_PERIOD', 1),
               ('OUTPUT_REGION', 2),
               ('KEY_REGION', 3),
               ('RECORDING', 4)])

    UPDATE_PERIOD_ITEMS = 1
    SDRAM_RECORDING_SDRAM_PER_ATOM = 4
    N_RECORDING_REGIONS = 1

    def __init__(
            self, outgoing_partition_slice, n_machine_time_steps,
            update_period, minimum_buffer_sdram, receive_buffer_host,
            maximum_sdram_for_buffering, using_auto_pause_and_resume,
            receive_buffer_port, is_recording_output,
            this_cores_matrix, label):
        AbstractNengoMachineVertex.__init__(self, label=label)
        MachineDataSpecableVertex.__init__(self)
        AbstractHasAssociatedBinary.__init__(self)
        AbstractTransmitsMulticastSignals.__init__(self)
        self._outgoing_partition_slice = outgoing_partition_slice
        self._n_machine_time_steps = n_machine_time_steps
        self._minimum_buffer_sdram = minimum_buffer_sdram
        self._maximum_sdram_for_buffering = maximum_sdram_for_buffering
        self._using_auto_pause_and_resume = using_auto_pause_and_resume
        self._receive_buffer_host = receive_buffer_host
        self._receive_buffer_port = receive_buffer_port
        self._update_period = update_period
        self._is_recording_output = is_recording_output
        self._output_data = this_cores_matrix

    @inject_items({"n_machine_time_steps": "TotalMachineTimeSteps",
                   "current_time_step": "FirstMachineTimeStep"})
    @overrides(
        MachineDataSpecableVertex.generate_machine_data_specification,
        additional_arguments=["n_machine_time_steps", "current_time_step"])
    def generate_machine_data_specification(
            self, spec, placement, machine_graph, routing_info, iptags,
            reverse_iptags, machine_time_step, time_scale_factor,
            n_machine_time_steps, current_time_step):

        # reserve data regions
        self._reverse_memory_regions(spec, self._output_data, machine_graph)

        # add system region
        spec.switch_write_focus(self.DATA_REGIONS.SYSTEM.value)
        spec.write_array(simulation_utilities.get_simulation_header_array(
            self.get_binary_file_name(), machine_time_step,
            time_scale_factor))

        # add recording region
        if self._is_recording_output:
            spec.switch_write_focus(self.DATA_REGIONS.RECORDING.value)
            ip_tags = iptags.get_ip_tags_for_vertex(self)
            recorded_region_sizes = \
                recording_utilities.get_recorded_region_sizes(
                    self._get_buffered_sdram(n_machine_time_steps),
                    self._maximum_sdram_for_buffering)
            spec.write_array(recording_utilities.get_recording_header_array(
                recorded_region_sizes, self._time_between_requests,
                self._buffer_size_before_receive, ip_tags))

        # add update period region
        spec.switch_write_focus(self.DATA_REGIONS.UPDATE_PERIOD.value)
        spec.write_value(self._update_period is not None)

        # add output region
        spec.switch_write_focus(self.DATA_REGIONS.OUTPUT_REGION.value)
        spec.write_array(helpful_functions.convert_numpy_array_to_s16_15(
            self._output_data))

        # add routing region
        spec.switch_write_focus(self.DATA_REGIONS.KEY_REGION.value)
        helpful_functions.write_routing_region(
            spec, routing_info, machine_graph, self)

        spec.end_specification()

    def _write_key_region(self, spec, routing_info, machine_graph):
        for outgoing_partition in (
                machine_graph.get_outgoing_edge_partitions_starting_at_vertex(
                    self)):
            spec.write_value(
                routing_info.get_first_key_from_partition(outgoing_partition))

    def _reverse_memory_regions(self, spec, output_data, machine_graph):
        input_n_keys = len(
            machine_graph.get_outgoing_edge_partitions_starting_at_vertex(self))

        spec.reserve_memory_region(
            self.DATA_REGIONS.SYSTEM.value,
            fec_constants.SYSTEM_BYTES_REQUIREMENT,
            label="system region")
        spec.reserve_memory_region(
            self.DATA_REGIONS.KEY_REGION.value,
            helpful_functions.sdram_size_in_bytes_for_routing_region(
                input_n_keys),
            label="routing region")
        spec.reserve_memory_region(
            self.DATA_REGIONS.OUTPUT_REGION.value, output_data.nbytes,
            label="output region")
        if self._is_recording_output:
            spec.reserve_memory_region(
                region=self.DATA_REGIONS.RECORDING.value,
                size=recording_utilities.get_recording_header_size(
                    self.N_RECORDING_REGIONS),
                label="recording")
        spec.reserve_memory_region(
            region=self.DATA_REGIONS.UPDATE_PERIOD.value,
            size=self._sdram_size_in_bytes_for_update_period_region(),
            label="update period")

    @overrides(AbstractHasAssociatedBinary.get_binary_start_type)
    def get_binary_start_type(self):
        return ExecutableType.USES_SIMULATION_INTERFACE

    @property
    @overrides(AbstractNengoMachineVertex.resources_required)
    def resources_required(self):
        return self.generate_static_resources(
            self._outgoing_partition_slice, self._n_machine_time_steps,
            self._is_recording_output)

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
            self, outgoing_partition_slice, n_machine_time_steps,
            is_recording_output):

        recording_regions = 0
        if is_recording_output:
            recording_regions += 1

        sdram = (
            # system region
            fec_constants.SYSTEM_BYTES_REQUIREMENT +
            # key region
            self._sdram_size_in_bytes_for_key_region(
                outgoing_partition_slice.n_atoms) +
            # output region
            self._sdram_size_in_bytes_for_output_region(
                outgoing_partition_slice.n_atoms, n_machine_time_steps) +
            # recordings
            recording_utilities.get_recording_header_size(recording_regions) +
            # update period
            self._sdram_size_in_bytes_for_update_period_region())

        # the basic sdram
        basic_res = ResourceContainer(sdram=SDRAMResource(sdram))

        # handle buffered recording res
        recording_sizes = recording_utilities.get_recording_region_sizes(
            [self._get_buffered_sdram(n_machine_time_steps)],
            self._minimum_buffer_sdram, self._maximum_sdram_for_buffering,
            self._using_auto_pause_and_resume)
        basic_res.extend(recording_utilities.get_recording_resources(
            recording_sizes, self._receive_buffer_host,
            self._receive_buffer_port))

        return basic_res

    def _get_buffered_sdram(self, n_machine_time_steps):
        if self._is_recording_output:
            return [
                (self.SDRAM_RECORDING_SDRAM_PER_ATOM *
                 self._outgoing_partition_slice.n_atoms * n_machine_time_steps)]
        else:
            return[0]

    @overrides(AbstractHasAssociatedBinary.get_binary_file_name)
    def get_binary_file_name(self):
        return "value_source.aplx"

    @overrides(AbstractTransmitsMulticastSignals.transmits_multicast_signals)
    def transmits_multicast_signals(self, transmission_params):
        return True
