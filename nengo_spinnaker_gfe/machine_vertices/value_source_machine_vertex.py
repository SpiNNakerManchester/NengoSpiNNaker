from enum import Enum
import random

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
               ('OUTPUT_REGION', 1),
               ('KEY_REGION', 2),
               ('NEURON_REGION', 3),
               ('RECORDING', 4)])

    SDRAM_RECORDING_SDRAM_PER_ATOM = 4
    N_RECORDING_REGIONS = 1
    NEURON_REGION_ITEMS = 4

    def __init__(
            self, outgoing_partition_slice,
            update_period, minimum_buffer_sdram, receive_buffer_host,
            maximum_sdram_for_buffering, using_auto_pause_and_resume,
            receive_buffer_port, is_recording_output,
            this_cores_matrix, label):

        AbstractNengoMachineVertex.__init__(self, label=label)
        MachineDataSpecableVertex.__init__(self)
        AbstractHasAssociatedBinary.__init__(self)
        AbstractTransmitsMulticastSignals.__init__(self)
        self._outgoing_partition_slice = outgoing_partition_slice
        self._minimum_buffer_sdram = minimum_buffer_sdram
        self._maximum_sdram_for_buffering = maximum_sdram_for_buffering
        self._using_auto_pause_and_resume = using_auto_pause_and_resume
        self._receive_buffer_host = receive_buffer_host
        self._receive_buffer_port = receive_buffer_port
        self._update_period = update_period
        self._is_recording_output = is_recording_output
        self._output_data = this_cores_matrix

    @inject_items({"n_machine_time_steps": "TotalMachineTimeSteps",
                   "current_time_step": "FirstMachineTimeStep",
                   "graph_mapper": "NengoGraphMapper"})
    @overrides(
        MachineDataSpecableVertex.generate_machine_data_specification,
        additional_arguments=[
            "n_machine_time_steps", "current_time_step", "graph_mapper"])
    def generate_machine_data_specification(
            self, spec, placement, machine_graph, routing_info, iptags,
            reverse_iptags, machine_time_step, time_scale_factor,
            n_machine_time_steps, current_time_step, graph_mapper):

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

        # add output region
        spec.switch_write_focus(self.DATA_REGIONS.OUTPUT_REGION.value)
        spec.write_array(helpful_functions.convert_numpy_array_to_s16_15(
            self._output_data))

        # add routing region
        spec.switch_write_focus(self.DATA_REGIONS.KEY_REGION.value)
        self._write_key_region(machine_graph, routing_info, spec)

        # add params region
        spec.switch_write_focus(self.DATA_REGIONS.PARAMS_REGION.value)
        spec.write_value(self._update_period is not None)
        spec.write_value(self._outgoing_partition_slice.n_atoms)

        # Write the random back off value
        app_vertex = graph_mapper.get_application_vertex(self)
        spec.write_value(random.randint(0, min(
            app_vertex.n_value_source_machine_vertices,
            constants.MICROSECONDS_PER_SECOND // machine_time_step)))

        # write time between spikes
        spikes_per_time_step = (
            self._outgoing_partition_slice.n_atoms / (
                constants.MICROSECONDS_PER_SECOND // machine_time_step))
        # avoid a possible division by zero / small number (which may
        # result in a value that doesn't fit in a uint32) by only
        # setting time_between_spikes if spikes_per_timestep is > 1
        time_between_spikes = 0.0
        if spikes_per_time_step > 1:
            time_between_spikes = (
                (machine_time_step * time_scale_factor) /
                (spikes_per_time_step * 2.0))
        spec.write_value(data=int(time_between_spikes))
        spec.end_specification()

    def _write_key_region(self, spec, routing_info, machine_graph):
        outgoing_partition = machine_graph. \
            get_outgoing_edge_partitions_starting_at_vertex(self)[0]
        partition_routing_info = \
            routing_info.get_routing_info_from_partition(outgoing_partition)
        keys = partition_routing_info.get_keys(
            self._outgoing_partition_slice.n_atoms)

        spec.write_value(self._outgoing_partition_slice.n_atoms)
        for key in keys:
            spec.write_value(key)

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
            region=self.DATA_REGIONS.NEURON_REGION.value,
            size=self.NEURON_REGION_ITEMS * constants.BYTE_TO_WORD_MULTIPLIER,
            label="n neurons")

    @overrides(AbstractHasAssociatedBinary.get_binary_start_type)
    def get_binary_start_type(self):
        return ExecutableType.USES_SIMULATION_INTERFACE

    @property
    @inject_items({"n_machine_time_steps": "TotalMachineTimeSteps"})
    @overrides(
        AbstractNengoMachineVertex.resources_required,
        additional_arguments=["n_machine_time_steps"])
    def resources_required(self, n_machine_time_steps):
        return self.generate_static_resources(
            self._outgoing_partition_slice, n_machine_time_steps,
            self._is_recording_output)

    @staticmethod
    def _sdram_size_in_bytes_for_key_region(n_atoms):
        return n_atoms * constants.BYTES_PER_KEY

    @staticmethod
    def _sdram_size_in_bytes_for_output_region(n_atoms, n_machine_time_steps):
        return ((n_atoms * n_machine_time_steps) *
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
            # params region
            (self.NEURON_REGION_ITEMS * constants.BYTE_TO_WORD_MULTIPLIER))

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
