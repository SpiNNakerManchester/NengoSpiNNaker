from enum import Enum
import random

from nengo_spinnaker_gfe import constants, helpful_functions
from nengo_spinnaker_gfe.abstracts.abstract_accepts_multicast_signals import \
    AbstractAcceptsMulticastSignals
from nengo_spinnaker_gfe.abstracts.abstract_nengo_machine_vertex import \
    AbstractNengoMachineVertex
from nengo_spinnaker_gfe.abstracts.abstract_transmits_multicast_signals import \
    AbstractTransmitsMulticastSignals
from nengo_spinnaker_gfe.nengo_filters import filter_region_writer
from pacman.executor.injection_decorator import inject_items
from pacman.model.resources import ResourceContainer, SDRAMResource
from spinn_front_end_common.abstract_models import AbstractHasAssociatedBinary
from spinn_front_end_common.abstract_models.impl import \
    MachineDataSpecableVertex
from spinn_front_end_common.interface.simulation import simulation_utilities
from spinn_front_end_common.utilities.utility_objs import ExecutableType
from spinn_front_end_common.utilities import constants as fec_constants
from spinn_utilities.overrides import overrides


class InterposerMachineVertex(
        AbstractNengoMachineVertex, MachineDataSpecableVertex,
        AbstractHasAssociatedBinary, AbstractAcceptsMulticastSignals,
        AbstractTransmitsMulticastSignals):

    __slots__ = [
        "_size_in",
        "_output_slice",
        "_transform_data",
        "_n_keys",
        "_filter_keys",
        "_output_slices",
        "_machine_time_step",
        "_filters",
        "_transmission_params"
    ]

    """Portion of the rows of the transform assigned to a parallel filter
    group, represents the load assigned to a single processing core.
    """

    DATA_REGIONS = Enum(
        value="DATA_REGIONS",
        names=[('SYSTEM', 0),
               ('SLICE_DATA', 1),
               ('KEYS', 2),
               ('INPUT_FILTERS', 3),
               ('INPUT_ROUTING', 4),
               ('TRANSFORM', 5),
               ('MC_TRANSMISSION', 6)])

    SLICE_DATA_ITEMS = 3
    MC_TRANSMISSION_REGION_ITEMS = 2

    def __init__(
            self, size_in, output_slice, transform_data, n_keys, filter_keys,
            output_slices, machine_time_step, filters, label, constraints):
        AbstractNengoMachineVertex.__init__(
            self, label=label, constraints=constraints)
        AbstractHasAssociatedBinary.__init__(self)
        AbstractAcceptsMulticastSignals.__init__(self)
        MachineDataSpecableVertex.__init__(self)
        AbstractTransmitsMulticastSignals.__init__(self)

        self._size_in = size_in
        self._output_slice = output_slice
        self._transform_data = transform_data
        self._n_keys = n_keys
        self._filter_keys = filter_keys
        self._output_slices = output_slices
        self._machine_time_step = machine_time_step
        self._filters = filters

        # Store which signal parameter slices we contain
        self._transmission_params = self._filter_transmission_params()

    def _filter_transmission_params(self):
        transmission_params = set()
        out_set = set(range(self._output_slice.start, self._output_slice.stop))
        for transmission_params, outs in self._output_slices:
            # If there is an intersection between the outs and the set of outs
            # we're responsible for then store transmission parameters.
            if out_set & outs:
                transmission_params.add(transmission_params)
        return out_set

    @inject_items(
        {"machine_time_step_in_seconds": "MachineTimeStepInSeconds",
         "graph_mapper": "NengoGraphMapper"})
    @overrides(MachineDataSpecableVertex.generate_machine_data_specification,
               additional_arguments=["machine_time_step_in_seconds"])
    def generate_machine_data_specification(
            self, spec, placement, machine_graph, routing_info, iptags,
            reverse_iptags, machine_time_step, time_scale_factor,
            machine_time_step_in_seconds, graph_mapper):

        self._allocate_memory_regions(spec)
        spec.switch_write_focus(self.DATA_REGIONS.SYSTEM.value)
        spec.write_array(simulation_utilities.get_simulation_header_array(
            self.get_binary_file_name(), machine_time_step,
            time_scale_factor))
        spec.switch_write_focus(self.DATA_REGIONS.SLICE_DATA.value)
        self._write_slice_data_to_region(spec)
        spec.switch_write_focus(self.DATA_REGIONS.KEYS.value)
        self._write_key_data(spec, routing_info)
        spec.switch_write_focus(self.DATA_REGIONS.INPUT_FILTERS.value)
        filter_to_index_map = filter_region_writer.write_filter_region(
            spec, machine_time_step_in_seconds, self._input_slice,
            self._input_filters)
        spec.switch_write_focus(self.DATA_REGIONS.INPUT_ROUTING.value)
        helpful_functions.write_routing_region(
            spec, routing_info, machine_graph, self, filter_to_index_map,
            self._input_filters)
        spec.switch_write_focus(self.DATA_REGIONS.TRANSFORM.value)
        spec.write_array(helpful_functions.convert_numpy_array_to_s16_15(
            self._transform_data))
        spec.switch_write_focus(self.DATA_REGIONS.MC_TRANSMISSION.value)
        self._write_mc_transmission_params(
            spec, graph_mapper, machine_time_step, time_scale_factor)
        spec.end_specification()

    def _write_mc_transmission_params(
            self, spec, graph_mapper, machine_time_step, time_scale_factor):
        # Write the random back off value
        app_vertex = graph_mapper.get_application_vertex(self)
        spec.write_value(random.randint(0, min(
            app_vertex.n_sdp_receiver_machine_vertices,
            constants.MICROSECONDS_PER_SECOND // machine_time_step)))

        # avoid a possible division by zero / small number (which may
        # result in a value that doesn't fit in a uint32) by only
        # setting time_between_spikes if spikes_per_timestep is > 1
        time_between_spikes = 0.0
        if self._output_slice.n_atoms > 1:
            time_between_spikes = (
                (machine_time_step * time_scale_factor) /
                (self._output_slice.n_atoms * 2.0)) / 2
        spec.write_value(data=int(time_between_spikes))

    def _write_key_data(self, spec, routing_info):
        partition_routing_info = routing_info.get_routing_info_from_partition(
            self._managing_outgoing_partition)
        if partition_routing_info is None:
            spec.write_value(0)
        else:
            spec.write_value(len(partition_routing_info.get_keys()))
            for key in partition_routing_info.get_keys():
                spec.write_value(key)

    def _write_slice_data_to_region(self, spec):
        spec.write_value(self._size_in.n_atoms)
        spec.write_value(self._size_in.lo_atom)
        spec.write_value(self._output_slice.n_atoms)

    def _allocate_memory_regions(self, spec):
        spec.reserve_memory_region(
            self.DATA_REGIONS.SYSTEM.value,
            fec_constants.SYSTEM_BYTES_REQUIREMENT, label="system region")
        self.reserve_memory_region(
            self.DATA_REGIONS.SLICE_DATA.value,
            self.SLICE_DATA_ITEMS * constants.BYTE_TO_WORD_MULTIPLIER,
            label="slice data")
        self.reserve_memory_region(
            self.DATA_REGIONS.KEYS.value,
            constants.BYTES_PER_KEY * self._n_keys,
            label="keys data")
        self.reserve_memory_region(
            self.DATA_REGIONS.TRANSFORM.value,
            self._transform_data.nbytes,
            label="transform data")
        self.reserve_memory_region(
            self.DATA_REGIONS.INPUT_FILTERS.value,
            helpful_functions.sdram_size_in_bytes_for_filter_region(
                self._filters),
            label="input filter data")
        self.reserve_memory_region(
            self.DATA_REGIONS.INPUT_ROUTING.value,
            helpful_functions.sdram_size_in_bytes_for_routing_region(
                self._n_keys),
            label="routing data")
        self.reserve_memory_region(
            self.DATA_REGIONS.MC_TRANSMISSION.value,
            (self.MC_TRANSMISSION_REGION_ITEMS *
             constants.BYTE_TO_WORD_MULTIPLIER),
            label="mc_transmission data")


    @overrides(AbstractAcceptsMulticastSignals.accepts_multicast_signals)
    def accepts_multicast_signals(self, transmission_params):
        return transmission_params.projects_to(self._column_slice)

    @overrides(AbstractHasAssociatedBinary.get_binary_file_name)
    def get_binary_file_name(self):
        return "interposer.aplx"  # this was filter in mundy code

    @property
    @overrides(AbstractNengoMachineVertex.resources_required)
    def resources_required(self):
        return self.generate_static_resources(
            self._transform_data, self._n_keys, self._filters)

    @staticmethod
    def generate_static_resources(transform_data, n_keys, filters):
        sdram = (
            fec_constants.SYSTEM_BYTES_REQUIREMENT +
            (InterposerMachineVertex.SLICE_DATA_ITEMS *
             constants.BYTE_TO_WORD_MULTIPLIER) +
            (InterposerMachineVertex.MC_TRANSMISSION_REGION_ITEMS *
             constants.BYTE_TO_WORD_MULTIPLIER) +
            transform_data.nbytes + (constants.BYTES_PER_KEY * n_keys) +
            helpful_functions.sdram_size_in_bytes_for_filter_region(filters) +
            helpful_functions.sdram_size_in_bytes_for_routing_region(n_keys))
        return ResourceContainer(sdram=SDRAMResource(sdram))

    @overrides(AbstractHasAssociatedBinary.get_binary_start_type)
    def get_binary_start_type(self):
        return ExecutableType.USES_SIMULATION_INTERFACE

    @overrides(AbstractAcceptsMulticastSignals.accepts_multicast_signals)
    def accepts_multicast_signals(self, transmission_params):
        return transmission_params.projects_to(self._size_in)

    @overrides(AbstractTransmitsMulticastSignals.transmits_multicast_signals)
    def transmits_multicast_signals(self, transmission_params):
        return transmission_params in self._transmission_params
