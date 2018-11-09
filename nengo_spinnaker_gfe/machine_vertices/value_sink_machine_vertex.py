from enum import Enum
import numpy

from data_specification.enums import DataType
from nengo_spinnaker_gfe import helpful_functions
from nengo_spinnaker_gfe.abstracts.abstract_accepts_multicast_signals import \
    AbstractAcceptsMulticastSignals
from nengo_spinnaker_gfe.abstracts.abstract_nengo_machine_vertex import \
    AbstractNengoMachineVertex
from nengo_spinnaker_gfe.nengo_filters import filter_region_writer
from pacman.executor.injection_decorator import inject_items
from pacman.model.resources import ResourceContainer, SDRAMResource, \
    CPUCyclesPerTickResource, DTCMResource
from spinn_front_end_common.abstract_models import AbstractHasAssociatedBinary
from spinn_front_end_common.abstract_models.impl import \
    MachineDataSpecableVertex
from spinn_front_end_common.interface.buffer_management import \
    recording_utilities
from spinn_front_end_common.interface.buffer_management.buffer_models import \
    AbstractReceiveBuffersToHost
from spinn_front_end_common.interface.provenance import \
    ProvidesProvenanceDataFromMachineImpl
from spinn_front_end_common.interface.simulation import simulation_utilities
from spinn_front_end_common.utilities import constants
from spinn_front_end_common.utilities import helpful_functions as \
    fec_helpful_functions
from spinn_front_end_common.utilities.utility_objs import ExecutableType, \
    ProvenanceDataItem
from spinn_utilities.overrides import overrides


class ValueSinkMachineVertex(
        AbstractNengoMachineVertex, MachineDataSpecableVertex,
        AbstractHasAssociatedBinary, AbstractAcceptsMulticastSignals,
        AbstractReceiveBuffersToHost, ProvidesProvenanceDataFromMachineImpl):

    __slots__ = [
        #
        '_input_slice',
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
        "_input_filters",
        #
        "_input_n_keys",
        #
        "_time_between_requests",
        #
        "_buffer_size_before_receive"

    ]

    DATA_REGIONS = Enum(
        value="DATA_REGIONS",
        names=[('SYSTEM', 0),
               ('SLICE_DATA', 1),
               ('FILTERS', 2),
               ('FILTER_ROUTING', 3),
               ('RECORDING', 4),
               ('PROVENANCE_DATA', 5)])

    # provenance region elements
    EXTRA_PROVENANCE_DATA_ENTRIES = Enum(
        value="EXTRA_PROVENANCE_DATA_ENTRIES",
        names=[('QUEUE_OVERFLOWS', 0)])

    # n provenance items
    N_LOCAL_PROVENANCE_ITEMS = 1

    SLICE_DATA_SDRAM_REQUIREMENT = 8
    SDRAM_RECORDING_SDRAM_PER_ATOM = 4
    N_RECORDING_REGIONS = 1
    RECORDING_REGION_ID = 0


    def __init__(
            self, input_slice, minimum_buffer_sdram, receive_buffer_host,
            maximum_sdram_for_buffering, using_auto_pause_and_resume,
            receive_buffer_port, input_filters, input_n_keys,
            time_between_requests, buffer_size_before_receive,
            label):
        AbstractNengoMachineVertex.__init__(self, label=label)
        MachineDataSpecableVertex.__init__(self)
        AbstractHasAssociatedBinary.__init__(self)
        AbstractAcceptsMulticastSignals.__init__(self)
        AbstractReceiveBuffersToHost.__init__(self)
        ProvidesProvenanceDataFromMachineImpl.__init__(self)

        self._input_slice = input_slice
        self._minimum_buffer_sdram = minimum_buffer_sdram
        self._maximum_sdram_for_buffering = maximum_sdram_for_buffering
        self._using_auto_pause_and_resume = using_auto_pause_and_resume
        self._receive_buffer_host = receive_buffer_host
        self._receive_buffer_port = receive_buffer_port
        self._input_filters = input_filters
        self._input_n_keys = input_n_keys
        self._time_between_requests = time_between_requests
        self._buffer_size_before_receive = buffer_size_before_receive
        self._buffered_sdram_per_timestep = [
            self.SDRAM_RECORDING_SDRAM_PER_ATOM * input_slice.n_atoms]

    @overrides(AbstractAcceptsMulticastSignals.accepts_multicast_signals)
    def accepts_multicast_signals(self, transmission_params):
        return transmission_params.projects_to(self._input_slice.as_slice)

    @inject_items(
        {"n_machine_time_steps": "TotalMachineTimeSteps",
         "machine_time_step_in_seconds": "MachineTimeStepInSeconds",
         "graph_mapper": "NengoGraphMapper",
         "nengo_graph": "NengoOperatorGraph"})
    @overrides(
        MachineDataSpecableVertex.generate_machine_data_specification,
        additional_arguments=[
            "n_machine_time_steps", "machine_time_step_in_seconds",
            "graph_mapper", "nengo_graph"])
    def generate_machine_data_specification(
            self, spec, placement, machine_graph, routing_info, iptags,
            reverse_iptags, machine_time_step, time_scale_factor,
            n_machine_time_steps, machine_time_step_in_seconds, graph_mapper,
            nengo_graph):

        print "sink at {}".format(placement)

        # reserve the memory region blocks
        self._reserve_memory_regions(spec)

        # fill in system region
        spec.switch_write_focus(self.DATA_REGIONS.SYSTEM.value)
        spec.write_array(simulation_utilities.get_simulation_header_array(
            self.get_binary_file_name(), machine_time_step,
            time_scale_factor))

        # fill in recording region
        spec.switch_write_focus(self.DATA_REGIONS.RECORDING.value)
        recorded_region_sizes = recording_utilities.get_recorded_region_sizes(
            self._get_buffered_sdram(self._input_slice, n_machine_time_steps),
            [self._maximum_sdram_for_buffering])
        spec.write_array(recording_utilities.get_recording_header_array(
            recorded_region_sizes, self._time_between_requests,
            self._buffer_size_before_receive, iptags))

        # data on slice, aka, input slice size and start point
        spec.switch_write_focus(self.DATA_REGIONS.SLICE_DATA.value)
        spec.write_value(self._input_slice.n_atoms)
        spec.write_array(self._input_slice.lo_atom)

        # add filer region
        spec.switch_write_focus(self.DATA_REGIONS.FILTERS.value)
        filter_to_index_map = filter_region_writer.write_filter_region(
            spec, machine_time_step_in_seconds, self._input_slice,
            self._input_filters)

        # add routing region
        spec.switch_write_focus(self.DATA_REGIONS.FILTER_ROUTING.value)
        helpful_functions.write_routing_region(
            spec, routing_info, machine_graph.get_edges_ending_at_vertex(self),
            filter_to_index_map, self._input_filters, graph_mapper, nengo_graph)

        spec.end_specification()

    def _reserve_memory_regions(self, spec):
        spec.reserve_memory_region(
            self.DATA_REGIONS.SYSTEM.value,
            constants.SYSTEM_BYTES_REQUIREMENT, label="system region")
        spec.reserve_memory_region(
            self.DATA_REGIONS.SLICE_DATA.value,
            self.SLICE_DATA_SDRAM_REQUIREMENT, label="slice region")
        spec.reserve_memory_region(
            self.DATA_REGIONS.FILTER_ROUTING.value,
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
        self.reserve_provenance_data_region(spec)

    @overrides(AbstractHasAssociatedBinary.get_binary_start_type)
    def get_binary_start_type(self):
        return ExecutableType.USES_SIMULATION_INTERFACE

    @property
    @inject_items({"n_machine_time_steps": "TotalMachineTimeSteps"})
    @overrides(
        AbstractNengoMachineVertex.resources_required,
        additional_arguments=["n_machine_time_steps"])
    def resources_required(self, n_machine_time_steps):
        container = ResourceContainer(
            sdram=SDRAMResource(
                constants.SYSTEM_BYTES_REQUIREMENT +
                ProvidesProvenanceDataFromMachineImpl.get_provenance_data_size(
                    self.N_LOCAL_PROVENANCE_ITEMS) +
                ValueSinkMachineVertex.SLICE_DATA_SDRAM_REQUIREMENT +
                helpful_functions.sdram_size_in_bytes_for_filter_region(
                    self._input_filters) +
                helpful_functions.sdram_size_in_bytes_for_routing_region(
                    self._input_n_keys) +
                recording_utilities.get_recording_header_size(
                    self.N_RECORDING_REGIONS)),
            dtcm=DTCMResource(0),
            cpu_cycles=CPUCyclesPerTickResource(0))

        recording_sizes = recording_utilities.get_recording_region_sizes(
            [self._get_buffered_sdram(self._input_slice, n_machine_time_steps)],
            self._minimum_buffer_sdram, self._maximum_sdram_for_buffering,
            self._using_auto_pause_and_resume)
        container.extend(recording_utilities.get_recording_resources(
            recording_sizes, self._receive_buffer_host,
            self._receive_buffer_port))
        return container

    def _get_buffered_sdram(self, input_slice, n_machine_time_steps):
        return [(self.SDRAM_RECORDING_SDRAM_PER_ATOM * input_slice.n_atoms *
                n_machine_time_steps)]

    @overrides(AbstractReceiveBuffersToHost.get_minimum_buffer_sdram_usage)
    def get_minimum_buffer_sdram_usage(self):
        return self._minimum_buffer_sdram

    @overrides(AbstractReceiveBuffersToHost.get_recording_region_base_address)
    def get_recording_region_base_address(self, txrx, placement):
        return fec_helpful_functions.locate_memory_region_for_placement(
            placement=placement, transceiver=txrx,
            region=self.DATA_REGIONS.RECORDING.value)

    @overrides(AbstractReceiveBuffersToHost.get_recorded_region_ids)
    def get_recorded_region_ids(self):
        return recording_utilities.get_recorded_region_ids(
            self._buffered_sdram_per_timestep)

    @overrides(AbstractReceiveBuffersToHost.get_n_timesteps_in_buffer_space)
    def get_n_timesteps_in_buffer_space(self, buffer_space, machine_time_step):
        return recording_utilities.get_n_timesteps_in_buffer_space(
            buffer_space, [self._buffered_sdram_per_timestep])

    @overrides(AbstractHasAssociatedBinary.get_binary_file_name)
    def get_binary_file_name(self):
        return "value_sink.aplx"

    @property
    @overrides(ProvidesProvenanceDataFromMachineImpl._n_additional_data_items)
    def _n_additional_data_items(self):
        return self.N_LOCAL_PROVENANCE_ITEMS

    @property
    @overrides(ProvidesProvenanceDataFromMachineImpl._provenance_region_id)
    def _provenance_region_id(self):
        return self.DATA_REGIONS.PROVENANCE_DATA.value

    @overrides(ProvidesProvenanceDataFromMachineImpl.
               get_provenance_data_from_machine)
    def get_provenance_data_from_machine(self, transceiver, placement):
        # get data from basic prov
        provenance_data = self._read_provenance_data(transceiver, placement)
        provenance_items = self._read_basic_provenance_items(
            provenance_data, placement)
        provenance_data = self._get_remaining_provenance_data_items(
            provenance_data)

        # get item in data
        queue_overflows = provenance_data[
            self.EXTRA_PROVENANCE_DATA_ENTRIES.QUEUE_OVERFLOWS.value]
        label, x, y, p, names = self._get_placement_details(placement)

        # translate into provenance data items
        provenance_items.append(ProvenanceDataItem(
            self._add_name(names, "Time_queue_overflows"),
            queue_overflows,
            report=queue_overflows > 0,
            message=(
                "The packets acquired by core {}:{}:{} running model {} "
                "failed to keep {} items in its buffer. Unknown how to "
                "rectify".format(x, y, p, self.get_binary_file_name(),
                                 queue_overflows))))
        return provenance_items

    def get_data_for_recording_region(
            self, run_time, placement, buffer_manager, sampling_interval):
        neuron_param_region, data_missing = buffer_manager.get_data_for_vertex(
            placement, self.RECORDING_REGION_ID)

        # convert to byte buffer and read into numpy array of uint32's
        data = numpy.frombuffer(
            neuron_param_region.read_all(), dtype=numpy.int32)

        # some shaping stuff
        data.shape = (run_time, -1)

        # TODO push this down into the c code for better efficency
        # apply sampling interval
        data = data[::sampling_interval]

        # Recast back to float and return
        return data / DataType.S1615.scale




