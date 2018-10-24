from enum import Enum
import numpy
import logging

from data_specification.enums import DataType
from nengo_spinnaker_gfe import constants, helpful_functions
from nengo_spinnaker_gfe.abstracts.abstract_accepts_multicast_signals import \
    AbstractAcceptsMulticastSignals
from nengo_spinnaker_gfe.abstracts.abstract_nengo_machine_vertex import \
    AbstractNengoMachineVertex
from nengo_spinnaker_gfe.abstracts.abstract_transmits_multicast_signals import \
    AbstractTransmitsMulticastSignals
from nengo_spinnaker_gfe.nengo_filters import filter_region_writer
from pacman.executor.injection_decorator import inject_items

from spinn_front_end_common.abstract_models import AbstractHasAssociatedBinary
from spinn_front_end_common.abstract_models.impl import \
    MachineDataSpecableVertex
from spinn_front_end_common.interface.buffer_management import \
    recording_utilities
from spinn_front_end_common.interface.buffer_management.buffer_models import \
    AbstractReceiveBuffersToHost
from spinn_front_end_common.interface.simulation import simulation_utilities
from spinn_front_end_common.utilities.utility_objs import ExecutableType
from spinn_front_end_common.utilities import constants as fec_constants
from spinn_front_end_common.utilities import helpful_functions as \
    fec_helpful_functions

from spinn_utilities.overrides import overrides

logger = logging.getLogger(__name__)


class LIFMachineVertex(
        AbstractNengoMachineVertex, MachineDataSpecableVertex,
        AbstractHasAssociatedBinary, AbstractAcceptsMulticastSignals,
        AbstractTransmitsMulticastSignals, AbstractReceiveBuffersToHost):

    __slots__ = [
        "_resources",
        "_neuron_slice",
        "_input_slice",
        "_output_slice",
        "_learnt_slice",
        "_n_profiler_samples",
        "_ensemble_size_in",
        "_encoders_with_gain",
        "_learnt_encoder_filters",
        "_sub_population_id",
        "_tau_refactory",
        "_tau_rc",
        "_input_filters",
        "_inhibitory_filters",
        "_modulatory_filters",
        "_local_pes_learning_rules",
        "_ensemble_radius"
    ]

    DATA_REGIONS = Enum(
        value="DATA_REGIONS",
        names=[
            ('SYSTEM', 0),
            # Ensemble, NEURON, POP length (from mundy)
            ('ENSEMBLE_PARAMS', 1),
            ('ENCODER', 2),
            ('BIAS', 3),
            ('GAIN', 4),
            ('DECODER', 5),
            ('LEARNT_DECODER', 6),
            # KEYS and LEARNT KEYS (from mundy)
            ('KEYS', 7),
            # INPUT FILTERS, INHIB FILTERS, MODULATORY  FILTERS,
            # LEARNT ENDCODER FILTERS (from mundy)
            ('FILTERS', 8),
            # INPUT ROUTING, INHIB ROUTING, MOD ROUTING, LEANT ENCODER ROUTING
            # (from mundy)
            ('ROUTING', 9),
            ('PES', 10),
            ('VOJA', 11),
            # only one for SPike Voltage Encoder (from mundy)
            ('RECORDING', 12)
           ])  # 13 from 26

    # SDRAM calculation
    ENSEMBLE_PARAMS_ITEMS = 17
    SDRAM_ITEMS_PER_LEARNT_INPUT_VECTOR = 2
    NEURON_PARAMS_ITEMS = 2
    POP_LENGTH_CONSTANT_ITEMS = 1
    PES_REGION_N_ELEMENTS = 1
    VOJA_REGION_N_ELEMENTS = 2
    VOJA_REGION_RULE_N_ELEMENT = 4
    PES_REGION_SLICED_RULE_N_ELEMENTS = 5
    SHARED_SDRAM_FOR_SEMAPHORES_IN_BYTES = 4

    FUNCTION_OF_NEURON_TIME_CONSTANT = (1.0 - 2**-11)
    ONE = 1.0

    def __init__(
            self, sub_population_id, neuron_slice, input_slice, output_slice,
            learnt_slice, resources, encoders_with_gain, tau_rc, tau_refactory,
            ensemble_size_in, label, learnt_encoder_filters, input_filters,
            inhibitory_filters, modulatory_filters, pes_learning_rules,
            ensemble_radius, minimum_buffer_sdram_usage,
            buffered_sdram_per_timestep, overflow_sdram):
        AbstractNengoMachineVertex.__init__(self, label=label)
        MachineDataSpecableVertex.__init__(self)
        AbstractHasAssociatedBinary.__init__(self)
        AbstractAcceptsMulticastSignals.__init__(self)
        AbstractTransmitsMulticastSignals.__init__(self)
        AbstractReceiveBuffersToHost.__init__(self)

        self._resources = resources
        self._neuron_slice = neuron_slice
        self._input_slice = input_slice
        self._output_slice = output_slice
        self._learnt_slice = learnt_slice
        self._sub_population_id = sub_population_id
        self._ensemble_size_in = ensemble_size_in
        self._encoders_with_gain = encoders_with_gain
        self._tau_rc = tau_rc
        self._tau_refactory = tau_refactory
        self._learnt_encoder_filters = learnt_encoder_filters
        self._input_filters = input_filters
        self._inhibitory_filters = inhibitory_filters
        self._modulatory_filters = modulatory_filters
        self._local_pes_learning_rules = pes_learning_rules
        self._ensemble_radius = ensemble_radius

        # recording params
        self._minimum_buffer_sdram_usage = minimum_buffer_sdram_usage
        self._buffered_sdram_per_timestep = buffered_sdram_per_timestep
        self._overflow_sdram = overflow_sdram

    @property
    def neuron_slice(self):
        return self._neuron_slice

    @property
    def input_slice(self):
        return self._input_slice

    @property
    def output_slice(self):
        return self._output_slice

    @property
    def learnt_slice(self):
        return self._learnt_slice

    @overrides(AbstractAcceptsMulticastSignals.accepts_multicast_signals)
    def accepts_multicast_signals(self, transmission_params):
        return True

    @overrides(AbstractTransmitsMulticastSignals.transmits_multicast_signals)
    def transmits_multicast_signals(self, transmission_params):
        return True

    @inject_items({"graph_mapper": "NengoGraphMapper",
                   "machine_time_step_in_seconds": "MachineTimeStepInSeconds",
                   "n_machine_time_steps": "TotalMachineTimeSteps"})
    @overrides(
        MachineDataSpecableVertex.generate_machine_data_specification,
        additional_arguments=[
            "graph_mapper", "machine_time_step_in_seconds",
            "n_machine_time_steps"])
    def generate_machine_data_specification(
            self, spec, placement, machine_graph, routing_info, iptags,
            reverse_iptags, machine_time_step, time_scale_factor,
            graph_mapper, machine_time_step_in_seconds, n_machine_time_steps):

        # get the associated app vertex.
        app_vertex = graph_mapper.get_application_vertex(self)

        # allocate the memory regions
        self._allocate_memory_regions(spec, app_vertex)

        # process the system region
        spec.switch_write_focus(self.DATA_REGIONS.SYSTEM.value)
        spec.write_array(simulation_utilities.get_simulation_header_array(
            self.get_binary_file_name(), machine_time_step, time_scale_factor))

        # process the ensemble params region
        spec.switch_write_focus(self.DATA_REGIONS.ENSEMBLE_PARAMS.value)
        self._write_ensemble_neuron_pop_length_params(
            spec, graph_mapper, machine_graph, machine_time_step_in_seconds,
            app_vertex)

        # process the filters region
        spec.switch_write_focus(self.DATA_REGIONS.FILTERS.value)
        self._write_filters_region(spec, machine_time_step_in_seconds)

        # process the routes region
        spec.switch_write_focus(self.DATA_REGIONS.ROUTING.value)
        self._write_routes_region(spec)

        # process the keys region
        spec.switch_write_focus(self.DATA_REGIONS.KEYS.value)
        self._write_keys_region(spec)

        # process the pes region
        spec.switch_write_focus(self.DATA_REGIONS.PES.value)
        self._write_pes_region(spec, app_vertex)

        # process the voja region
        spec.switch_write_focus(self.DATA_REGIONS.VOJA.value)
        self._write_voja_region(spec, app_vertex)

        # process recording region
        spec.switch_write_focus(self.DATA_REGIONS.RECORDING.value)
        self._write_recording_region(
            spec, iptags, n_machine_time_steps, app_vertex)

        spec.end_specification()

    def _write_recording_region(
            self, spec, ip_tags, n_machine_time_steps, app_vertex):
        """
        
        :param spec: 
        :param ip_tags: 
        :param n_machine_time_steps: 
        :return: 
        """
        recorded_region_sizes = recording_utilities.get_recorded_region_sizes(
            app_vertex.get_buffered_sdram(
                self._neuron_slice, n_machine_time_steps),
            app_vertex.maximum_sdram_for_buffering)
        spec.write_array(recording_utilities.get_recording_header_array(
            recorded_region_sizes, app_vertex.time_between_requests,
            app_vertex.buffer_size_before_receive, ip_tags))

    def _write_pes_region(self, spec, app_vertex):
        """
        
        :param spec: 
        :param app_vertex:
        :return: 
        """
        spec.write_value(len(self._local_pes_learning_rules))
        for pes_learning_rule in self._local_pes_learning_rules:
            # Error signal starts either at 1st dimension or the first
            # dimension of decoder that occurs within learnt output slice
            error_start_dim = max(
                0, self._learnt_slice.lo_atom - pes_learning_rule.decoder_start)

            # Error signal end either at last dimension or
            # the last dimension of learnt output slice
            error_end_dim = min(
                pes_learning_rule.decoder_stop -
                pes_learning_rule.decoder_start,
                self._learnt_slice.hi_atom - pes_learning_rule.decoder_start)

            # The row of the global decoder is the learnt row relative to the
            # start of the learnt output slice with the number of static
            # decoder rows added to put it into combined decoder space
            decoder_row = self._output_slice.n_atoms + max(
                0, pes_learning_rule.decoder_start - self._learnt_slice.lo_atom)

            spec.write_value(
                pes_learning_rule.learning_rate / app_vertex.n_neurons,
                data_type=DataType.S1615)
            spec.write_value(pes_learning_rule.error_filter_index)
            spec.write_value(error_start_dim)
            spec.write_value(error_end_dim)
            spec.write_value(decoder_row, data_type=DataType.INT32)

    def _write_voja_region(self, spec, app_vertex):
        """
        
        :param spec: 
        :param app_vertex: 
        :return: 
        """
        spec.write_value(len(app_vertex.voja_learning_rules))
        spec.write_value(
            self.ONE / self._ensemble_radius, data_type=DataType.S1615)

        for learning_rule in app_vertex.voja_learning_rules:
            spec.write_value(
                learning_rule.learning_rate, data_type=DataType.S1615)
            spec.write_value(
                learning_rule.learning_signal_filter_index,
                data_type=DataType.INT32)
            spec.write_value(learning_rule.encoder_offset)
            spec.write_value(learning_rule.decoded_input_filter_index)

    def _write_keys_region(self, spec):
        pass

    def _write_routes_region(self, spec):
        pass

    def _write_filters_region(self, spec, machine_time_step_in_seconds):
        """
        
        :param spec: 
        :param machine_time_step_in_seconds: 
        :return: 
        """
        filter_to_index_map = filter_region_writer.write_filter_region(
            spec, machine_time_step_in_seconds, self._input_slice,
            self._input_filters)
        filter_region_writer.write_filter_region(
            spec, machine_time_step_in_seconds, self._input_slice,
            self._inhibitory_filters)
        filter_region_writer.write_filter_region(
            spec, machine_time_step_in_seconds, self._input_slice,
            self._modulatory_filters)
        filter_region_writer.write_filter_region(
            spec, machine_time_step_in_seconds, self._input_slice,
            self._learnt_encoder_filters)

    def _write_ensemble_neuron_pop_length_params(
            self, spec, graph_mapper, machine_graph,
            machine_time_step_in_seconds, app_vertex):
        """
        
        :param spec: 
        :param graph_mapper: 
        :param machine_graph: 
        :param machine_time_step_in_seconds: 
        :param app_vertex: 
        :return: 
        """

        spec.write_value(self._neuron_slice.n_atoms)
        spec.write_value(self._ensemble_size_in)
        spec.write_value(self._encoders_with_gain.shape[1])
        spec.write_value(app_vertex.n_neurons)
        spec.write_value(len(graph_mapper.get_machine_vertices(app_vertex)))
        spec.write_value(self._sub_population_id)
        spec.write_value(self._input_slice.lo_atom)
        spec.write_value(self._input_slice.n_atoms)
        spec.write_value(self._output_slice.n_atoms)
        spec.write_value(self._learnt_slice.n_atoms)

        # my local input memory point
        machine_graph_edge = \
            machine_graph.get_edges_ending_at_vertex_with_partition_name(
                self, app_vertex.SDRAM_OUTGOING_INPUT)[0]
        outgoing_partition = \
            machine_graph.get_outgoing_partition_for_edge(machine_graph_edge)
        if outgoing_partition.sdram_base_address is None:
            logger.warn(
                "No sdram memory address was assigned. Therefore will assume "
                "running in virtual machine mode and carry on.")
        else:
            spec.write_value(outgoing_partition.sdram_base_address)
            spec.write_value(machine_graph_edge.sdram_base_address)

        # my local spike point
        machine_graph_edge = \
            machine_graph.get_edges_ending_at_vertex_with_partition_name(
                self,
                app_vertex.SDRAM_OUTGOING_SPIKE_VECTOR)[0]
        outgoing_partition = \
            machine_graph.get_outgoing_partition_for_edge(machine_graph_edge)
        if outgoing_partition.sdram_base_address is None:
            logger.warn(
                "No sdram memory address was assigned. Therefore will assume "
                "running in virtual machine mode and carry on.")
        else:
            spec.write_value(outgoing_partition.sdram_base_address)
            spec.write_value(machine_graph_edge.sdram_base_address)

        # the semaphore point
        semaphore_edge = \
            machine_graph.get_edges_ending_at_vertex_with_partition_name(
                self, app_vertex.SDRAM_OUTGOING_SEMAPHORE)[0]
        if semaphore_edge.sdram_base_address is None:
            logger.warn(
                "No sdram memory address was assigned. Therefore will assume "
                "running in virtual machine mode and carry on.")
        else:
            spec.write_value(semaphore_edge.sdram_base_address)

        # write each sdram address for each learnt encoder
        spec.write_value(len(self._learnt_encoder_filters))
        for learnt_encoder_filter in self._learnt_encoder_filters:
            machine_graph_edge = \
                machine_graph.get_edges_ending_at_vertex_with_partition_name(
                    self,
                    (app_vertex.SDRAM_OUTGOING_LEARNT,
                     learnt_encoder_filter))[0]
            outgoing_partition = \
                machine_graph.get_outgoing_partition_for_edge(
                    machine_graph_edge)
            if outgoing_partition.sdram_base_address is None:
                logger.warn(
                    "No sdram memory address was assigned. Therefore will "
                    "assume running in virtual machine mode and carry on.")
            else:
                spec.write_value(outgoing_partition.sdram_base_address)
                spec.write_value(machine_graph_edge.sdram_base_address)

        # add the neuron params to this region.
        spec.write_value(
            (-numpy.expm1(-machine_time_step_in_seconds / self._tau_rc) *
             self.FUNCTION_OF_NEURON_TIME_CONSTANT),
            data_type=DataType.S1615)
        spec.write_value((self._tau_refactory // machine_time_step_in_seconds),
                         data_type=DataType.S1615)

        # add pop length data
        chip_level_machine_vertex_slices = app_vertex.machine_vertex_slices[
            app_vertex.core_slice_to_chip_slice[self.neuron_slice]]
        spec.write_value(len(chip_level_machine_vertex_slices))
        for chip_level_core_slice in chip_level_machine_vertex_slices:
            spec.write_value(chip_level_core_slice.n_atoms)

    def _allocate_memory_regions(self, spec, app_vertex):
        """
        
        :param spec: 
        :param app_vertex: 
        :return: 
        """

        # standard system region
        spec.reserve_memory_region(
            self.DATA_REGIONS.SYSTEM.value,
            fec_constants.SYSTEM_BYTES_REQUIREMENT, label="system region")

        # ensemble / neuron / pop length region
        n_pop_length_sizes = len(app_vertex.machine_vertex_slices[
            app_vertex.core_slice_to_chip_slice[self.neuron_slice]])
        spec.reserve_memory_region(
            self.DATA_REGIONS.ENSEMBLE_PARAMS.value,
            (self.ENSEMBLE_PARAMS_ITEMS + self.NEURON_PARAMS_ITEMS +
             (len(self._learnt_encoder_filters) *
              self.SDRAM_ITEMS_PER_LEARNT_INPUT_VECTOR) +
             self.POP_LENGTH_CONSTANT_ITEMS + n_pop_length_sizes) *
            constants.BYTE_TO_WORD_MULTIPLIER,
            label="ensemble params")

        # reserve filter region
        spec.reserve_memory_region(
            self.DATA_REGIONS.FILTERS.value,
            (helpful_functions.sdram_size_in_bytes_for_filter_region(
                self._input_filters) +
             helpful_functions.sdram_size_in_bytes_for_filter_region(
                 self._inhibitory_filters) +
             helpful_functions.sdram_size_in_bytes_for_filter_region(
                 self._modulatory_filters) +
             helpful_functions.sdram_size_in_bytes_for_filter_region(
                 self._learnt_encoder_filters)),
            label="filters")

        # reserve routing region
        spec.reserve_memory_region(
            self.DATA_REGIONS.ROUTING.value,
            (helpful_functions.sdram_size_in_bytes_for_routing_region(
                app_vertex.input_n_keys) +
             helpful_functions.sdram_size_in_bytes_for_routing_region(
                 app_vertex.inhibition_n_keys) +
             helpful_functions.sdram_size_in_bytes_for_routing_region(
                 app_vertex.modulatory_n_keys) +
             helpful_functions.sdram_size_in_bytes_for_routing_region(
                 app_vertex.learnt_encoders_n_keys)),
            label="routing")

        # reserve the key region
        spec.reserve_memory_region(
            self.DATA_REGIONS.KEYS.value,
            ((constants.BYTES_PER_KEY * app_vertex.output_n_keys *
             self._output_slice.n_atoms) +
             (constants.BYTES_PER_KEY * app_vertex.learnt_output_n_keys *
              self._learnt_slice.n_atoms)),
            label="keys region")

        # reserve the pes region
        spec.reserve_memory_region(
            self.DATA_REGIONS.PES.value,
            # pes learning rule region
            (self.PES_REGION_N_ELEMENTS + len(self._local_pes_learning_rules) +
             self.PES_REGION_SLICED_RULE_N_ELEMENTS),
            label="pes region")

        # reserve the voja region
        spec.reserve_memory_region(
            self.DATA_REGIONS.VOJA.value,
            ((self.VOJA_REGION_N_ELEMENTS +
              (len(app_vertex.voja_learning_rules) *
               self.VOJA_REGION_RULE_N_ELEMENT)) *
             constants.BYTE_TO_WORD_MULTIPLIER))

        # reserve recording region
        spec.reserve_memory_region(
            region=self.DATA_REGIONS.RECORDING.value,
            size=recording_utilities.get_recording_header_size(
                len(app_vertex.get_possible_probeable_variables())))

    @overrides(AbstractHasAssociatedBinary.get_binary_start_type)
    def get_binary_start_type(self):
        return ExecutableType.USES_SIMULATION_INTERFACE

    @property
    @overrides(AbstractNengoMachineVertex.resources_required)
    def resources_required(self):
        return self._resources

    @overrides(AbstractHasAssociatedBinary.get_binary_file_name)
    def get_binary_file_name(self):
        return "lif.aplx"

    @overrides(AbstractReceiveBuffersToHost.get_recording_region_base_address)
    def get_recording_region_base_address(self, txrx, placement):
        return fec_helpful_functions.locate_memory_region_for_placement(
            placement, self.DATA_REGIONS.RECORDING.value, txrx)

    @overrides(AbstractReceiveBuffersToHost.get_recorded_region_ids)
    def get_recorded_region_ids(self):
        return recording_utilities.get_recorded_region_ids(
            self._buffered_sdram_per_timestep)

    @overrides(AbstractReceiveBuffersToHost.get_n_timesteps_in_buffer_space)
    def get_n_timesteps_in_buffer_space(self, buffer_space, machine_time_step):
        safe_space = buffer_space - self._overflow_sdram
        return recording_utilities.get_n_timesteps_in_buffer_space(
            safe_space, self._buffered_sdram_per_timestep)

    @overrides(AbstractReceiveBuffersToHost.get_minimum_buffer_sdram_usage)
    def get_minimum_buffer_sdram_usage(self):
        return sum(self._minimum_buffer_sdram_usage)