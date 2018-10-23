from enum import Enum
import numpy

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
from spinn_front_end_common.interface.simulation import simulation_utilities
from spinn_front_end_common.utilities.utility_objs import ExecutableType
from spinn_front_end_common.utilities import constants as fec_constants

from spinn_utilities.overrides import overrides


class LIFMachineVertex(
        AbstractNengoMachineVertex, MachineDataSpecableVertex,
        AbstractHasAssociatedBinary, AbstractAcceptsMulticastSignals,
        AbstractTransmitsMulticastSignals):

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
        "_local_pes_learning_rules"
    ]

    DATA_REGIONS = Enum(
        value="DATA_REGIONS",
        names=[
            ('SYSTEM', 0),
            ('ENSEMBLE_PARAMS', 1),  # Ensemble, NEURON, POP length
            ('ENCODER', 2),
            ('BIAS', 3),
            ('GAIN', 4),
            ('DECODER', 5),
            ('LEARNT_DECODER', 6),
            ('KEYS', 7),  # KEYS and LEARNT KEYS
            ('FILTERS', 8),  # INPUT FILTERS, INHIB FILTERS, MODULATORY  FILTERS, LEARNT ENDCODER FILTERS
            ('ROUTING', 9),  # INPUT ROUTING, INHIB ROUTING, MOD ROUTING, LEANT ENCODER ROUTING
            ('PES', 10),
            ('VOJA', 11),
            ('RECORDING', 12)  # only one for SPike Voltage Encoder
           ])  # 26

    # SDRAM calculation
    ENSEMBLE_PARAMS_ITEMS = 17
    SDRAM_ITEMS_PER_LEARNT_INPUT_VECTOR = 2
    NEURON_PARAMS_ITEMS = 2
    POP_LENGTH_CONSTANT_ITEMS = 1
    PES_REGION_N_ELEMENTS = 1
    VOJA_REGION_N_ELEMENTS = 2
    VOJA_REGION_RULE_N_ELEMENT = 5
    PES_REGION_SLICED_RULE_N_ELEMENTS = 6
    SHARED_SDRAM_FOR_SEMAPHORES_IN_BYTES = 4

    FUNCTION_OF_NEURON_TIME_CONSTANT = (1.0 - 2**-11)

    def __init__(
            self, sub_population_id, neuron_slice, input_slice, output_slice,
            learnt_slice, resources, encoders_with_gain, tau_rc, tau_refactory,
            ensemble_size_in, label, learnt_encoder_filters, input_filters,
            inhibitory_filters, modulatory_filters, pes_learning_rules):
        AbstractNengoMachineVertex.__init__(self, label=label)
        MachineDataSpecableVertex.__init__(self)
        AbstractHasAssociatedBinary.__init__(self)
        AbstractAcceptsMulticastSignals.__init__(self)
        AbstractTransmitsMulticastSignals.__init__(self)

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
                   "machine_time_step_in_seconds": "MachineTimeStepInSeconds"})
    @overrides(
        MachineDataSpecableVertex.generate_machine_data_specification,
        additional_arguments=["graph_mapper", "machine_time_step_in_seconds"])
    def generate_machine_data_specification(
            self, spec, placement, machine_graph, routing_info, iptags,
            reverse_iptags, machine_time_step, time_scale_factor,
            graph_mapper, machine_time_step_in_seconds):

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
        self._write_pes_region(spec)

        # process the voja region
        spec.switch_write_focus(self.DATA_REGIONS.VOJA.value)
        self._write_voja_region(spec)

        spec.end_specification()

    def _write_pes_region(self, spec):
        pass

    def _write_voja_region(self, spec):
        pass

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
        spec.write_value(outgoing_partition.sdram_base_address)
        spec.write_value(machine_graph_edge.sdram_base_address)

        # my local spike point
        machine_graph_edge = \
            machine_graph.get_edges_ending_at_vertex_with_partition_name(
                self,
                app_vertex.SDRAM_OUTGOING_SPIKE_VECTOR)[0]
        outgoing_partition = \
            machine_graph.get_outgoing_partition_for_edge(machine_graph_edge)
        spec.write_value(outgoing_partition.sdram_base_address)
        spec.write_value(machine_graph_edge.sdram_base_address)

        # the semaphore point
        spec.write_value(
            machine_graph.get_edges_ending_at_vertex_with_partition_name(
                self,
                app_vertex.SDRAM_OUTGOING_SEMAPHORE)[0].sdram_base_address)

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
