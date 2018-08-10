import numpy
import math

from collections import defaultdict

from enum import Enum

from nengo.learning_rules import PES as NengoPES
from nengo.learning_rules import Voja as NengoVoja
from nengo_spinnaker_gfe.abstracts.abstract_supports_nengo_partitioner import \
    AbstractSupportNengoPartitioner
from nengo_spinnaker_gfe.learning_rules.pes_learning_rule import PESLearningRule
from nengo_spinnaker_gfe.learning_rules.voja_learning_rule import \
    VojaLearningRule
from nengo_spinnaker_gfe.machine_vertices.lif_machine_vertex import \
    LIFMachineVertex
from nengo_spinnaker_gfe.nengo_filters.\
    filter_and_routing_region_generator import FilterAndRoutingRegionGenerator
from pacman.executor.injection_decorator import inject_items
from pacman.model.graphs.common import Slice
from pacman.model.resources import CPUCyclesPerTickResource, DTCMResource, \
    ResourceContainer, SDRAMResource
from spinn_machine import Processor, SDRAM
from spinn_utilities.overrides import overrides
from nengo_spinnaker_gfe import constants, helpful_functions
from nengo_spinnaker_gfe.abstracts. \
    abstract_nengo_application_vertex import \
    AbstractNengoApplicationVertex
from nengo_spinnaker_gfe.connection_parameters. \
    ensemble_transmission_parameters import \
    EnsembleTransmissionParameters
from nengo_spinnaker_gfe.nengo_exceptions import NengoException

from nengo_spinnaker_gfe.abstracts.abstract_probeable import AbstractProbeable


class LIFApplicationVertex(
        AbstractNengoApplicationVertex, AbstractProbeable,
        AbstractSupportNengoPartitioner):

    __slots__ = [
        "_eval_points",
        "_encoders",
        "_scaled_encoders",
        "_max_rates",
        "_intercepts",
        "_gain",
        "_bias",
        "_probeable_variables",
        "_is_recording_probeable_variable",
        "_probeable_variables_supported_elsewhere",
        "_direct_input",
        "_n_neurons",
        "_cluster_size_out",
        "_cluster_size_in",
        "_cluster_learnt_size_out",
        "_ensemble_size_in",
        "_n_profiler_samples",
        "_n_neurons_in_current_cluster",
        "_encoders_with_gain",
        "_max_resources_to_use_per_core",
        "_learnt_encoder_filters",
        "_pes_learning_rules",
        "_voja_learning_rules",
        "_decoders",
        "_learnt_decoders",
        "_n_output_keys",
        "_n_learnt_output_keys",
        "_input_filters",
        "_inhibition_filters",
        "_modulatory_filters",
        "_inputs_n_keys",
        "_inhibition_n_keys",
        "_modulatory_n_keys",
        "_learnt_encoders_n_keys"]

    ENSEMBLE_PROFILER_TAGS = Enum(
        value="PROFILER_TAGS",
        names=[("INPUT_FILTER", 0),
               ("NEURON_UPDATE", 1),
               ("DECODE_AND_TRANSMIT_OUTPUT", 2),
               ("PES", 3),
               ("VOJA", 4)])

    # sdp ports used by c code, to track with fec sdp ports.
    SLICES_POSITIONS = Enum(
        value="POSITIONS_OF_SPECIFIC_SLICES_IN_LIST",
        names=[
            ("INPUT", 0),
            ("NEURON", 1),
            ("OUTPUT", 2),
            ("LEARNT_OUTPUT", 3)])

    # flag saying if the ensemble can operate over multiple chips
    ENSEMBLE_PARTITIONING_OVER_MULTIPLE_CHIPS = False

    # expected resource limits to allow collaboration cores to work
    MAX_DTCM_USAGE_PER_CORE = 0.75
    MAX_CPU_USAGE_PER_CORE = 0.4
    MAX_SDRAM_USAGE_PER_CORE = 0.0625

    # magic numbers from mundy's thesis, no idea what they are, or how they
    #  were derived from.
    INPUT_FILTERING_CYCLES_1 = 39
    INPUT_FILTERING_CYCLES_2 = 135
    NEURON_UPDATE_CYCLES_1 = 9
    NEURON_UPDATE_CYCLES_2 = 61
    NEURON_UPDATE_CYCLES_3 = 174
    DECODE_AND_TRANSMIT_CYCLES_1 = 2
    DECODE_AND_TRANSMIT_CYCLES_2 = 143
    DECODE_AND_TRANSMIT_CYCLES_3 = 173

    # dtcm calculation
    DTCM_BYTES_PER_NEURON = 3

    # SDRAM requirements
    ENSEMBLE_REGION_N_ELEMENTS = 18
    LIF_REGION_N_ELEMENTS = 2
    PES_REGION_N_ELEMENTS = 1
    PES_REGION_SLICED_RULE_N_ELEMENTS = 6
    VOJA_REGION_N_ELEMENTS = 2
    VOJA_REGION_RULE_N_ELEMENT = 5
    MATRIX_REGIONS_PARTITION_INDEX = 0
    POPULATION_LENGTH_REGION_SIZE_IN_BYTES = 4
    PROFILER_SAMPLE_SIZE = 2
    PROFILER_N_SAMPLES_SIZE = 1
    FILTER_PARAMETERS_SIZE = 4
    FILTER_N_FILTERS_SIZE = 1
    ROUTING_N_ROUTES_SIZE = 1
    ROUTING_ENTRIES_PER_ROUTE = 4

    def __init__(
            self, label, rng, seed, eval_points, encoders, scaled_encoders,
            max_rates, intercepts, gain, bias, size_in, n_neurons,
            utilise_extra_core_for_output_types_probe, n_profiler_samples):
        """ constructor for lifs
        
        :param label: label of the vertex
        :param rng: random number generator
        :param eval_points: ????
        :param encoders: ????
        :param scaled_encoders: ??? 
        :param max_rates: ????
        :param intercepts: ????
        :param gain: ????
        :param bias: ????
        """
        AbstractNengoApplicationVertex.__init__(
            self, label=label, rng=rng, seed=seed)
        self._eval_points = eval_points
        self._encoders = encoders
        self._scaled_encoders = scaled_encoders
        self._encoders_with_gain = scaled_encoders
        self._max_rates = max_rates
        self._intercepts = intercepts
        self._gain = gain
        self._bias = bias
        self._direct_input = numpy.zeros(size_in)
        self._ensemble_size_in = size_in
        self._n_neurons = n_neurons

        # params to be used during partitioning
        self._cluster_size_out = None
        self._cluster_size_in = None
        self._cluster_learnt_size_out = None
        self._n_profiler_samples = n_profiler_samples
        self._n_neurons_in_current_cluster = None
        self._pes_learning_rules = list()
        self._voja_learning_rules = list()
        self._decoders = numpy.array([])
        self._learnt_decoders = None
        self._n_output_keys = 0
        self._n_learnt_output_keys = 0
        self._inputs_n_keys = 0
        self._inhibition_n_keys = 0
        self._modulatory_n_keys = 0
        self._learnt_encoders_n_keys = 0
        self._input_filters = defaultdict(list)
        self._inhibition_filters = defaultdict(list)
        self._modulatory_filters = defaultdict(list)
        self._learnt_encoder_filters = defaultdict(list)

        # build max resources planned to be used per machine vertex when
        # partitioning occurs
        self._max_resources_to_use_per_core = ResourceContainer(
            dtcm=DTCMResource(
                int(math.ceil(
                    Processor.DTCM_AVAILABLE * self.MAX_DTCM_USAGE_PER_CORE))),
            cpu_cycles=CPUCyclesPerTickResource(
                int(math.ceil(
                    Processor.CLOCK_SPEED * self.MAX_CPU_USAGE_PER_CORE))),
            sdram=SDRAMResource(
                int(math.floor(
                    SDRAM.DEFAULT_SDRAM_BYTES *
                    self.MAX_SDRAM_USAGE_PER_CORE))))

        # the variables probeable
        self._probeable_variables = [
            constants.RECORD_OUTPUT_FLAG, constants.RECORD_SPIKES_FLAG,
            constants.RECORD_VOLTAGE_FLAG, constants.SCALED_ENCODERS_FLAG]

        # update all recordings to false
        self._is_recording_probeable_variable = dict()
        for flag in self._probeable_variables:
            self._is_recording_probeable_variable[flag] = False

        # add extra probes based off recording on core or not
        if not utilise_extra_core_for_output_types_probe:
            self._probeable_variables.append(
                constants.DECODER_OUTPUT_FLAG)
            self._is_recording_probeable_variable[
                constants.DECODER_OUTPUT_FLAG] = False

    @property
    def direct_input(self):
        return self._direct_input

    @direct_input.setter
    def direct_input(self, new_value):
        self._direct_input = new_value

    @property
    def eval_points(self):
        return self._eval_points

    @property
    def encoders(self):
        return self._encoders

    @property
    def scaled_encoders(self):
        return self._scaled_encoders

    @property
    def max_rates(self):
        return self._max_rates

    @property
    def intercepts(self):
        return self._intercepts

    @property
    def gain(self):
        return self._gain

    @property
    def bias(self):
        return self._bias

    @overrides(AbstractProbeable.set_probeable_variable)
    def set_probeable_variable(self, variable):
        self._is_recording_probeable_variable[variable] = True

    @overrides(AbstractProbeable.get_data_for_variable)
    def get_data_for_variable(self, variable):
        pass

    @overrides(AbstractProbeable.can_probe_variable)
    def can_probe_variable(self, variable):
            if variable in self._probeable_variables:
                return True
            else:
                return False

    

    @inject_items({"operator_graph": "NengoOperatorGraph",
                   "machine_time_step": "MachineTimeStep"})
    @overrides(
        AbstractNengoApplicationVertex.create_machine_vertices,
        additional_arguments=["operator_graph", "machine_time_step"])
    def create_machine_vertices(
            self, resource_tracker, nengo_partitioner,
            operator_graph, machine_time_step):

        machine_vertices = list()

        outgoing_partitions = operator_graph.\
            get_outgoing_edge_partitions_starting_at_vertex(self)
        incoming_edges = operator_graph.get_edges_ending_at_vertex(self)

        standard_outgoing_partitions = list()
        outgoing_learnt_partitions = list()
        incoming_learnt_edges = list()
        incoming_modulatory_learning_rules = dict()
        incoming_standard_edges = list()
        incoming_global_inhibition_edges = list()

        # filter incoming partitions
        for in_edge in incoming_edges:
            # verify there's no neurons incoming partitions
            if in_edge.input_port.destination_input_port == \
                    constants.ENSEMBLE_INPUT_PORT.NEURONS:
                raise Exception("not suppose to have neurons incoming")

            # locate all modulating incoming partitions learning rules
            if (in_edge.input_port.destination_input_port ==
                    constants.ENSEMBLE_INPUT_PORT.LEARNING_RULE or
                    in_edge.input_port.destination_input_port ==
                    constants.ENSEMBLE_INPUT_PORT.LEARNT):
                if in_edge.reception_parameters.learning_rule is not None:
                    incoming_modulatory_learning_rules[
                        in_edge.reception_parameters.learning_rule] = in_edge
                else:
                    incoming_modulatory_learning_rules[
                        in_edge.input_port.learning_rule] = in_edge

            # build map of edges and learning rule
            if (in_edge.input_port.destination_input_port ==
                    constants.ENSEMBLE_INPUT_PORT.LEARNT):
                incoming_learnt_edges.append(
                    (in_edge, in_edge.reception_parameters.learning_rule))

            if (in_edge.input_port.destination_input_port ==
                    constants.INPUT_PORT.STANDARD):
                incoming_standard_edges.append(in_edge)

            if (in_edge.input_port.destination_input_port ==
                    constants.ENSEMBLE_INPUT_PORT.GLOBAL_INHIBITION):
                incoming_global_inhibition_edges.append(in_edge)

        # filter outgoing partitions
        for outgoing_partition in outgoing_partitions:
            # locate all standard outgoing partitions
            if outgoing_partition.identifier.source_port == \
                    constants.OUTPUT_PORT.STANDARD:
                standard_outgoing_partitions.append(outgoing_partition)

            # locate all learnt partitions
            if outgoing_partition.identifier.source_port == \
                    constants.ENSEMBLE_OUTPUT_PORT.LEARNT:
                outgoing_learnt_partitions.append(outgoing_partition)

        # locate decoders and n keys
        if len(standard_outgoing_partitions) != 0:
            self._decoders, self._n_output_keys = self._get_decoders_and_n_keys(
                standard_outgoing_partitions, True)

        # convert to cluster sizes
        self._cluster_size_out = self._decoders.shape[0]
        self._cluster_size_in = self._scaled_encoders.shape[1]
        self._cluster_learnt_size_out = self._determine_cluster_learnt_size_out(
            outgoing_learnt_partitions, incoming_modulatory_learning_rules,
            machine_time_step)

        # locate incoming voja learning rules
        self._determine_voja_learning_rules_and_modulatory_filters(
            incoming_learnt_edges, incoming_modulatory_learning_rules,
            operator_graph)

        # create input filters
        for input_edge in incoming_standard_edges:
            FilterAndRoutingRegionGenerator.add_filters(
                self._input_filters, input_edge,
                operator_graph.get_outgoing_partition_for_edge(input_edge),
                minimise=True)
            self._inputs_n_keys += 1

        # create inhibition filters
        for global_inhibition_edge in incoming_global_inhibition_edges:
            FilterAndRoutingRegionGenerator.add_filters(
                self._inhibition_filters, global_inhibition_edge,
                operator_graph.get_outgoing_partition_for_edge(
                    global_inhibition_edge), minimise=True)
            self._inhibition_n_keys += 1

        # start the partitioning process, now that all the data required to
        # do so has been deduced
        n_atoms_partitioned = 0
        while n_atoms_partitioned < self._n_neurons:
            max_cores = resource_tracker.get_maximum_cores_available_on_a_chip()

            # if supporting ensembles over multiple chips, do cluster
            # partitioning. else assume one chip and partition accordingly.
            cluster_vertices = list()
            if self.ENSEMBLE_PARTITIONING_OVER_MULTIPLE_CHIPS:
                slices = nengo_partitioner.create_slices(
                    Slice(0, self._n_neurons - 1), self,
                    self._max_resources_to_use_per_core, max_cores,
                    self._n_neurons)
                for neuron_slice in slices:
                    self._n_neurons_in_current_cluster = (
                        neuron_slice.hi_atom - neuron_slice.lo_atom)
                    cluster_vertices = self._create_cluster_and_verts(
                        neuron_slice, max_cores, nengo_partitioner)
                    machine_vertices.extend(cluster_vertices)
            else:
                self._n_neurons_in_current_cluster = self._n_neurons
                cluster_vertices = self._create_cluster_and_verts(
                    Slice(0, self._n_neurons - 1), max_cores, nengo_partitioner)
                machine_vertices.extend(cluster_vertices)

            # update the atom tracker
            n_atoms_partitioned += self._n_neurons_in_current_cluster

            # update the resource tracker to take out a valid chip from the
            # available set.
            group_resource = list()
            for vertex in cluster_vertices:
                group_resource.append(
                    (vertex.resources_required, vertex.constraints))

            # allocate resources to resource tracker, to ensure next vertex
            # doesnt partition on incorrect data
            resource_tracker.allocate_constrained_group_resources(
                group_resource)
        return machine_vertices

    def _create_cluster_and_verts(
            self, neuron_slice, max_cores, nengo_partitioner):

        cluster_vertices = list()

        # Partition the slice of neurons that we have
        sliced_objects = [
            Slice(0, int(self._encoders_with_gain.shape[1])),  # Input subspace
            neuron_slice,  # Neurons
            Slice(0, int(self._decoders.shape[0])),  # Outputs
            Slice(0, len(self._learnt_encoder_filters))  # Learnt output
        ]

        # create core sized partitions
        all_slices_and_resources = list(
            nengo_partitioner.create_slices_for_multiple(
                sliced_objects, self, self._max_resources_to_use_per_core,
                max_cores, max_cores))

        neuron_slices = list()
        for ((_, neuron_slice, _, _), used_resources) in \
                all_slices_and_resources:
            neuron_slices.append(neuron_slice)

        for vertex_index, (
                (input_slice, neuron_slice, output_slice, learnt_slice),
                used_resources) in enumerate(all_slices_and_resources):
            vertex = LIFMachineVertex(
                vertex_index, neuron_slices, input_slice, output_slice,
                learnt_slice, used_resources)
            cluster_vertices.append(vertex)
        return cluster_vertices

    def _get_input_filtering_cycles(self, size_in):
        """Cycles required to perform filtering of received values."""
        # Based on thesis profiling
        return (self.INPUT_FILTERING_CYCLES_1 * size_in +
                self.INPUT_FILTERING_CYCLES_2)

    def _get_neuron_update_cycles(self, size_in, n_neurons_on_core):
        """Cycles required to simulate neurons."""
        # Based on thesis profiling
        return (self.NEURON_UPDATE_CYCLES_1 * n_neurons_on_core * size_in +
                self.NEURON_UPDATE_CYCLES_2 * n_neurons_on_core +
                self.NEURON_UPDATE_CYCLES_3)

    def _get_decode_and_transmit_cycles(self, n_neurons_in_cluster, size_out):
        """Cycles required to decode spikes and transmit packets."""
        # Based on thesis profiling
        return (self.DECODE_AND_TRANSMIT_CYCLES_1 * n_neurons_in_cluster *
                size_out + self.DECODE_AND_TRANSMIT_CYCLES_2 * size_out +
                self.DECODE_AND_TRANSMIT_CYCLES_3)

    def _get_sliced_learning_rules(self, learnt_output_slice):
        return [l for l in self._pes_learning_rules
                if (l.decoder_start < learnt_output_slice.hi_atom and
                    l.decoder_stop > learnt_output_slice.lo_atom)]

    @overrides(AbstractSupportNengoPartitioner.get_resources_for_slices)
    def get_resources_for_slices(self, slices, n_cores):
        return ResourceContainer(
            cpu_cycles=self._cpu_usage_for_slices(slices, n_cores),
            dtcm=self._dtcm_usage_for_slices(slices, n_cores),
            sdram=self._sdram_usage_for_slices(slices, n_cores))

    def _sdram_for_filter(self, filters):
        total = 0
        total_n_filters = 0
        for outgoing_partition in filters:
            for input_filter in filters[outgoing_partition]:
                total += input_filter.size_words()
                total_n_filters += 1
        total += (
            (self.FILTER_N_FILTERS_SIZE + (
                self.FILTER_PARAMETERS_SIZE * total_n_filters)) *
            constants.BYTE_TO_WORD_MULTIPLIER)
        return total

    def _sdram_for_routing_region(self, n_keys):
        return ((self.ROUTING_N_ROUTES_SIZE + (
            self.ROUTING_ENTRIES_PER_ROUTE * n_keys)) *
                constants.BYTE_TO_WORD_MULTIPLIER)

    def _sdram_usage_for_slices(self, slices, n_cores):
        # build slices accordingly
        if len(slices) == 1:
            neuron_slice = slices[0]
            output_slice = Slice(0, int(self._decoders.shape[0]))
            learnt_output_slice = Slice(0, len(self._learnt_encoder_filters))

        else:
            neuron_slice = slices[self.SLICES_POSITIONS.NEURON.value]
            learnt_output_slice = slices[
                self.SLICES_POSITIONS.LEARNT_OUTPUT.value]
            output_slice = slices[self.SLICES_POSITIONS.OUTPUT.value]

        # pes learning rule region
        pes_region = (
            self.PES_REGION_N_ELEMENTS + len(
                self._get_sliced_learning_rules(learnt_output_slice)) +
            self.PES_REGION_SLICED_RULE_N_ELEMENTS)

        # constant based regions
        ensemble_region = ((self.ENSEMBLE_REGION_N_ELEMENTS +
                            len(self._learnt_encoder_filters)) *
                           constants.BYTE_TO_WORD_MULTIPLIER)
        lif_region = (
            self.LIF_REGION_N_ELEMENTS * constants.BYTE_TO_WORD_MULTIPLIER)
        voja_region = ((self.VOJA_REGION_N_ELEMENTS +
                        (len(self._voja_learning_rules) *
                         self.VOJA_REGION_RULE_N_ELEMENT)) *
                       constants.BYTE_TO_WORD_MULTIPLIER)

        # matrix based regions
        decoders_region = self._decoders[helpful_functions.expand_slice(
            output_slice, self.MATRIX_REGIONS_PARTITION_INDEX,
            self._decoders.ndim)].nbytes

        learnt_decoders_region = self._learnt_decoders[
            helpful_functions.expand_slice(
                learnt_output_slice, self.MATRIX_REGIONS_PARTITION_INDEX,
                self._learnt_decoders.ndim)].nbytes

        encoders_region = self._encoders_with_gain[
            helpful_functions.expand_slice(
                neuron_slice, self.MATRIX_REGIONS_PARTITION_INDEX,
                self._encoders_with_gain.ndim)].nbytes

        bias_region = self._bias[helpful_functions.expand_slice(
            neuron_slice, self.MATRIX_REGIONS_PARTITION_INDEX,
            self._bias.ndim)].nbytes

        gain_region = self._gain[helpful_functions.expand_slice(
            neuron_slice, self.MATRIX_REGIONS_PARTITION_INDEX,
            self._gain.ndim)].nbytes

        # basic key regions
        key_region = constants.BYTES_PER_KEY * self._n_output_keys
        learnt_key_region = constants.BYTES_PER_KEY * self._n_learnt_output_keys

        # partitioning data for the machine vertices
        population_length_region = (
            self.POPULATION_LENGTH_REGION_SIZE_IN_BYTES * n_cores)

        # filter regions
        input_filter_region = self._sdram_for_filter(self._input_filters)
        inhib_filter_region = self._sdram_for_filter(self._inhibition_filters)
        modulatory_filters_region = self._sdram_for_filter(
            self._modulatory_filters)
        learnt_encoder_filters_region = self._sdram_for_filter(
            self._learnt_encoder_filters)

        # routing regions
        input_routing_region = \
            self._sdram_for_routing_region(self._inputs_n_keys)
        inhib_routing_region = \
            self._sdram_for_routing_region(self._inhibition_n_keys)
        modulatory_routing_region = \
            self._sdram_for_routing_region(self._modulatory_n_keys)
        learnt_encoder_routing_region = \
            self._sdram_for_routing_region(self._learnt_encoders_n_keys)

        # profile region
        profiler_region = (self.PROFILER_N_SAMPLES_SIZE + (
            self.PROFILER_SAMPLE_SIZE * self._n_profiler_samples) *
            constants.BYTE_TO_WORD_MULTIPLIER)

        # recording region
        #TODO tie into buffering
        recording_region_size = 0

        total = (
            ensemble_region + lif_region + pes_region + voja_region +
            decoders_region + learnt_decoders_region + encoders_region +
            bias_region + gain_region + key_region + learnt_key_region +
            population_length_region + profiler_region + input_filter_region
            + inhib_filter_region + modulatory_filters_region +
            learnt_encoder_filters_region + input_routing_region +
            inhib_routing_region + modulatory_routing_region +
            learnt_encoder_routing_region + recording_region_size)

        if len(slices) == 1:
            return SDRAMResource(total / n_cores)
        else:
            return SDRAMResource(total)

    def _dtcm_usage_for_slices(self, slices, n_cores):
        # handles clusters
        if len(slices) == 1:
            neuron_slice = slices[0]
            size_learnt_out_per_core = \
                int(math.ceil(float(self._cluster_learnt_size_out) / n_cores))
            size_out_per_core = \
                int(math.ceil((float(self._cluster_size_out) / n_cores)))

            n_neurons = neuron_slice.hi_atom - neuron_slice.lo_atom
            neurons_per_core = int(math.ceil((float(n_neurons) / n_cores)))

            encoder_cost = neurons_per_core * self._cluster_size_in
            decoder_cost = n_neurons * (
                size_out_per_core + size_learnt_out_per_core)
            neurons_cost = neurons_per_core * self.DTCM_BYTES_PER_NEURON

            return DTCMResource(
                (encoder_cost + decoder_cost + neurons_cost) *
                constants.BYTE_TO_WORD_MULTIPLIER)
        else:  # handling core size chunks
            """Get the amount of memory required."""
            neuron_slice = slices[self.SLICES_POSITIONS.NEURON.value]
            output_slice = slices[self.SLICES_POSITIONS.OUTPUT.value]
            learnt_output_slice = slices[
                self.SLICES_POSITIONS.LEARNT_OUTPUT.value]

            n_neurons = neuron_slice.hi_atom - neuron_slice.lo_atom
            size_out = output_slice.hi_atom - output_slice.lo_atom
            size_learnt_out = (
                learnt_output_slice.hi_atom - learnt_output_slice.lo_atom)

            encoder_cost = n_neurons * self._encoders_with_gain.shape[1]
            decoder_cost = (
                self._n_neurons_in_current_cluster * (
                    size_out + size_learnt_out))
            neurons_cost = n_neurons * self.DTCM_BYTES_PER_NEURON

            return DTCMResource(
                (encoder_cost + decoder_cost + neurons_cost) *
                constants.BYTE_TO_WORD_MULTIPLIER)

    def _cpu_usage_for_slices(self, slices, n_cores):
        # handles clusters
        if len(slices) == 1:
            neuron_slice = slices[0]
            # Compute the number of neurons in the slice and the number
            # allocated to the most heavily loaded core.
            n_neurons = neuron_slice.hi_atom - neuron_slice.lo_atom
            neurons_per_core = int(math.ceil((float(n_neurons) / n_cores)))
            size_in_per_core = \
                int(math.ceil((float(self._cluster_size_in) / n_cores)))
            size_out_per_core = \
                int(math.ceil((float(self._cluster_size_out) / n_cores)))
            size_learnt_out_per_core = \
                int(math.ceil(float(self._cluster_learnt_size_out) / n_cores))

            # Compute the loading
            # TODO Profile PES and Voja
            return CPUCyclesPerTickResource(
                self._get_input_filtering_cycles(size_in_per_core) +
                self._get_neuron_update_cycles(
                    self._cluster_size_in, neurons_per_core) +
                self._get_decode_and_transmit_cycles(
                    n_neurons, size_out_per_core) +
                self._get_decode_and_transmit_cycles(
                    n_neurons, size_learnt_out_per_core))
        else:  # handling core size chunks
            filtered_dims_slice = slices[self.SLICES_POSITIONS.INPUT.value]
            neuron_slice = slices[self.SLICES_POSITIONS.NEURON.value]
            output_slice = slices[self.SLICES_POSITIONS.OUTPUT.value]
            learnt_output_slice = slices[
                self.SLICES_POSITIONS.LEARNT_OUTPUT.value]

            n_neurons = neuron_slice.hi_atom - neuron_slice.lo_atom
            size_out = output_slice.hi_atom - output_slice.lo_atom
            size_learnt_out = (
                learnt_output_slice.hi_atom - learnt_output_slice.lo_atom)
            filtered_dims = (
                filtered_dims_slice.hi_atom - filtered_dims_slice.lo_atom)

            # Compute the loading
            # TODO Profile PES and Voja
            return CPUCyclesPerTickResource(
                self._get_input_filtering_cycles(filtered_dims) +
                self._get_neuron_update_cycles(
                    self._encoders_with_gain.shape[1], n_neurons) +
                self._get_decode_and_transmit_cycles(
                    self._n_neurons_in_current_cluster, size_out) +
                self._get_decode_and_transmit_cycles(
                    self._n_neurons_in_current_cluster, size_learnt_out))

    def _determine_cluster_learnt_size_out(
            self, outgoing_learnt_partitions,
            incoming_modulatory_learning_rules, machine_time_step):
        self._learnt_decoders = numpy.array([])
        self._pes_learning_rules = list()

        for learnt_outgoing_partition in outgoing_learnt_partitions:
            partition_identifier = learnt_outgoing_partition.identifier
            transmission_parameter = partition_identifier.transmission_parameter
            learning_rule_type = \
                transmission_parameter.learning_rule.learning_rule_type

            # verify that the transmission parameter type is as expected
            if not isinstance(transmission_parameter,
                              EnsembleTransmissionParameters):
                raise NengoException(
                    "the ensemble {} expects a EnsembleTransmissionParameters "
                    "for its learning rules. got {} instead".format(
                        self, transmission_parameter))

            # verify that the learning rule is a PES rule
            if not isinstance(learning_rule_type, NengoPES):
                raise NengoException(
                    "The SpiNNaker Nengo Conversion currently only "
                    "supports PES learning rules")

            # verify that there's a modulatory connection to the learning
            #  rule
            if transmission_parameter.learning_rule not in \
                    incoming_modulatory_learning_rules.keys():
                raise NengoException(
                            "Ensemble %s has outgoing connection with PES "
                            "learning, but no corresponding modulatory "
                            "connection" % self.label)

            decoder_start = self._learnt_decoders.shape[0]

            rule_decoders, self._n_learnt_output_keys = \
                self._get_decoders_and_n_keys([learnt_outgoing_partition])

            # If there are no existing decodes, hstacking doesn't
            # work so set decoders to new learnt decoder matrix
            if decoder_start == 0:
                self._learnt_decoders = rule_decoders
            # Otherwise, stack learnt decoders
            # alongside existing matrix
            else:
                self._learnt_decoders = numpy.vstack(
                    (self._learnt_decoders, rule_decoders))

            decoder_stop = self._learnt_decoders.shape[0]

            pes_learning_rule = PESLearningRule(
                learning_rate=(
                    transmission_parameter.learning_rule.learning_rule_type.
                    learning_rate / machine_time_step),
                decoder_start=decoder_start,
                decoder_stop=decoder_stop)
            self._pes_learning_rules.append(pes_learning_rule)

            # Add error connection to lists of modulatory filters and routes
            modulatory_edge = incoming_modulatory_learning_rules[
                transmission_parameter.learning_rule]
            FilterAndRoutingRegionGenerator.add_filters(
                self._modulatory_filters, modulatory_edge,
                learnt_outgoing_partition, minimise=False)
            self._modulatory_n_keys += 1

            # Create a duplicate copy of the original size_in columns of
            # the encoder matrix for modification by this learning rule
            base_encoders = self._encoders_with_gain[:, :self._ensemble_size_in]
            self._encoders_with_gain = numpy.hstack(
                (self._encoders_with_gain, base_encoders))
        return self._learnt_decoders.shape[0]

    def _determine_voja_learning_rules_and_modulatory_filters(
            self, incoming_learnt_edges_and_learning_rules,
            incoming_modulatory_learning_rules, operator_graph):

        for (edge, learning_rule) in incoming_learnt_edges_and_learning_rules:

            if isinstance(learning_rule.learning_rule_type, NengoVoja):
                voja_learning_rule = VojaLearningRule(
                    learning_rule.learning_rule_type.learning_rate,
                    encoder_offset=self._encoders_with_gain.shape[1])
                self._voja_learning_rules.append(voja_learning_rule)

                # If there is a modulatory connection
                # associated with the learning rule
                if learning_rule in incoming_modulatory_learning_rules.keys():
                    learning_edge = incoming_modulatory_learning_rules[
                        learning_rule]

                    # Add learning connection to lists
                    # of modulatory filters and route
                    FilterAndRoutingRegionGenerator.add_filters(
                        self._modulatory_filters, learning_edge,
                        operator_graph.get_outgoing_partition_for_edge(
                            learning_edge),
                        minimise=False)
                    self._modulatory_n_keys += 1

                # Add learnt connection to list of filters
                # and routes with learnt encoders
                FilterAndRoutingRegionGenerator.add_filters(
                    self._learnt_encoder_filters, edge,
                    operator_graph.get_outgoing_partition_for_edge(edge),
                    minimise=False)
                self._learnt_encoders_n_keys += 1

    def _get_decoders_and_n_keys(
            self, standard_outgoing_partitions, minimise=False):

        decoders = list()
        n_keys = 0
        for standard_outgoing_partition in standard_outgoing_partitions:
            partition_identifier = standard_outgoing_partition.identifier
            if not isinstance(partition_identifier.transmission_parameter,
                              EnsembleTransmissionParameters):
                raise NengoException(
                    "To determine the decoders and keys, the ensemble {} "
                    "assumes it only has ensemble transmission params. this "
                    "was not the case.".format(self))
            decoder = partition_identifier.transmission_parameter.full_decoders
            if not minimise:
                keep = numpy.array([True for _ in range(decoder.shape[0])])
            else:
                # We can reduce the number of packets sent and the memory
                # requirements by removing columns from the decoder matrix which
                # will always result in packets containing zeroes.
                keep = numpy.any(decoder != 0, axis=1)
            decoders.append(decoder[keep, :])
            n_keys += decoder.shape[0]

        # Stack the decoders
        if len(decoders) > 0:
            decoders = numpy.vstack(decoders)
        else:
            decoders = numpy.array([[]])
        return decoders, n_keys
