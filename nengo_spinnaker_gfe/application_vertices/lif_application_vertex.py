import numpy
import math

from collections import defaultdict

from enum import Enum

from nengo.learning_rules import PES as NengoPES
from nengo.learning_rules import Voja as NengoVoja

from nengo_spinnaker_gfe.abstracts.abstract_probeable import AbstractProbeable
from nengo_spinnaker_gfe.abstracts.abstract_supports_nengo_partitioner import \
    AbstractSupportNengoPartitioner
from nengo_spinnaker_gfe.graph_components.connection_application_edge import \
    ConnectionApplicationEdge
from nengo_spinnaker_gfe.graph_components.connection_machine_edge import \
    ConnectionMachineEdge
from nengo_spinnaker_gfe.graph_components.\
    constant_sdram_machine_partition import \
    ConstantSDRAMMachinePartition
from nengo_spinnaker_gfe.graph_components.sdram_machine_edge import \
    SDRAMMachineEdge
from nengo_spinnaker_gfe.graph_components.\
    segmented_input_sdram_machine_partition import \
    SegmentedInputSDRAMMachinePartition
from nengo_spinnaker_gfe.graph_components.\
    segmented_spikes_sdram_machine_partition import \
    SegmentedSpikesSDRAMMachinePartition
from nengo_spinnaker_gfe.learning_rules.pes_learning_rule import \
    PESLearningRule
from nengo_spinnaker_gfe.learning_rules.voja_learning_rule import \
    VojaLearningRule
from nengo_spinnaker_gfe.machine_vertices.lif_machine_vertex import \
    LIFMachineVertex
from nengo_spinnaker_gfe.nengo_filters.\
    filter_and_routing_region_generator import FilterAndRoutingRegionGenerator
from nengo_spinnaker_gfe.overridden_mapping_algorithms.\
    nengo_partitioner import NengoPartitioner
from nengo_spinnaker_gfe.utility_objects.neuron_recorder import NeuronRecorder
from nengo_spinnaker_gfe import constants, helpful_functions
from nengo_spinnaker_gfe.abstracts. \
    abstract_nengo_application_vertex import \
    AbstractNengoApplicationVertex
from nengo_spinnaker_gfe.connection_parameters. \
    ensemble_transmission_parameters import \
    EnsembleTransmissionParameters
from nengo_spinnaker_gfe.nengo_exceptions import NengoException

from pacman.executor.injection_decorator import inject_items
from pacman.model.graphs.common import Slice
from pacman.model.resources import CPUCyclesPerTickResource, DTCMResource, \
    ResourceContainer, SDRAMResource

from spinn_machine import Processor, SDRAM

from spinn_utilities.overrides import overrides

from spinn_front_end_common.interface.buffer_management\
    import recording_utilities


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
        "_learnt_encoders_n_keys",
        "_tau_rc",
        "_tau_refactory",
        "_machine_vertex_slices",
        "_core_slice_to_chip_slice",
        "_radius"]

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

    # the outgoing partition ids for sdram partitions
    SDRAM_OUTGOING_INPUT = "SDRAM_LINK_FOR_INPUT_VECTOR"
    SDRAM_OUTGOING_LEARNT = "SDRAM_LINK_FOR_LEARNT_VECTOR"
    SDRAM_OUTGOING_SPIKE_VECTOR = "SDRAM_LINK_FOR_SPIKE"
    SDRAM_OUTGOING_SEMAPHORE = "SDRAM_LINK_FOR_SEMAPHORE"

    def __init__(
            self, label, rng, seed, eval_points, encoders, scaled_encoders,
            max_rates, intercepts, gain, bias, size_in, n_neurons, radius,
            utilise_extra_core_for_output_types_probe, tau_rc, tau_refactory):
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
        AbstractProbeable.__init__(self)
        AbstractSupportNengoPartitioner.__init__(self)

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
        self._radius = radius

        # neuron params
        self._tau_rc = tau_rc
        self._tau_refactory = tau_refactory

        # partition data
        self._machine_vertex_slices = defaultdict(list)
        self._core_slice_to_chip_slice = dict()

        # params to be used during partitioning
        self._cluster_size_out = None
        self._cluster_size_in = None
        self._cluster_learnt_size_out = None
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
            constants.RECORD_SPIKES_FLAG, constants.RECORD_VOLTAGE_FLAG,
            constants.SCALED_ENCODERS_FLAG]

        # update all recordings to false
        self._is_recording_probeable_variable = dict()
        for flag in self._probeable_variables:
            self._is_recording_probeable_variable[flag] = False

        self._neuron_recorder = NeuronRecorder(
            self._is_recording_probeable_variable.keys(), n_neurons)

        # add extra probes based off recording on core or not
        if not utilise_extra_core_for_output_types_probe:
            self._probeable_variables.append(
                constants.DECODER_OUTPUT_FLAG)
            self._is_recording_probeable_variable[
                constants.DECODER_OUTPUT_FLAG] = False

        self._maximum_sdram_for_buffering = None

    @property
    def input_n_keys(self):
        return self._inputs_n_keys

    @property
    def inhibition_n_keys(self):
        return self._inhibition_n_keys

    @property
    def modulatory_n_keys(self):
        return self._modulatory_n_keys

    @property
    def learnt_encoders_n_keys(self):
        return self._learnt_encoders_n_keys

    @property
    def output_n_keys(self):
        return self._n_output_keys

    @property
    def learnt_output_n_keys(self):
        return self._n_learnt_output_keys

    @property
    def machine_vertex_slices(self):
        return self._machine_vertex_slices

    @property
    def core_slice_to_chip_slice(self):
        return self._core_slice_to_chip_slice

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

    @property
    def n_neurons(self):
        return self._n_neurons

    @property
    def voja_learning_rules(self):
        return self._voja_learning_rules

    @property
    def maximum_sdram_for_buffering(self):
        return self._maximum_sdram_for_buffering

    @overrides(AbstractProbeable.set_probeable_variable)
    def set_probeable_variable(self, variable):
        if variable == constants.RECORD_OUTPUT_FLAG:
            variable = constants.RECORD_SPIKES_FLAG
        self._is_recording_probeable_variable[variable] = True

    @overrides(AbstractProbeable.is_set_probeable_variable)
    def is_set_probeable_variable(self, variable):
        if variable == constants.RECORD_OUTPUT_FLAG:
            variable = constants.RECORD_SPIKES_FLAG
        return self._is_recording_probeable_variable[variable]

    @overrides(AbstractProbeable.get_possible_probeable_variables)
    def get_possible_probeable_variables(self):
        return self._probeable_variables

    @overrides(AbstractProbeable.get_data_for_variable)
    def get_data_for_variable(
            self, variable, n_machine_time_steps, placements, graph_mapper,
            buffer_manager, machine_time_step):
        if variable == constants.RECORD_OUTPUT_FLAG:
            variable = constants.RECORD_SPIKES_FLAG
        return self._neuron_recorder.get_matrix_data(
            self.label, buffer_manager, self._probeable_variables[variable],
            placements, graph_mapper, self, variable, n_machine_time_steps)

    @overrides(AbstractProbeable.can_probe_variable)
    def can_probe_variable(self, variable):
        if variable == constants.RECORD_OUTPUT_FLAG:
            variable = constants.RECORD_SPIKES_FLAG

        if variable in self._probeable_variables:
            return True
        else:
            return False

    def _create_internal_data_maps(
            self, operator_graph, machine_time_step):

        standard_outgoing_partitions = list()
        outgoing_learnt_partitions = list()
        incoming_learnt_edges_and_learning_rule = list()
        incoming_standard_edges = list()
        incoming_global_inhibition_edges = list()

        outgoing_partitions = operator_graph.\
            get_outgoing_edge_partitions_starting_at_vertex(self)
        incoming_edges = operator_graph.get_edges_ending_at_vertex(self)

        # locate modulatory learning rules
        incoming_modulatory_learning_rules = \
            self.locate_all_modulatory_learning_rules(operator_graph, self)

        # filter incoming partitions
        for in_edge in incoming_edges:
            # verify there's no neurons incoming partitions
            if in_edge.input_port.destination_input_port == \
                    constants.ENSEMBLE_INPUT_PORT.NEURONS:
                raise Exception("not suppose to have neurons incoming")

            # build map of edges and learning rule
            if (in_edge.input_port.destination_input_port ==
                    constants.ENSEMBLE_INPUT_PORT.LEARNT):
                incoming_learnt_edges_and_learning_rule.append(
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
            machine_time_step, operator_graph)

        # locate incoming voja learning rules
        self._determine_voja_learning_rules_and_modulatory_filters(
            incoming_learnt_edges_and_learning_rule,
            incoming_modulatory_learning_rules, operator_graph)

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

    @staticmethod
    def locate_all_modulatory_learning_rules(graph, vertex):
        incoming_modulatory_learning_rules = dict()
        for incoming_edge in graph.get_edges_ending_at_vertex(vertex):
            if (isinstance(incoming_edge, ConnectionMachineEdge) or
                    isinstance(incoming_edge, ConnectionApplicationEdge)):
                if (incoming_edge.input_port.destination_input_port ==
                        constants.ENSEMBLE_INPUT_PORT.LEARNING_RULE or
                        incoming_edge.input_port.destination_input_port ==
                        constants.ENSEMBLE_INPUT_PORT.LEARNT):
                    if (incoming_edge.reception_parameters.learning_rule is
                            not None):
                        incoming_modulatory_learning_rules[
                            incoming_edge.reception_parameters.
                            learning_rule] = incoming_edge
                    else:
                        incoming_modulatory_learning_rules[
                            incoming_edge.input_port.learning_rule] = \
                            incoming_edge
        return incoming_modulatory_learning_rules

    @overrides(AbstractSupportNengoPartitioner.get_shared_resources_for_slices)
    def get_shared_resources_for_slices(self, slices):

        # deduce the shared sdram requirements per cluster.
        shared_sdram_requirements = (
            LIFMachineVertex.SHARED_SDRAM_FOR_SEMAPHORES_IN_BYTES +
            (self._ensemble_size_in * constants.BYTE_TO_WORD_MULTIPLIER))
        for _ in range(len(self._learnt_encoder_filters)):
            shared_sdram_requirements += (
                self._ensemble_size_in * constants.BYTE_TO_WORD_MULTIPLIER)
        shared_sdram_requirements += self._get_bytes_for_unpacked_spike_vector(
            slices)

        shared_sdram_requirements = SDRAMResource(shared_sdram_requirements)
        return ResourceContainer(sdram=shared_sdram_requirements)

    def _get_bytes_for_unpacked_spike_vector(self, slices):
        """Compute the number of bytes necessary to store the unpacked spike \
        vector for the given ranges of neurons.
        """
        words = 0

        for internal_slices in slices:
            # Get the number of neurons
            neuron_slice = internal_slices[self.SLICES_POSITIONS.NEURON.value]

            n_neurons = neuron_slice.hi_atom - neuron_slice.lo_atom

            # Update the word count, padding to allow an integral number of \
            # words for each slice.
            words += math.ceil(n_neurons // constants.WORD_TO_BIT_CONVERSION)

        # Multiply to get bytes
        return words * constants.BYTE_TO_WORD_MULTIPLIER

    @inject_items({
        "operator_graph": "NengoOperatorGraph",
        "machine_time_step": "MachineTimeStep",
        "n_machine_time_steps": "TotalMachineTimeSteps",
        "machine_time_step_in_seconds": "MachineTimeStepInSeconds",
        "minimum_buffer_sdram": "MinBufferSize"})
    @overrides(
        AbstractNengoApplicationVertex.create_machine_vertices,
        additional_arguments=[
            "operator_graph", "machine_time_step", "n_machine_time_steps",
            "machine_time_step_in_seconds", "minimum_buffer_sdram"])
    def create_machine_vertices(
            self, resource_tracker, machine_graph, graph_mapper,
            operator_graph, machine_time_step, n_machine_time_steps,
            machine_time_step_in_seconds, minimum_buffer_sdram):

        machine_vertices = list()

        # build a bunch of maps and data objects used during partitioning
        self._create_internal_data_maps(operator_graph, machine_time_step)

        # start the partitioning process, now that all the data required to
        # do so has been deduced
        n_atoms_partitioned = 0
        while n_atoms_partitioned < self._n_neurons:
            max_cores = resource_tracker.get_maximum_cores_available_on_a_chip()

            # if supporting ensembles over multiple chips, do cluster
            # partitioning. else assume one chip and partition accordingly.
            cluster_vertices = list()
            if self.ENSEMBLE_PARTITIONING_OVER_MULTIPLE_CHIPS:
                slices = NengoPartitioner.create_slices(
                    Slice(0, self._n_neurons - 1), self,
                    self._max_resources_to_use_per_core, max_cores,
                    self._n_neurons, resource_tracker)
                for neuron_slice in slices:
                    self._n_neurons_in_current_cluster = (
                        neuron_slice.hi_atom - neuron_slice.lo_atom)
                    cluster_vertices = self._create_cluster_and_verts(
                        neuron_slice, max_cores, resource_tracker,
                        n_machine_time_steps, machine_time_step_in_seconds,
                        minimum_buffer_sdram)
                    machine_vertices.extend(cluster_vertices)
            else:
                self._n_neurons_in_current_cluster = self._n_neurons
                cluster_vertices = self._create_cluster_and_verts(
                    Slice(0, self._n_neurons - 1), max_cores, resource_tracker,
                    n_machine_time_steps, machine_time_step_in_seconds,
                    minimum_buffer_sdram)
                machine_vertices.extend(cluster_vertices)

            # update the atom tracker
            n_atoms_partitioned += self._n_neurons_in_current_cluster

            # update the resource tracker to take out a valid chip from the
            # available set.
            group_resource = list()
            for vertex in cluster_vertices:
                group_resource.append(
                    (vertex.resources_required, vertex.constraints))
                machine_graph.add_vertex(vertex)
                graph_mapper.add_vertex_mapping(
                    machine_vertex=vertex, application_vertex=self)

            # allocate resources to resource tracker, to ensure next vertex
            # doesnt partition on incorrect data
            resource_tracker.allocate_constrained_group_resources(
                group_resource)

            # add sdram edges as required
            self._add_sdram_outgoing_partitions_and_edges(
                machine_graph, cluster_vertices)

    def _add_sdram_outgoing_partitions_and_edges(
            self, machine_graph, cluster_vertices):

        # get one vertex to be the source, dont matter which, as chip wide
        pre_vertex_for_sdram_edges = cluster_vertices[0]

        # handle input vector
        sdram_outgoing_partition = SegmentedInputSDRAMMachinePartition(
            identifier=self.SDRAM_OUTGOING_INPUT,
            label=self.SDRAM_OUTGOING_INPUT,
            pre_vertex=pre_vertex_for_sdram_edges)
        machine_graph.add_outgoing_edge_partition(sdram_outgoing_partition)
        for post_vertex in cluster_vertices:
            edge = SDRAMMachineEdge(
                pre_vertex=pre_vertex_for_sdram_edges,
                post_vertex=post_vertex,
                sdram_size=(
                    post_vertex.input_slice.n_atoms *
                    constants.BYTE_TO_WORD_MULTIPLIER),
                label="SDRAM_EDGE_FOR_INPUT_VECTOR_FOR_BITS{}:{}".format(
                    post_vertex.input_slice.lo_atom,
                    post_vertex.input_slice.hi_atom))
            machine_graph.add_edge(
                edge, self.SDRAM_OUTGOING_INPUT)

        # handle learnt encoders shared sdram
        for learnt_encoder_filter in self._learnt_encoder_filters:
            identifier = (self.SDRAM_OUTGOING_LEARNT, learnt_encoder_filter)
            sdram_outgoing_partition = SegmentedInputSDRAMMachinePartition(
                identifier=identifier,
                label=(self.SDRAM_OUTGOING_LEARNT + "{}".format(
                    learnt_encoder_filter)),
                pre_vertex=pre_vertex_for_sdram_edges)
            machine_graph.add_outgoing_edge_partition(sdram_outgoing_partition)

            for post_vertex in cluster_vertices:
                edge = SDRAMMachineEdge(
                    pre_vertex=pre_vertex_for_sdram_edges,
                    post_vertex=post_vertex,
                    sdram_size=(post_vertex.input_slice.n_atoms *
                                constants.BYTE_TO_WORD_MULTIPLIER),
                    label="SDRAM_EDGE_FOR_INPUT_VECTOR_FOR_BITS{}:{}".format(
                        post_vertex.input_slice.lo_atom,
                        post_vertex.input_slice.hi_atom))
                machine_graph.add_edge(edge, identifier)

        # handle spike vector
        sdram_outgoing_partition = SegmentedSpikesSDRAMMachinePartition(
            identifier=self.SDRAM_OUTGOING_SPIKE_VECTOR,
            label=self.SDRAM_OUTGOING_SPIKE_VECTOR,
            pre_vertex=pre_vertex_for_sdram_edges)
        machine_graph.add_outgoing_edge_partition(sdram_outgoing_partition)
        for post_vertex in cluster_vertices:
            edge = SDRAMMachineEdge(
                pre_vertex=pre_vertex_for_sdram_edges,
                post_vertex=post_vertex,
                sdram_size=(
                    math.ceil(post_vertex.neuron_slice.n_atoms /
                              constants.WORD_TO_BIT_CONVERSION) *
                    constants.BYTE_TO_WORD_MULTIPLIER),
                label="SDRAM_EDGE_FOR_SPIKE_VECTOR_FOR_BITS{}:{}".format(
                    post_vertex.input_slice.lo_atom,
                    post_vertex.input_slice.hi_atom))
            machine_graph.add_edge(edge, self.SDRAM_OUTGOING_SPIKE_VECTOR)

        # handle semaphore sdram
        sdram_outgoing_partition = ConstantSDRAMMachinePartition(
            identifier=self.SDRAM_OUTGOING_SEMAPHORE,
            label=self.SDRAM_OUTGOING_SEMAPHORE,
            pre_vertex=pre_vertex_for_sdram_edges)
        machine_graph.add_outgoing_edge_partition(sdram_outgoing_partition)
        for post_vertex in cluster_vertices:
            edge = SDRAMMachineEdge(
                pre_vertex=pre_vertex_for_sdram_edges,
                post_vertex=post_vertex,
                sdram_size=(
                    LIFMachineVertex.SHARED_SDRAM_FOR_SEMAPHORES_IN_BYTES),
                label="SDRAM_EDGE_FOR_SEMAPHORE")
            machine_graph.add_edge(edge, self.SDRAM_OUTGOING_SEMAPHORE)

    def _create_cluster_and_verts(
            self, chip_neuron_slice, max_cores, resource_tracker,
            n_machine_time_steps, machine_time_step_in_seconds,
            minimum_buffer_sdram):

        cluster_vertices = list()

        # Partition the slice of neurons that we have
        sliced_objects = [
            Slice(0, int(self._encoders_with_gain.shape[1])),  # Input subspace
            chip_neuron_slice,  # Neurons
            Slice(0, int(self._decoders.shape[0])),  # Outputs
            Slice(0, len(self._learnt_encoder_filters))  # Learnt output
        ]
        # create core sized partitions
        all_slices_and_resources = NengoPartitioner.create_slices_for_multiple(
            sliced_objects, self, self._max_resources_to_use_per_core,
            max_cores, max_cores, resource_tracker)

        # TODO this is assuming the core_neuron_slice's are incremental. This
        # TODO is valid so far.
        for slices, _ in all_slices_and_resources:
            self._machine_vertex_slices[chip_neuron_slice].append(
                slices[self.SLICES_POSITIONS.NEURON.value])
            self._core_slice_to_chip_slice[
                slices[self.SLICES_POSITIONS.NEURON.value]] = chip_neuron_slice

        for vertex_index, (slices, used_resources) in enumerate(
                all_slices_and_resources):

            # recording items
            buffered_sdram = self.get_buffered_sdram(
                slices[self.SLICES_POSITIONS.NEURON.value],
                n_machine_time_steps)
            this_verts_minimum_buffer_sdram = \
                recording_utilities.get_minimum_buffer_sdram(
                buffered_sdram, minimum_buffer_sdram)
            buffered_sdram_per_timestep = self._get_buffered_sdram_per_timestep(
                slices[self.SLICES_POSITIONS.NEURON.value])
            overflow_sdram = self._neuron_recorder.get_sampling_overflow_sdram(
                slices[self.SLICES_POSITIONS.NEURON.value])

            # Tile direct input across all encoder copies (used for learning)
            tiled_direct_input = numpy.tile(
                self.direct_input,
                self._encoders_with_gain.shape[1] // self._ensemble_size_in)

            # Combine the direct input with the bias
            bias_with_di = self._bias + numpy.dot(
                self._encoders_with_gain, tiled_direct_input)

            # build the new machine vertex
            vertex = LIFMachineVertex(
                sub_population_id=vertex_index,
                neuron_slice=slices[self.SLICES_POSITIONS.NEURON.value],
                input_slice=slices[self.SLICES_POSITIONS.INPUT.value],
                output_slice=slices[self.SLICES_POSITIONS.OUTPUT.value],
                learnt_slice=slices[self.SLICES_POSITIONS.LEARNT_OUTPUT.value],
                resources=used_resources,
                ensemble_size_in=self._ensemble_size_in,
                encoders_with_gain_shape=self._encoders_with_gain.shape[1],
                learnt_encoder_filters=self._learnt_encoder_filters,
                tau_rc=self._tau_rc, tau_refactory=self._tau_refactory,
                input_filters=self._input_filters,
                inhibitory_filters=self._inhibition_filters,
                modulatory_filters=self._modulatory_filters,
                pes_learning_rules=(
                    self._get_sliced_learning_rules(
                        slices[self.SLICES_POSITIONS.LEARNT_OUTPUT.value])),
                ensemble_radius=self._radius,
                label=(
                    "LIF_machine_vertex_covering_slices for lif "
                    "app vertex {}.".format(self.label)),
                overflow_sdram=overflow_sdram,
                buffered_sdram_per_timestep=buffered_sdram_per_timestep,
                minimum_buffer_sdram_usage=this_verts_minimum_buffer_sdram,
                is_recording=(
                    len(self._neuron_recorder.recording_variables) > 0),
                encoders_with_gain=(
                    helpful_functions.convert_matrix_to_machine_vertex_level(
                        self._encoders_with_gain,
                        slices[self.SLICES_POSITIONS.NEURON.value].as_slice,
                        constants.MATRIX_CONVERSION_PARTITIONING.ROWS)),
                bias_with_di=(
                    helpful_functions.convert_matrix_to_machine_vertex_level(
                        bias_with_di,
                        slices[self.SLICES_POSITIONS.NEURON.value].as_slice,
                        constants.MATRIX_CONVERSION_PARTITIONING.ROWS)),
                gain=(helpful_functions.convert_matrix_to_machine_vertex_level(
                    self._gain,
                    slices[self.SLICES_POSITIONS.NEURON.value].as_slice,
                    constants.MATRIX_CONVERSION_PARTITIONING.ROWS)),
                decoders=(
                    helpful_functions.convert_matrix_to_machine_vertex_level(
                        self._decoders / machine_time_step_in_seconds,
                        slices[self.SLICES_POSITIONS.NEURON.value].as_slice,
                        constants.MATRIX_CONVERSION_PARTITIONING.ROWS)),
                learnt_decoders=(
                    helpful_functions.convert_matrix_to_machine_vertex_level(
                        self._learnt_decoders / machine_time_step_in_seconds,
                        slices[self.SLICES_POSITIONS.NEURON.value].as_slice,
                        constants.MATRIX_CONVERSION_PARTITIONING.ROWS)))
            cluster_vertices.append(vertex)
        return cluster_vertices

    def _get_buffered_sdram_per_timestep(self, vertex_slice):
        values = list()
        for recordable_variable in self._is_recording_probeable_variable:
            if self._is_recording_probeable_variable[recordable_variable]:
                values.append(
                    self._neuron_recorder.get_buffered_sdram_per_timestep(
                        recordable_variable, vertex_slice))
        return values

    def get_buffered_sdram(self, vertex_slice, n_machine_time_steps):
        values = list()
        for recordable_variable in self._is_recording_probeable_variable:
            if self._is_recording_probeable_variable[recordable_variable]:
                values.append(
                    self._neuron_recorder.get_buffered_sdram(
                        recordable_variable, vertex_slice,
                        n_machine_time_steps))
        return values

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

    @inject_items(
        {"n_machine_time_steps": "TotalMachineTimeSteps",
         "spike_buffer_max_size": "SpikeBufferMaxSize",
         "variable_buffer_max_size": "VariableBufferMaxSize",
         "minimum_buffer_sdram": "MinBufferSize",
         "using_auto_pause_and_resume": "UsingAutoPauseAndResume",
         "receive_buffer_host": "ReceiveBufferHost",
         "receive_buffer_port": "ReceiveBufferPort"})
    @overrides(
        AbstractSupportNengoPartitioner.get_resources_for_slices,
        additional_arguments=[
            "n_machine_time_steps", "spike_buffer_max_size",
            "variable_buffer_max_size", "minimum_buffer_sdram",
            "using_auto_pause_and_resume", "receive_buffer_host",
            "receive_buffer_port"])
    def get_resources_for_slices(
            self, slices, n_cores, n_machine_time_steps,
            spike_buffer_max_size, variable_buffer_max_size,
            minimum_buffer_sdram, using_auto_pause_and_resume,
            receive_buffer_host, receive_buffer_port):
        resources = ResourceContainer(
            cpu_cycles=self._cpu_usage_for_slices(slices, n_cores),
            dtcm=self._dtcm_usage_for_slices(slices, n_cores),
            sdram=self._sdram_usage_for_slices(slices, n_cores))

        # set up max sdram for buffering if not done already
        if self._maximum_sdram_for_buffering is None:
            self._maximum_sdram_for_buffering = list()
            for variable in self._is_recording_probeable_variable:
                if variable == constants.RECORD_SPIKES_FLAG:
                    self._maximum_sdram_for_buffering.append(
                        spike_buffer_max_size)
                else:
                    self._maximum_sdram_for_buffering.append(
                        variable_buffer_max_size)

        recording_sizes = recording_utilities.get_recording_region_sizes(
            self.get_buffered_sdram(
                slices[self.SLICES_POSITIONS.NEURON.value],
                n_machine_time_steps),
            minimum_buffer_sdram, self._maximum_sdram_for_buffering,
            using_auto_pause_and_resume)
        resources.extend(recording_utilities.get_recording_resources(
            recording_sizes, receive_buffer_host, receive_buffer_port))
        return resources

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
            LIFMachineVertex.PES_REGION_N_ELEMENTS + len(
                self._get_sliced_learning_rules(learnt_output_slice)) +
            LIFMachineVertex.PES_REGION_SLICED_RULE_N_ELEMENTS)

        # constant based regions
        ensemble_region = (
            (LIFMachineVertex.ENSEMBLE_PARAMS_ITEMS +
             (len(self._learnt_encoder_filters) *
              LIFMachineVertex.SDRAM_ITEMS_PER_LEARNT_INPUT_VECTOR) +
             n_cores) *
            constants.BYTE_TO_WORD_MULTIPLIER)

        # lif region
        lif_region = (
            LIFMachineVertex.NEURON_PARAMS_ITEMS *
            constants.BYTE_TO_WORD_MULTIPLIER)

        # voja region
        voja_region = (
            (LIFMachineVertex.VOJA_REGION_N_ELEMENTS +
             (len(self._voja_learning_rules) *
              LIFMachineVertex.VOJA_REGION_RULE_N_ELEMENT)
             ) * constants.BYTE_TO_WORD_MULTIPLIER)

        # matrix based regions
        decoders_region = \
            helpful_functions.convert_matrix_to_machine_vertex_level(
                self._decoders, output_slice.as_slice,
                constants.MATRIX_CONVERSION_PARTITIONING.ROWS).nbytes

        learnt_decoders_region = \
            helpful_functions.convert_matrix_to_machine_vertex_level(
                self._learnt_decoders, learnt_output_slice.as_slice,
                constants.MATRIX_CONVERSION_PARTITIONING.ROWS).nbytes

        encoders_region = \
            helpful_functions.convert_matrix_to_machine_vertex_level(
                self._encoders_with_gain, neuron_slice.as_slice,
                constants.MATRIX_CONVERSION_PARTITIONING.ROWS).nbytes

        bias_region = \
            helpful_functions.convert_matrix_to_machine_vertex_level(
                self._bias, neuron_slice.as_slice,
                constants.MATRIX_CONVERSION_PARTITIONING.ROWS).nbytes

        gain_region = \
            helpful_functions.convert_matrix_to_machine_vertex_level(
                self._gain, neuron_slice.as_slice,
                constants.MATRIX_CONVERSION_PARTITIONING.ROWS).nbytes

        # basic key regions
        key_region = (
            constants.BYTES_PER_KEY * self._n_output_keys *
            output_slice.n_atoms)

        learnt_key_region = (
            constants.BYTES_PER_KEY * self._n_learnt_output_keys *
            learnt_output_slice.n_atoms)

        # partitioning data for the machine vertices
        population_length_region = constants.BYTE_TO_WORD_MULTIPLIER * n_cores

        # filter regions
        input_filter_region = helpful_functions.\
            sdram_size_in_bytes_for_filter_region(self._input_filters)
        inhib_filter_region = helpful_functions.\
            sdram_size_in_bytes_for_filter_region(self._inhibition_filters)
        modulatory_filters_region = helpful_functions.\
            sdram_size_in_bytes_for_filter_region(self._modulatory_filters)
        learnt_encoder_filters_region = helpful_functions.\
            sdram_size_in_bytes_for_filter_region(self._learnt_encoder_filters)

        # routing regions
        input_routing_region = helpful_functions.\
            sdram_size_in_bytes_for_routing_region(self._inputs_n_keys)
        inhib_routing_region = helpful_functions.\
            sdram_size_in_bytes_for_routing_region(self._inhibition_n_keys)
        modulatory_routing_region = helpful_functions.\
            sdram_size_in_bytes_for_routing_region(self._modulatory_n_keys)
        learnt_encoder_routing_region = helpful_functions.\
            sdram_size_in_bytes_for_routing_region(self._learnt_encoders_n_keys)

        # recording index region
        recording_index_region = (
            (LIFMachineVertex.N_RECORDING_VARIABLE_SIZE +
             len(self._probeable_variables)) *
            constants.BYTE_TO_WORD_MULTIPLIER)

        # provenance data region
        provenance_data = LIFMachineVertex.get_provenance_data_size(
            LIFMachineVertex.EXTRA_PROVENANCE_DATA_ENTRIES
            .N_LOCAL_PROVENANCE_ITEMS.value)

        total = (
            ensemble_region + lif_region + pes_region + voja_region +
            decoders_region + learnt_decoders_region + encoders_region +
            bias_region + gain_region + key_region + learnt_key_region +
            population_length_region + input_filter_region +
            inhib_filter_region + modulatory_filters_region +
            learnt_encoder_filters_region + input_routing_region +
            inhib_routing_region + modulatory_routing_region +
            learnt_encoder_routing_region + recording_index_region +
            provenance_data)

        if len(slices) == 1:
            return SDRAMResource(int(math.ceil(total / n_cores)))
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
            incoming_modulatory_learning_rules, machine_time_step,
            operator_graph):
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
                decoder_stop=decoder_stop,
                error_filter_index=len(self._modulatory_filters))
            self._pes_learning_rules.append(pes_learning_rule)

            # Add error connection to lists of modulatory filters and routes
            modulatory_edge = incoming_modulatory_learning_rules[
                transmission_parameter.learning_rule]
            FilterAndRoutingRegionGenerator.add_filters(
                self._modulatory_filters, modulatory_edge,
                operator_graph.get_outgoing_partition_for_edge(modulatory_edge),
                minimise=False)
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

            learn_sig_filter_index = None
            decoded_input_filter_index = None
            if isinstance(learning_rule.learning_rule_type, NengoVoja):

                learn_sig_filter_index = -1
                # If there is a modulatory connection
                # associated with the learning rule
                if learning_rule in incoming_modulatory_learning_rules.keys():
                    learning_edge = incoming_modulatory_learning_rules[
                        learning_rule]

                    learn_sig_filter_index = len(self._modulatory_filters)

                    # Add learning connection to lists
                    # of modulatory filters and route
                    FilterAndRoutingRegionGenerator.add_filters(
                        self._modulatory_filters, learning_edge,
                        operator_graph.get_outgoing_partition_for_edge(
                            learning_edge),
                        minimise=False)
                    self._modulatory_n_keys += 1

                decoded_input_filter_index = len(self._learnt_encoder_filters)

                # Add learnt connection to list of filters
                # and routes with learnt encoders
                FilterAndRoutingRegionGenerator.add_filters(
                    self._learnt_encoder_filters, edge,
                    operator_graph.get_outgoing_partition_for_edge(edge),
                    minimise=False)
                self._learnt_encoders_n_keys += 1

            voja_learning_rule = VojaLearningRule(
                learning_rule.learning_rule_type.learning_rate,
                encoder_offset=self._encoders_with_gain.shape[1],
                learning_signal_filter_index=learn_sig_filter_index,
                decoded_input_filter_index=decoded_input_filter_index)
            self._voja_learning_rules.append(voja_learning_rule)

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
