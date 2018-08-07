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
    ResourceContainer
from spinn_machine import Processor
from spinn_utilities.overrides import overrides
from nengo_spinnaker_gfe import constants
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
        "_ensamble_size_in",
        "_n_profiler_samples",
        "_n_neurons_in_current_cluster",
        "_encoders_with_gain",
        "_max_resources_to_use_per_core"]

    ENSEMBLE_PROFILER_TAGS = Enum(
        value="PROFILER_TAGS",
        names=[("INPUT_FILTER", 0),
               ("NEURON_UPDATE", 1),
               ("DECODE_AND_TRANSMIT_OUTPUT", 2),
               ("PES", 3),
               ("VOJA", 4)])

    # flag saying if the ensemble can operate over multiple chips
    ENSAMBLE_PARTITIONING_OVER_MULTIPLE_CHIPS = False

    # expected resource limits to allow collaboration cores to work
    DTCM_USAGE_PER_CORE = 0.75
    CPU_USAGE_PER_CORE = 0.4

    # magic numbers from mundy's thesis, no idea what they are, or where they
    #  come from.
    INPUT_FILTERING_CYCLES_1 = 39
    INPUT_FILTERING_CYCLES_2 = 135
    NEURON_UPDATE_CYCLES_1 = 9
    NEURON_UPDATE_CYCLES_2 = 61
    NEURON_UPDATE_CYCLES_3 = 174
    DECODE_AND_TRANSMIT_CYCLES_1 = 2
    DECODE_AND_TRANSMIT_CYCLES_2 = 143
    DECODE_AND_TRANSMIT_CYCLES_3 = 173

    DTCM_BYTES_PER_NEURON = 3
    DTCM_BYTES_MULTIPLIER = 4

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
        self._ensamble_size_in = size_in
        self._n_neurons = n_neurons
        self._cluster_size_out = None
        self._cluster_size_in = None
        self._cluster_learnt_size_out = None
        self._n_profiler_samples = n_profiler_samples
        self._n_neurons_in_current_cluster = None
        self._max_resources_to_use_per_core = ResourceContainer(
            dtcm=DTCMResource(
                int(math.ceil(
                    Processor.DTCM_AVAILABLE * self.DTCM_USAGE_PER_CORE))),
            cpu_cycles=CPUCyclesPerTickResource(
                int(math.ceil(
                    Processor.CLOCK_SPEED * self.CPU_USAGE_PER_CORE))))

        self._probeable_variables = [
            constants.RECORD_OUTPUT_FLAG, constants.RECORD_SPIKES_FLAG,
            constants.RECORD_VOLTAGE_FLAG, constants.SCALED_ENCODERS_FLAG]

        self._is_recording_probeable_variable = dict()
        for flag in self._probeable_variables:
            self._is_recording_probeable_variable[flag] = False

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
                        in_edge.reception_parameters.learning_rule] = \
                            operator_graph.get_outgoing_partition_for_edge(
                                in_edge)
                else:
                    incoming_modulatory_learning_rules[
                        in_edge.input_port.learning_rule] = \
                             operator_graph.get_outgoing_partition_for_edge(
                                 in_edge)

            if (in_edge.input_port.destination_input_port ==
                    constants.ENSEMBLE_INPUT_PORT.LEARNT):
                incoming_learnt_edges.append(
                    (in_edge, in_edge.reception_parameters.learning_rule))

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
        decoders = numpy.array([])
        if len(standard_outgoing_partitions) != 0:
            decoders, _ = self._get_decoders_and_n_keys(
                standard_outgoing_partitions, True)

        # convert to cluster sizes
        self._cluster_size_out = decoders.shape[0]
        self._cluster_size_in = self._scaled_encoders.shape[1]
        (self._cluster_learnt_size_out, pes_learning_rules, mod_filters) = \
            self._determine_cluster_learnt_size_out(
                outgoing_learnt_partitions, incoming_modulatory_learning_rules,
                machine_time_step)

        voja_learning_rules, learnt_encoder_filters = \
            self._determine_voja_learning_rules(
                incoming_learnt_edges, incoming_modulatory_learning_rules,
                mod_filters, operator_graph)

        n_atoms_partitioned = 0

        while n_atoms_partitioned < self._n_neurons:
            max_cores = resource_tracker.get_maximum_cores_available_on_a_chip()
            if self.ENSAMBLE_PARTITIONING_OVER_MULTIPLE_CHIPS:
                slices = nengo_partitioner.create_slices(
                    Slice(0, self._n_neurons - 1), self,
                    self._max_resources_to_use_per_core, max_cores,
                    self._n_neurons)
                for neuron_slice in slices:
                    self._n_neurons_in_current_cluster = (
                        neuron_slice.hi_atom - neuron_slice.lo_atom)
                    cluster_vertices = self._create_cluster_and_verts(
                        neuron_slice, decoders.shape[0],
                        len(learnt_encoder_filters), max_cores,
                        nengo_partitioner)
                    machine_vertices.extend(cluster_vertices)
                    n_atoms_partitioned += self._n_neurons_in_current_cluster
            else:
                self._n_neurons_in_current_cluster = self._n_neurons
                cluster_vertices = self._create_cluster_and_verts(
                    Slice(0, self._n_neurons - 1), decoders.shape[0],
                    len(learnt_encoder_filters), max_cores, nengo_partitioner)
                machine_vertices.extend(cluster_vertices)
                n_atoms_partitioned += self._n_neurons
        return machine_vertices

    def _create_cluster_and_verts(
            self, neuron_slice, size_out, n_learnt_input_signals, max_cores,
            nengo_partitioner):

        cluster_vertices = list()
        # Partition the slice of neurons that we have
        sliced_objects = [
            Slice(0, int(self._encoders_with_gain.shape[1])),  # Input subspace
            neuron_slice,  # Neurons
            Slice(0, int(size_out)),  # Outputs
            Slice(0, int(n_learnt_input_signals)),  # Learnt output
        ]

        # create core sized partitions
        all_slices = list(
            nengo_partitioner._create_slices_for_multiple(
                sliced_objects, self, self._max_resources_to_use_per_core,
                max_cores, max_cores))

        neuron_slices = list()
        for _, neuron_slice, _, _ in all_slices:
            neuron_slices.append(neuron_slice)

        for vertex_index, (
                input_slice, _, output_slice, learnt_slice) in enumerate(
                    all_slices):
            vertex = LIFMachineVertex(
                vertex_index, neuron_slices, input_slice, output_slice,
                learnt_slice)
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

    @overrides(AbstractSupportNengoPartitioner.dtcm_usage_for_slices)
    def dtcm_usage_for_slices(self, slices, n_cores):

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
                self.DTCM_BYTES_MULTIPLIER)
        else:
            """Get the amount of memory required."""
            # filtered_dims = input_slice.stop - input_slice.start
            n_neurons = slices[1].hi_atom - slices[1].lo_atom
            size_out = slices[2].hi_atom - slices[2].lo_atom
            size_learnt_out = slices[3].hi_atom - slices[3].lo_atom

            encoder_cost = n_neurons * self._encoders_with_gain.shape[1]
            decoder_cost = (
                self._n_neurons_in_current_cluster * (
                    size_out + size_learnt_out))
            neurons_cost = n_neurons * 3

            return (encoder_cost + decoder_cost + neurons_cost) * 4

    @overrides(AbstractSupportNengoPartitioner.cpu_usage_for_slices)
    def cpu_usage_for_slices(self, slices, n_cores):
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
        else: # handling core size chunks
            filtered_dims = slices[0].hi_atom - slices[0].lo_atom
            n_neurons = slices[1].hi_atom - slices[1].lo_atom
            size_out = slices[2].hi_atom - slices[2].lo_atom
            size_learnt_out = slices[3].hi_atom - slices[3].lo_atom

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
        learnt_decoders = numpy.array([])
        pes_learning_rules = list()
        mod_filters = defaultdict(list)
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

            decoder_start = learnt_decoders.shape[0]

            pes_learning_rule = PESLearningRule(
                learning_rate=(
                    transmission_parameter.learning_rule.learning_rule_type.
                    learning_rate / machine_time_step),
                decoder_start=decoder_start,
                decoder_stop=learnt_decoders.shape[0])
            pes_learning_rules.append(pes_learning_rule)

            # Add error connection to lists
            # of modulatory filters and routes
            FilterAndRoutingRegionGenerator.add_filters(
                mod_filters, learnt_outgoing_partition, pes_learning_rule,
                minimise=False)

            # Create a duplicate copy of the original size_in columns of
            # the encoder matrix for modification by this learning rule
            base_encoders = self._encoders_with_gain[:, :self._ensamble_size_in]
            self._encoders_with_gain = numpy.hstack(
                (self._encoders_with_gain, base_encoders))
        return learnt_decoders.shape[0], pes_learning_rules, mod_filters

    def _determine_voja_learning_rules(
            self, incoming_learnt_edges_and_learning_rules,
            incoming_modulatory_learning_rules, mod_filters, operator_graph):

        voja_learning_rules = list()
        learnt_encoder_filters = defaultdict(list)
        for (edge, learning_rule) in incoming_learnt_edges_and_learning_rules:

            if isinstance(learning_rule.learning_rule_type, NengoVoja):
                voja_learning_rule = VojaLearningRule(
                    learning_rule.learning_rule_type.learning_rate,
                    encoder_offset=self._encoders_with_gain.shape[1])
                voja_learning_rules.append(voja_learning_rule)

                # If there is a modulatory connection
                # associated with the learning rule
                if learning_rule in incoming_modulatory_learning_rules.keys():
                    outgoing_partition = incoming_modulatory_learning_rules[
                        learning_rule]

                    # Add learning connection to lists
                    # of modulatory filters and routes
                    FilterAndRoutingRegionGenerator.add_filters(
                        mod_filters, outgoing_partition, voja_learning_rule,
                        minimise=False)

                # Add learnt connection to list of filters
                # and routes with learnt encoders
                FilterAndRoutingRegionGenerator.add_filters(
                    learnt_encoder_filters,
                    operator_graph.get_outgoing_partition_for_edge(edge),
                    voja_learning_rule, minimise=False)

        return voja_learning_rules, learnt_encoder_filters

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
