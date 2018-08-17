from nengo_spinnaker_gfe.abstracts.abstract_supports_nengo_partitioner import \
    AbstractSupportNengoPartitioner
from nengo_spinnaker_gfe.abstracts.abstract_transmits_multicast_signals import \
    AbstractTransmitsMulticastSignals
from nengo_spinnaker_gfe.graph_components.\
    connection_machine_outgoing_partition import \
    ConnectionMachineOutgoingPartition
from pacman.model.graphs.common import Slice
from pacman.model.graphs.machine import MachineGraph, MachineEdge
from pacman.utilities.utility_objs import ResourceTracker
from nengo_spinnaker_gfe import constants
from nengo_spinnaker_gfe.abstracts. \
    abstract_accepts_multicast_signals import AbstractAcceptsMulticastSignals
from nengo_spinnaker_gfe.graph_components. \
    graph_mapper import GraphMapper
from nengo_spinnaker_gfe.nengo_exceptions import NotPartitionable
from spinn_machine import Processor, SDRAM
from spinn_utilities.progress_bar import ProgressBar


class NengoPartitioner(object):
    """ partitions the app graph for the nengo_spinnaker_gfe graph, and turns /
    it into a machine graph recognised by the main tool chain
    
    """

    __slots__ = [
        "_resource_tracker"
    ]

    def __init__(self):
        self._resource_tracker = None

    def __call__(
            self, nengo_operator_graph, machine, nengo_random_number_generator,
            pre_allocated_resources=None):
        machine_graph = MachineGraph(label=constants.MACHINE_GRAPH_LABEL)
        graph_mapper = GraphMapper()

        self._resource_tracker = ResourceTracker(
            machine, preallocated_resources=pre_allocated_resources)

        progress_bar = ProgressBar(
            total_number_of_things_to_do=(
                len(nengo_operator_graph.vertices) +
                len(nengo_operator_graph.outgoing_edge_partitions)),
            string_describing_what_being_progressed="partitioning")

        # convert application vertices into machine vertices
        for operator in progress_bar.over(
                nengo_operator_graph.vertices, False):

            # create the machine verts
            operator.create_machine_vertices(
                self._resource_tracker, machine_graph, graph_mapper)

        self._handle_edges(nengo_operator_graph, machine_graph, graph_mapper,
                           progress_bar, nengo_random_number_generator)

        return machine_graph, graph_mapper

    def _handle_edges(
            self, operator_graph, machine_graph, graph_mapper, progress_bar,
            nengo_random_number_generator):

        for outgoing_edge_partition in progress_bar.over(
                operator_graph.outgoing_edge_partitions, True):
            machine_vertices = graph_mapper.get_machine_vertices(
                outgoing_edge_partition.pre_vertex)
            transmission_parameter = \
                outgoing_edge_partition.identifier.transmission_parameter

            # locate valid sources for machine edges
            valid_sources = list()
            for machine_vertex in machine_vertices:
                if isinstance(machine_vertex, AbstractTransmitsMulticastSignals):
                    if machine_vertex.transmits_multicast_signals(
                            transmission_parameter):
                        valid_sources.append(machine_vertex)
                else:
                    raise NotPartitionable(
                        "The vertex {} can not inform the partitioner if it"
                        " transmits these signals".format(machine_vertex))

            # locate valid destinations for machine edges, and build machine
            # edges
            for app_edge in outgoing_edge_partition.edges:
                destination_app_vertex = app_edge.post_vertex
                machine_vertices = graph_mapper.get_machine_vertices(
                    destination_app_vertex)
                for machine_vertex in machine_vertices:

                    # if accepts this signal, make machine vertex
                    if isinstance(machine_vertex, AbstractAcceptsMulticastSignals):
                        if machine_vertex.accepts_multicast_signals(
                                transmission_parameter):
                            self._create_machine_edge(
                                valid_sources, outgoing_edge_partition,
                                machine_graph, graph_mapper, machine_vertex,
                                app_edge, nengo_random_number_generator)
                    else:
                        raise NotPartitionable(
                            "The vertex {} can not inform the partitioner if it"
                            " accepts these signals".format(machine_vertex))

    @staticmethod
    def _create_machine_edge(
            valid_sources, outgoing_edge_partition, machine_graph, graph_mapper,
            sink, app_edge, nengo_random_number_generator):
            for source in valid_sources:
                machine_outgoing_edge_partition = machine_graph.\
                    get_outgoing_edge_partition_starting_at_vertex(
                        source, outgoing_edge_partition.identifier)
                if machine_outgoing_edge_partition is None:
                    machine_outgoing_edge_partition = \
                        ConnectionMachineOutgoingPartition(
                            identifier=outgoing_edge_partition.identifier,
                            seed=None, pre_vertex=source,
                            rng=nengo_random_number_generator)
                    machine_graph.add_outgoing_edge_partition(
                        machine_outgoing_edge_partition)
                edge = MachineEdge(
                    source, sink, outgoing_edge_partition.identifier.weight)
                machine_graph.add_edge(
                    edge, outgoing_edge_partition.identifier)
                graph_mapper.add_edge_mapping(
                    machine_edge=edge, application_edge=app_edge)

    @staticmethod
    def create_slices(
            initial_slice, application_vertex, max_resources_to_use_per_core,
            max_cores, max_cuts, resource_tracker):
        """
        
        :param initial_slice: 
        :param application_vertex: 
        :param max_resources_to_use_per_core: 
        :param max_cores: 
        :param max_cuts:
        :param resource_tracker:
        :return: 
        """

        return NengoPartitioner.create_slices_for_multiple(
            [initial_slice], application_vertex, max_resources_to_use_per_core,
            max_cores, max_cuts, resource_tracker)

    @staticmethod
    def create_slices_for_multiple(
            initial_slices, application_vertex, max_resources_to_use_per_core,
            max_cores, user_max_cuts, resource_tracker):
        """
        
        :param initial_slices: 
        :param application_vertex: 
        :param max_resources_to_use_per_core: 
        :param max_cores: 
        :param user_max_cuts:
        :param resource_tracker
        :return: dict of tuple to ResourceContainer
        """

        if not isinstance(application_vertex, AbstractSupportNengoPartitioner):
            raise NotPartitionable(
                "The vertex is not of type AbstractSupportNengoPartitioner and"
                " therefore the partitioner cannot determine how to slice the"
                "atoms effectively.")

        # While any of the slices fail to satisfy a constraint we partition
        # further
        n_cuts = 1

        # max atoms = max cuts, as 1 atom per core is the worse you can do
        max_cuts = 0
        for neuron_slice in initial_slices:
            if user_max_cuts is None:
                max_cuts = max(
                    max_cuts, neuron_slice.hi_atom - neuron_slice.lo_atom)
            else:
                max_cuts = user_max_cuts

        slices = [initial_slices]
        resources_used = list()

        while NengoPartitioner._constraints_unsatisfied(
                slices, application_vertex, max_resources_to_use_per_core,
                max_cores, resources_used, resource_tracker):

            # clear resources used, as splitting again
            resources_used = list()

            # If we haven't performed any partitioning then we get the first
            # number of cuts to make.
            if n_cuts == 1:
                for internal_slices in slices:
                    used_resources = \
                        application_vertex.get_resources_for_slices(
                            internal_slices, max_cores)

                    n_cuts = max(
                        n_cuts,
                        used_resources.dtcm.get_value() /
                        Processor.DTCM_AVAILABLE,
                        used_resources.cpu_cycles.get_value() /
                        Processor.CLOCK_SPEED,
                        used_resources.sdram.get_value() /
                        SDRAM.DEFAULT_SDRAM_BYTES)
            else:
                # Otherwise just increment the number of cuts rather than
                # honing in on the expensive elements.
                n_cuts += 1

            if n_cuts > max_cuts:
                # We can't cut any further, so the problem can't be solved.
                raise NotPartitionable(
                    "Even when partitioned to 1 atom per core, the resource "
                    "constraints of the {} vertex cannot be satifisied".format(
                        application_vertex))

            # Partition
            new_slices = list()
            for internal_slices in slices:

                for neuron_slice in internal_slices:
                    new_slices.append(list())
                    for new_neuron_slice in NengoPartitioner.divide_slice(
                            neuron_slice, n_cuts):
                        new_slices[-1].append(new_neuron_slice)
            slices = new_slices

        # map sets of slices to resources used. this allows us to not need to
        # recall the resources again
        results = list()
        for internal_slices, used_resources in zip(slices, resources_used):
            results.append((internal_slices, used_resources))
        return results

    @staticmethod
    def divide_slice(initial_slice, n_slices):
        """
        
        :param initial_slice:  A slice which must have `start` and `stop` set.
        :param n_slices: Number of slices to produce.
        :type initial_slice: :py:class: pacman.model.graphs.common.slice.Slice
        :type n_slices: int
        :return: iterator of slices
        """
        # Extract current position, start and stop
        pos = start = initial_slice.lo_atom
        stop = initial_slice.hi_atom

        # Determine the chunk sizes
        chunk = (stop - start) // n_slices
        n_larger = (stop - start) % n_slices

        # Yield the larger slices
        for _ in range(n_larger):
            yield Slice(pos, pos + chunk)
            pos += chunk

        # Yield the standard sized slices
        for _ in range(n_slices - n_larger):
            yield Slice(pos, pos + chunk)
            pos += chunk

    @staticmethod
    def _constraints_unsatisfied(
            slices, application_vertex, max_resources_to_use_per_core, n_cores,
            resources_used, resource_tracker):

        max_sdram_usage = (
            application_vertex.get_shared_resources_for_slices(
                slices)).sdram.get_value()
        failed = False
        for internal_slices in slices:
            used_resources = application_vertex.get_resources_for_slices(
                internal_slices, n_cores)
            resources_used.append(used_resources)
            max_sdram_usage += used_resources.sdram.get_value()

            failed = (
                used_resources.dtcm.get_value() >
                max_resources_to_use_per_core.dtcm.get_value() or
                used_resources.cpu_cycles.get_value() >
                max_resources_to_use_per_core.cpu_cycles.get_value() or
                used_resources.sdram.get_value() >
                max_resources_to_use_per_core.sdram.get_value())
            if failed:
                break

        # if still valid, check that all sdram allocated including shared
        # will be acceptable
        if not failed:
            max_available_resources = \
                resource_tracker.get_maximum_resources_available()
            failed = max_sdram_usage > max_available_resources.sdram.get_value()
        return failed

