import math

from nengo_spinnaker_gfe.abstracts.abstract_supports_nengo_partitioner import \
    AbstractSupportNengoPartitioner
from pacman.model.graphs.common import Slice
from pacman.model.graphs.machine import MachineGraph, MachineEdge
from pacman.utilities.utility_objs import ResourceTracker
from nengo_spinnaker_gfe import constants
from nengo_spinnaker_gfe.abstracts. \
    abstract_accepts_multicast_signals import AcceptsMulticastSignals
from nengo_spinnaker_gfe.graph_components. \
    graph_mapper import GraphMapper
from nengo_spinnaker_gfe.nengo_exceptions import \
    NotAbleToBeConnectedToAsADestination, NotPartitionable

from nengo_spinnaker_gfe.machine_vertices.interposer_machine_vertex import \
    InterposerMachineVertex
from spinn_machine import Processor
from spinn_utilities.progress_bar import ProgressBar


class NengoPartitioner(object):
    """ partitions the app graph for the nengo_spinnaker_gfe graph, and turns /
    it into a machine graph recognised by the main tool chain
    
    """

    def __call__(self, nengo_operator_graph, machine,
                 pre_allocated_resources=None):
        machine_graph = MachineGraph(label=constants.MACHINE_GRAPH_LABEL)
        graph_mapper = GraphMapper()

        resource_tracker = ResourceTracker(
            machine, preallocated_resources=pre_allocated_resources)

        progress_bar = ProgressBar(
            total_number_of_things_to_do=(
                len(nengo_operator_graph.vertices) +
                len(nengo_operator_graph.edges)),
            string_describing_what_being_progressed="partitioning")

        # convert application vertices into machine vertices
        for operator in progress_bar.over(
                nengo_operator_graph.vertices, False):

            # create the machine verts
            machine_vertices = operator.create_machine_vertices(
                resource_tracker, self)

            # update data objects
            for machine_vertex in machine_vertices:
                machine_graph.add_vertex(machine_vertex)
                graph_mapper.add_vertex_mapping(
                    machine_vertex=machine_vertex, application_vertex=operator)

        # Construct edges from the application edges
        for edge in progress_bar.over(nengo_operator_graph.edges):
            for machine_vertex_source in graph_mapper.get_machine_vertices(
                    edge.pre_vertex):
                if (isinstance(
                        machine_vertex_source, InterposerMachineVertex) and
                        machine_vertex_source.transmits_signal(
                            edge.transmission_parameters)):
                    self._check_destination_vertices(
                        edge, machine_vertex_source, graph_mapper,
                        machine_graph)
        return machine_graph, graph_mapper

    @staticmethod
    def _check_destination_vertices(
            edge, machine_vertex_source, graph_mapper, machine_graph):
        for machine_vertex_sink in \
                graph_mapper.get_machine_vertices(edge.post_vertex):
            if (isinstance(
                    machine_vertex_sink, AcceptsMulticastSignals) and
                    machine_vertex_sink.accepts_multicast_signals(
                        edge.transmission_parameters)):
                machine_edge = MachineEdge(
                    pre_vertex=machine_vertex_source,
                    post_vertex=machine_vertex_sink)
                machine_graph.add_edge(machine_edge)
                graph_mapper.add_edge_mapping(
                    machine_edge=machine_edge, application_edge=edge)
            elif not isinstance(machine_vertex_sink, AcceptsMulticastSignals):
                raise NotAbleToBeConnectedToAsADestination(
                    "The vertex {} is not meant to receive connections. But "
                    "it received a connection from {}".format(
                        machine_vertex_sink, edge))

    def create_slices(self, initial_slice, application_vertex,
                      max_resources_to_use_per_core, max_cores, max_cuts):
        """
        
        :param initial_slice: 
        :param application_vertex: 
        :param max_resources_to_use_per_core: 
        :param max_cores: 
        :return: 
        """

        for sl, in self._create_slices_for_multiple(
                [initial_slice], application_vertex,
                max_resources_to_use_per_core, max_cores, max_cuts):
            yield sl

    def _create_slices_for_multiple(
            self, initial_slices, application_vertex,
            max_resources_to_use_per_core, max_cores, user_max_cuts):
        """
        
        :param initial_slices: 
        :param application_vertex: 
        :param max_resources_to_use_per_core: 
        :param max_cores: 
        :return: 
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

        while any(self._constraints_unsatisfied(
                slices, application_vertex, max_resources_to_use_per_core,
                max_cores)):

            if n_cuts == 1:
                # If we haven't performed any partitioning then we get the first
                # number of cuts to make.
                for internal_slices in slices:
                    dtcm_usage = application_vertex.dtcm_usage_for_slices(
                        internal_slices, max_cores)
                    cpu_usage = application_vertex.cpu_usage_for_slices(
                        internal_slices, max_cores)
                    n_cuts = max(
                        n_cuts,
                        dtcm_usage.get_value() / Processor.DTCM_AVAILABLE,
                        cpu_usage.get_value() / Processor.CLOCK_SPEED)
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
                    for new_neuron_slice in self.divide_slice(
                            neuron_slice, n_cuts):
                        new_slices[-1].append(new_neuron_slice)
            slices = new_slices

        # Yield the partitioned slices
        return zip(*(self.divide_slice(sl, n_cuts) for sl in initial_slices))

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
            slices, application_vertex, max_resources_to_use_per_core, n_cores):
        for internal_slices in slices:
                dtcm_usage = application_vertex.dtcm_usage_for_slices(
                    internal_slices, n_cores)
                cpu_usage = application_vertex.cpu_usage_for_slices(
                    internal_slices, n_cores)

                yield not (
                    dtcm_usage >
                    max_resources_to_use_per_core.dtcm.get_value() or
                    cpu_usage >
                    max_resources_to_use_per_core.cpu_cycles.get_value()
                )
