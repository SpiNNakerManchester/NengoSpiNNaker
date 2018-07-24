from pacman.model.graphs.machine import MachineGraph, MachineEdge
from pacman.utilities.utility_objs import ResourceTracker
from nengo_spinnaker_gfe import constants
from nengo_spinnaker_gfe.abstracts. \
    abstract_accepts_multicast_signals import AcceptsMulticastSignals
from nengo_spinnaker_gfe.graph_components. \
    graph_mapper import GraphMapper
from nengo_spinnaker_gfe.nengo_exceptions import \
    NotAbleToBeConnectedToAsADestination

from nengo_spinnaker_gfe.machine_vertices.interposer_machine_vertex import \
    InterposerMachineVertex


class NengoPartitioner(object):
    """ partitions the app graph for the nengo_spinnaker_gfe graph, and turns it into a 
    machine graph recognised by the main tool chain
    
    """

    def __call__(self, nengo_operator_graph, machine,
                 pre_allocated_resources=None):
        machine_graph = MachineGraph(label=constants.MACHINE_GRAPH_LABEL)
        graph_mapper = GraphMapper()

        resource_tracker = ResourceTracker(
            machine, preallocated_resources=pre_allocated_resources)

        # convert application vertices into machine vertices
        for operator in nengo_operator_graph.vertices:

            # create the machine verts
            machine_vertices = operator.create_machine_vertices(
                resource_tracker)

            # update data objects
            for machine_vertex in machine_vertices:
                machine_graph.add_vertex(machine_vertex)
                graph_mapper.add_vertex_mapping(
                    machine_vertex=machine_vertex, application_vertex=operator)

        # Construct edges from the application edges
        for edge in nengo_operator_graph.edges:
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

    @staticmethod
    def partition_set_of_constraints_and_atoms(n_atoms, partition_constraints):
        pass
