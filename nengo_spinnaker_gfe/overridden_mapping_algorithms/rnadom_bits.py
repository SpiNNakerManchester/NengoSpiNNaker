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