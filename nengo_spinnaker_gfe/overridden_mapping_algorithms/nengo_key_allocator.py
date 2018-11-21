from collections import defaultdict, deque

from six import iteritems, iterkeys

from nengo_spinnaker_gfe import constants
from nengo_spinnaker_gfe.abstracts.abstract_nengo_machine_vertex import \
    AbstractNengoMachineVertex
from nengo_spinnaker_gfe.constraints.nengo_key_constraint import \
    NengoKeyConstraint
from nengo_spinnaker_gfe.constraints.nengo_key_constraints import \
    NengoKeyConstraints
from nengo_spinnaker_gfe.overridden_pacman_objects.\
    nengo_base_key_and_masks import NengoBaseKeysAndMasks
from nengo_spinnaker_gfe.utility_objects.rigs_bitfield import BitField
from pacman.model.graphs.common import EdgeTrafficType
from pacman.model.routing_info import RoutingInfo, PartitionRoutingInfo
from pacman.utilities import utility_calls
from spinn_machine import Router


class NengoKeyAllocator(object):

    def __call__(
            self, machine_graph, nengo_operator_graph, graph_mapper,
            routing_by_partition, placements, machine):

        bit_field = BitField(length=constants.KEY_BIT_SIZE)

        # Assign cluster IDs based on the placement and the routing
        vertex_cluster_id = self._assign_cluster_ids(
            nengo_operator_graph, graph_mapper, placements, machine_graph,
            machine, routing_by_partition)

        # set the fields and the fields values in the bit field component
        outgoing_partition_key_spaces = self._allocate_fields_to_keyspaces(
            machine_graph, routing_by_partition, placements, bit_field,
            machine, vertex_cluster_id)

        # Fix all keyspaces
        bit_field.assign_fields()

        # construct the PACMAN routing info object
        routing_info = self._construct_routing_info(
            machine_graph, outgoing_partition_key_spaces)

        # return pacman routing info
        return routing_info

    @staticmethod
    def _construct_routing_info(machine_graph, outgoing_partition_key_spaces):
        """ wrap a nengo bit field key space in a NengoBaseKeysAndMasks object. 
        so that it can get the keys when requested
        
        :param machine_graph: the machine graph
        :param outgoing_partition_key_spaces: 
        :return: 
        """
        routing_infos = RoutingInfo()
        for outgoing_partition in machine_graph.outgoing_edge_partitions:
            if outgoing_partition.traffic_type == EdgeTrafficType.MULTICAST:
                keys_and_masks = list([NengoBaseKeysAndMasks(
                    outgoing_partition_key_spaces[outgoing_partition])])
                routing_infos.add_partition_info(
                    PartitionRoutingInfo(keys_and_masks, outgoing_partition))
        return routing_infos

    def _assign_cluster_ids(
            self, nengo_operator_graph, graph_mapper, placements,
            machine_graph, machine, routing_tables):
        """Assign identifiers to the clusters of vertices owned by each 
        operator to the extent that multicast nets belonging to the same signal 
        which originate at multiple chips can be differentiated if required.

        An operator may be partitioned into a number of vertices which are 
        placed onto the cores of two SpiNNaker chips.  The vertices on these 
        chips form two clusters; packets from these clusters need to be 
        differentiated in order to be routed correctly.  For example:

            +--------+                      +--------+
            |        | ------- (a) ------>  |        |
            |   (A)  |                      |   (B)  |
            |        | <------ (b) -------  |        |
            +--------+                      +--------+

        Packets traversing `(a)` need to be differentiated from packets
        traversing `(b)`.  This can be done by including an additional field 
        in the packet keys which indicates from which chip the packet was 
        sent - in this case a single bit will suffice with packets from `(A)` 
        using a key with the bit not set and packets from `(B)` setting the bit.

        This method will assign an ID to each cluster of vertices (e.g., `(A)` 
        and `(B)`) by storing the index in the `cluster` attribute of each 
        vertex. Later this ID can be used in the keyspace of all nets 
        originating from the cluster.
        
        :param nengo_operator_graph: the machine graph
        :param placements: placements
        """
        # Build a dictionary mapping each operator to the signals and routes for
        # which it is the source.

        # Assign identifiers to each of the clusters of vertices contained
        # within each operator.

        vertex_cluster_id = dict()

        for operator in nengo_operator_graph.vertices:
            chip_placements = set()
            cluster_partitions = list()
            for machine_vertex in graph_mapper.get_machine_vertices(operator):
                placement = placements.get_placement_of_vertex(machine_vertex)
                chip_placements.add((placement.x, placement.y))
                for outgoing_partition in machine_graph.\
                    get_outgoing_edge_partitions_starting_at_vertex(
                        machine_vertex):
                    if (outgoing_partition.traffic_type ==
                            EdgeTrafficType.MULTICAST):
                        cluster_partitions.append(outgoing_partition)

            n_chips = len(chip_placements)
            n_outgoing_partitions = len(cluster_partitions)
            if (n_outgoing_partitions == 0) or (n_chips == 1):

                # If the operator has no outgoing signals, or only one cluster,
                # then assign the same identifier to all of the vertices.
                for machine_vertex in graph_mapper.get_machine_vertices(
                        operator):
                    vertex_cluster_id[machine_vertex] = 0
            else:
                # Otherwise try to allocate as few cluster IDs as are required
                # to differentiate between multicast nets which take
                # different routes at the same router.
                #
                # Build a graph identifying which clusters may or may not share
                # identifiers.

                graph = self._build_cluster_graph(
                    cluster_partitions, placements, machine, routing_tables)

                # Colour this graph to assign identifiers to the clusters
                cluster_ids = self._colour_graph(graph)

                # Assign these colours to the vertices.
                for machine_vertex in graph_mapper.get_machine_vertices(
                        operator):
                    placement = placements.get_placement_of_vertex(
                        machine_vertex)
                    vertex_cluster_id[machine_vertex] = cluster_ids[placement]

        self._verify_all_cluster_ids_consistent(
            vertex_cluster_id, placements, nengo_operator_graph, graph_mapper)
        return vertex_cluster_id

    @staticmethod
    def _verify_all_cluster_ids_consistent(
            vertex_cluster_id, placements, nengo_operator_graph, graph_mapper):
        """ verify all cores on the same chip have the same cluster id
        
        :param vertex_cluster_id: 
        :param placements: 
        :param nengo_operator_graph: 
        :param graph_mapper: 
        :return: 
        """
        for application_vertex in nengo_operator_graph.vertices:
            chip_cluster_id = dict()
            for machine_vertex in graph_mapper.get_machine_vertices(
                    application_vertex):
                placement = placements.get_placement_of_vertex(machine_vertex)
                cluster_id = vertex_cluster_id[machine_vertex]
                if (placement.x, placement.y) not in chip_cluster_id:
                    chip_cluster_id[(placement.x, placement.y)] = cluster_id
                elif chip_cluster_id[(placement.x, placement.y)] != cluster_id:
                    raise Exception("odd")

    def _build_cluster_graph(
            self, outgoing_partitions, placements, machine, routing_tables):
        """Build a graph the nodes of which represent the chips on which the
        vertices representing a single operator have been placed and the edges 
        of which represent constraints upon which of these chips may share 
        routing identifiers for the purposes of this set of vertices.

        :param outgoing_partitions : list of outgoing partitions 
        :param placements:
        :param machine:
        :param routing_tables:

        Returns
        -------
        {(x, y): {(x, y), ...}, ...}
            An adjacency list representation of the graph described above.
        """

        # Adjacency list represent of the graph
        cluster_graph = defaultdict(set)

        for outgoing_partition in outgoing_partitions:
            # Build up a dictionary which maps each chip to a mapping of the
            # routes from this chip to the source cluster of the multicast
            # nets which take these routes. This will allow us to determine
            # which clusters need to be uniquely identified.
            chips_routes_clusters = defaultdict(lambda: defaultdict(set))

            source_placement = placements.get_placement_of_vertex(
                outgoing_partition.pre_vertex)

            for edge in outgoing_partition.edges:
                dest_placement = placements.get_placement_of_vertex(
                    edge.post_vertex)
                path = self._recursive_trace_to_destination(
                    source_placement.x, source_placement.y, outgoing_partition,
                    dest_placement.x, dest_placement.y, machine, routing_tables)

                # Ensure every cluster is in the graph
                source_chip = machine.get_chip_at(
                    source_placement.x, source_placement.y)
                for (chip, entry) in path:
                    route = (
                        Router.convert_routing_table_entry_to_spinnaker_route(
                            entry))
                    # Add this cluster to the set of clusters whose net takes
                    # this route at this point.
                    chips_routes_clusters[chip][route].add(source_chip)

                    # Add constraints to the cluster graph dependent on which
                    # multicast nets take different routes at this point.
                    routes_from_chip = chips_routes_clusters[chip]
                    for other_route, clusters in iteritems(routes_from_chip):
                        # We care about different routes
                        if other_route != route:
                            for cluster in clusters:
                                # This cluster cannot share an identifier with
                                # any of the clusters whose nets take a
                                # different route at this point.
                                cluster_graph[source_chip].add(cluster)
                                cluster_graph[cluster].add(source_chip)
        return cluster_graph

    def _allocate_fields_to_keyspaces(
            self, machine_graph, routing_by_partition, placements,
            bit_field, machine, vertex_cluster_id):

        # stores to keep the key spaces mapped.
        outgoing_partition_to_key_space = dict()
        created_fields = dict()
        created_field_names = list()
        partition_ids = self._assign_mn_net_ids(
            machine_graph, routing_by_partition, placements, machine)

        # go through each outgoing partition and create a given key space
        for outgoing_partition, connection_id in iteritems(partition_ids):

            # check all edges within the partition have the same widths (
            # input key space)
            max_width = 0
            for edge in outgoing_partition.edges:
                max_width = max(max_width, edge.reception_parameters.width)
                if ((max_width != 0) and (
                            max_width != edge.reception_parameters.width)):
                    raise Exception("I have no idea what keyspace size is for "
                                    "differing reception param widths.")

            # get constraints set by the outgoing partition
            key_constraints = self._get_key_constraints(outgoing_partition)
            extra_to_add = None

            # if the constraints contain a nengo user field value. treat as a
            #  nengo vertex that's using this outgoing partition and therefore
            #  their values need to be set
            if ((key_constraints.get_value_for_field(constants.USER_FIELD_ID)
                    == constants.USER_FIELDS.NENGO.value)):

                # set connection field value
                key_constraints.set_value_for_field(
                    constants.CONNECTION_FIELD_ID, connection_id)

                # update cluster id
                key_constraints.set_value_for_field(
                    constants.CLUSTER_FIELD_ID,
                    vertex_cluster_id[outgoing_partition.pre_vertex])
                extra_to_add = NengoKeyConstraints(
                    [NengoKeyConstraint(
                        field_id=constants.INDEX_FIELD_ID, field_value=None,
                        tags=None, start_at=constants.INDEX_FIELD_START_POINT,
                        length=None)])

            # set up the fields and values in the bitfield component for the
            # nengo key constraint
            partition_key_space = self._create_partition_key_space(
                key_constraints, bit_field, created_fields,
                created_field_names, extra_to_add=extra_to_add)

            # HORRIBLE HACK!
            if ((key_constraints.get_value_for_field(
                    constants.USER_FIELD_ID) ==
                    constants.USER_FIELDS.NENGO.value)):
                # force the max width of the neurons.
                field = {constants.INDEX_FIELD_ID: max_width - 1}
                partition_key_space(**field)

            # add to tracker for partition to its specific bit field key space
            outgoing_partition_to_key_space[outgoing_partition] = (
                partition_key_space)

        # return the partition to bit field key space component
        return outgoing_partition_to_key_space

    @staticmethod
    def _create_partition_key_space(
            key_constraints, bit_field, created_fields, created_field_names,
            extra_to_add):
        """ creates the values and fields in the bitfield component.
        
        :param key_constraints: 
        :param bit_field: 
        :param created_fields: 
        :param created_field_names: 
        :param extra_to_add:
        :return: 
        """
        new_bit_field = bit_field
        # build the fields as needed

        # add new field into bit field space if it doesnt already exist in
        # the given value
        field_level = list()
        for key_constraint in key_constraints.constraints:
            if key_constraint.field_id not in created_field_names:
                new_bit_field.add_field(
                    identifier=key_constraint.field_id,
                    length=key_constraint.length,
                    tags=key_constraint.tags, start_at=key_constraint.start_at)
                field_level.append(
                    (key_constraint.field_id, key_constraint.field_value))
                created_field_names.append(key_constraint.field_id)

        if extra_to_add is not None:
            for key_constraint in extra_to_add.constraints:
                if key_constraint.field_id not in created_field_names:
                    new_bit_field.add_field(
                        identifier=key_constraint.field_id,
                        length=key_constraint.length,
                        tags=key_constraint.tags,
                        start_at=key_constraint.start_at)
                    created_field_names.append(key_constraint.field_id)

        field_level = list()
        for key_constraint in key_constraints.constraints:
            field_level.append(
                (key_constraint.field_id, key_constraint.field_value))
            if frozenset(field_level) in created_fields:
                new_bit_field = created_fields[frozenset(field_level)]
            else:
                field = {key_constraint.field_id: key_constraint.field_value}
                new_bit_field = new_bit_field(**field)
                created_fields[frozenset(field_level)] = new_bit_field
        return new_bit_field

    @staticmethod
    def _get_key_constraints(outgoing_partition):
        """ locate 
        
        :param outgoing_partition: the outgoing partition to find constraints 
        for
        :return: the nengo key constraints holder or None if none exist
        """
        if isinstance(
                outgoing_partition.pre_vertex, AbstractNengoMachineVertex):
            outgoing_partition_constraints = \
                outgoing_partition.pre_vertex.\
                get_outgoing_partition_constraints(outgoing_partition)
            return utility_calls.locate_constraints_of_type(
                constraints=outgoing_partition_constraints,
                constraint_type=NengoKeyConstraints)[0]
        else:
            raise Exception(
                "this outgoing partition has no nengo key constraints. "
                "Dont know how to handle this")

    def _assign_mn_net_ids(
            self, machine_graph, routing_by_partition, placements, machine):
        return self._colour_graph(
            self._build_nm_net_graph(
                machine_graph, routing_by_partition, placements, machine))

    @staticmethod
    def _colour_graph(net_graph):
        """Assign colours to each node in a graph such that connected nodes 
            do not share a colour.

            Parameters
            ----------
            :param net_graph : {node: {node, ...}, ...}
                An adjacency list representation of a graph where the presence 
                of an edge indicates that two nodes may not share a colour.

            Returns
            -------
            {node: int}
                Mapping from each node to an identifier (colour).
            
        """
        # This follows a heuristic of first assigning a colour to the node
        # with the highest degree and then progressing through other nodes in a
        # breadth-first search.
        colours = deque()  # List of sets which contain vertices
        unvisited = set(iterkeys(net_graph))  # Nodes which haven't been visited

        # While there are still unvisited nodes -- note that this may be
        # true more than once if there are disconnected cliques in the graph,
        #  e.g.:
        #
        #           (c)  (d)
        #            |   /
        #            |  /            (f) --- (g)
        #            | /               \     /
        #   (a) --- (b)                 \   /
        #             \                  (h)
        #              \
        #               \          (i)
        #               (e)
        #
        # Where a valid colouring would be:
        #   0: (b), (f), (i)
        #   1: (a), (c), (d), (e), (g)
        #   2: (h)
        #
        # Nodes might be visited in the order [(b) is always first]:
        #   (b), (a), (c), (d), (e) - new clique - (f), (g), (h) - again - (i)
        while unvisited:
            queue = deque()  # Queue of nodes to visit

            # Add the node with the greatest degree to the queue
            queue.append(max(unvisited, key=lambda vx: len(net_graph[vx])))

            # Perform a breadth-first search of the tree and colour nodes as we
            # touch them.
            while queue:
                node = queue.popleft()  # Get the next node to process

                if node in unvisited:
                    # If the node is unvisited then mark it as visited
                    unvisited.remove(node)

                    # Colour the node, using the first legal colour or by
                    # creating a new colour for the node.
                    for group in colours:
                        if net_graph[node].isdisjoint(group):
                            group.add(node)
                            break
                    else:
                        # Cannot colour this node with any of the existing
                        # colours, so create a new colour.
                        colours.append({node})

                    # Add unvisited connected nodes to the queue
                    for vx in net_graph[node]:
                        queue.append(vx)

        # Reverse the data format to result in {node: colour, ...}, for
        # each group of equivalently coloured nodes mark the colour on the node.
        colouring = dict()
        for i, group in enumerate(colours):
            for vx in group:
                colouring[vx] = i
        return colouring

    def _build_nm_net_graph(
            self, machine_graph, routing_by_partition, placements, machine):
        net_graph = defaultdict(set)
        chip_route_nets = defaultdict(lambda: defaultdict(deque))
        for out_going_partition in machine_graph.outgoing_edge_partitions:
            if out_going_partition.traffic_type == EdgeTrafficType.MULTICAST:
                source_placement = placements.get_placement_of_vertex(
                    out_going_partition.pre_vertex)
                for edge in out_going_partition.edges:
                    self._process_edge(
                        placements, source_placement, net_graph, edge,
                        chip_route_nets, machine, routing_by_partition,
                        out_going_partition)
        return net_graph

    def _process_edge(
            self, placements, source_placement, net_graph, edge,
            chip_route_nets, machine, routing_by_partition,
            out_going_partition):
        dest_placement = placements.get_placement_of_vertex(
            edge.post_vertex)
        traversal = self._recursive_trace_to_destination(
            source_placement.x, source_placement.y,
            out_going_partition, dest_placement.x, dest_placement.y,
            machine, routing_by_partition)
        for (chip, entry) in traversal:
            route = Router.convert_routing_table_entry_to_spinnaker_route(
                entry)
            chip_route_nets[chip][route].append(out_going_partition)
            net_graph[out_going_partition] = set()

            # Add constraints to the net graph dependent on which nets take
            # different routes at this point.
            routes_in_chip = chip_route_nets[chip]
            for other_route, other_partitions in iteritems(routes_in_chip):
                if other_route != route:
                    for other_partition in other_partitions:
                        # This partition cannot share an identifier with any of
                        # the other partitions who take a different route at
                        # this point.
                        if other_partition != out_going_partition:
                            net_graph[out_going_partition].add(other_partition)
                            net_graph[other_partition].add(out_going_partition)

    # locates the next dest position to check
    def _recursive_trace_to_destination(
            self, chip_x, chip_y, outgoing_partition,
            dest_chip_x, dest_chip_y, machine, routing_tables):
        """ Recursively search though routing tables till no more entries are\
            registered with this key
        """

        chip = machine.get_chip_at(chip_x, chip_y)
        chips_traversed = list()

        # If reached destination, return the core
        if chip_x == dest_chip_x and chip_y == dest_chip_y:
            entry = routing_tables.get_entry_on_coords_for_edge(
                outgoing_partition, chip_x, chip_y)
            chips_traversed.append([chip, entry])
            return chips_traversed

        # If the current chip is real, find the link to the destination
        entry = routing_tables.get_entry_on_coords_for_edge(
            outgoing_partition, chip_x, chip_y)
        if entry is not None:
            chips_traversed.append([chip, entry])
            for link_id in entry.link_ids:
                link = chip.router.get_link(link_id)
                chips_traversed.extend(self._recursive_trace_to_destination(
                    link.destination_x, link.destination_y, outgoing_partition,
                    dest_chip_x, dest_chip_y, machine, routing_tables))
        return chips_traversed
