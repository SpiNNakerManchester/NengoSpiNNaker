import logging

from nengo_spinnaker_gfe.abstracts.\
    abstract_traffic_type_secure_outgoing_partition import \
    AbstractTrafficTypeSecureOutgoingPartition
from pacman.model.graphs.application import ApplicationEdge
from pacman.model.graphs.common import EdgeTrafficType
from pacman.model.graphs.impl import OutgoingEdgePartition
from spinn_utilities.log import FormatAdapter
from spinn_utilities.overrides import overrides

from nengo_spinnaker_gfe.abstracts.abstract_nengo_object \
    import AbstractNengoObject
logger = FormatAdapter(logging.getLogger(__name__))


class ConnectionOutgoingPartition(
        AbstractTrafficTypeSecureOutgoingPartition, AbstractNengoObject):

    __slots__ = [
        '_outgoing_edges_destinations',
        #
        '_transmission_params',
        #
        '_latching_required',
        #
        '_weight',
        #
        '_source_output_port',
        #
        "_traffic_type"
    ]

    _REPR_TEMPLATE = \
        "ConnectionOutgoingPartition(\n" \
        "pre_vertex={},identifier={}, edges={}, constraints={}, label={}, " \
        "seed={})"

    def __init__(self, rng, identifier, pre_vertex, seed):
        AbstractTrafficTypeSecureOutgoingPartition.__init__(
            self, identifier=identifier, label="connection_partition",
            allowed_edge_types=ApplicationEdge,
            traffic_type=EdgeTrafficType.MULTICAST)
        AbstractNengoObject.__init__(self, rng=rng, seed=seed)
        self._outgoing_edges_destinations = list()
        self._pre_vertex = pre_vertex

    @property
    def pre_vertex(self):
        return self._pre_vertex

    @overrides(AbstractTrafficTypeSecureOutgoingPartition.add_edge)
    def add_edge(self, edge):
        super(ConnectionOutgoingPartition, self).add_edge(edge)
        self._outgoing_edges_destinations.append(edge.post_vertex)

    @property
    def edge_destinations(self):
        return self._outgoing_edges_destinations

    @property
    @overrides(OutgoingEdgePartition.traffic_weight)
    def traffic_weight(self):
        return self._identifier.weight

    def __repr__(self):
        edges = ""
        for edge in self._edges:
            if edge.label is not None:
                edges += edge.label + ","
            else:
                edges += str(edge) + ","

        return self._REPR_TEMPLATE.format(
            self._pre_vertex, self._identifier, edges, self.constraints,
            self.label, self._seed)

    def __str__(self):
        return self.__repr__()