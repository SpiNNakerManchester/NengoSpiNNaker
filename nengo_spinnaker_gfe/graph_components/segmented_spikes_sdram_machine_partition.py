from nengo_spinnaker_gfe import constants
from nengo_spinnaker_gfe.abstracts.\
    abstract_traffic_type_secure_outgoing_partition import \
    AbstractTrafficTypeSecureOutgoingPartition
from nengo_spinnaker_gfe.graph_components.sdram_machine_edge import\
    SDRAMMachineEdge
from pacman.model.graphs.common import EdgeTrafficType
import math


class SegmentedSpikesSDRAMMachinePartition(
        AbstractTrafficTypeSecureOutgoingPartition):

    __slots__ = [
        "_sdram_base_address",
        "_pre_vertex"
    ]

    def __init__(self, identifier, pre_vertex, label):
        AbstractTrafficTypeSecureOutgoingPartition.__init__(
            self,  identifier=identifier, allowed_edge_types=SDRAMMachineEdge,
            label=label, traffic_type=EdgeTrafficType.SDRAM)
        self._sdram_base_address = None
        self._pre_vertex = pre_vertex

    def total_sdram_requirements(self):
        total = 0
        for edge in self.edges:
            total += edge.sdram_size
        return total

    @property
    def pre_vertex(self):
        return self._pre_vertex

    @property
    def sdram_base_address(self):
        return self._sdram_base_address

    @sdram_base_address.setter
    def sdram_base_address(self, new_value):
        self._sdram_base_address = new_value

        sorted_by_low_atom = sorted(
            self.edges, key=lambda e: e.post_vertex.neuron_slice.lo_atom)

        base_address_offset = new_value
        for edge in sorted_by_low_atom:
            edge.sdram_base_address = new_value
            base_address_offset += (
                math.ceil(edge.post_vertex.neuron_slice.n_atoms /
                          constants.WORD_TO_BIT_CONVERSION))
