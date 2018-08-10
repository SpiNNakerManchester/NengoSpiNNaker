from nengo_spinnaker_gfe import constants
from nengo_spinnaker_gfe.graph_components.sdram_machine_edge import\
    SDRAMMachineEdge
from pacman.model.graphs.impl import OutgoingEdgePartition


class SegmentedSpikesSDRAMMachinePartition(OutgoingEdgePartition):

    __slots__ = [
        "_sdram_base_address"
    ]

    def __init__(self, identifier, label):
        OutgoingEdgePartition.__init__(
            self,  identifier=identifier, allowed_edge_types=SDRAMMachineEdge,
            label=label)
        self._sdram_base_address = None

    def total_sdram_requirements(self):
        total = 0
        for edge in self.edges:
            total += edge.sdram_size
        return total

    @property
    def sdram_base_address(self):
        return self._sdram_base_address

    @sdram_base_address.setter
    def sdram_base_address(self, new_value):
        self._sdram_base_address = new_value
        for edge in self.edges:
            edge.sdram_base_address(
                new_value + )
