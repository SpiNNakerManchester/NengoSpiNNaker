from nengo_spinnaker_gfe import constants
from nengo_spinnaker_gfe.graph_components.sdram_machine_edge import\
    SDRAMMachineEdge
from nengo_spinnaker_gfe.nengo_exceptions import NengoSDRAMSizeException
from pacman.model.graphs.impl import OutgoingEdgePartition


class ConstantSDRAMMachinePartition(OutgoingEdgePartition):

    __slots__ = [
        "_sdram_base_address"
    ]

    def __init__(self, identifier, label):
        OutgoingEdgePartition.__init__(
            self,  identifier=identifier, allowed_edge_types=SDRAMMachineEdge,
            label=label)
        self._sdram_base_address = None

    def total_sdram_requirements(self):
        expected_size = self.edges[0].sdram_size
        for edge in self.edges:
            if edge.sdram_size != expected_size:
                raise NengoSDRAMSizeException(
                    "The edges within the constant sdram partition {} have "
                    "inconsistent memory size requests. ")
        return expected_size

    @property
    def sdram_base_address(self):
        return self._sdram_base_address

    @sdram_base_address.setter
    def sdram_base_address(self, new_value):
        self._sdram_base_address = new_value
        for edge in self.edges:
            edge.sdram_base_address(self._sdram_base_address)
