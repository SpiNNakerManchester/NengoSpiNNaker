from pacman.model.graphs.common import EdgeTrafficType
from pacman.model.graphs.machine import MachineEdge


class ConnectionMachineEdge(MachineEdge):

    def __init__(self, pre_vertex, post_vertex, input_port,
                 reception_parameters, traffic_weight, label=None):
        MachineEdge.__init__(
            self, pre_vertex, post_vertex,
            traffic_type=EdgeTrafficType.MULTICAST, label=label,
            traffic_weight=traffic_weight)
        self._input_port = input_port
        self._reception_parameters = reception_parameters

    @property
    def input_port(self):
        return self._input_port

    @property
    def reception_parameters(self):
        return self._reception_parameters

    def __repr__(self):
        return "edge between {} and {}".format(
            self._pre_vertex, self._post_vertex)

    def __str__(self):
        return self.__repr__()
