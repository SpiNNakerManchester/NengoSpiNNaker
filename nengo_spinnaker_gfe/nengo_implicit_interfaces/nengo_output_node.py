from spinn_utilities.overrides import overrides

import nengo
from nengo_spinnaker_gfe.nengo_implicit_interfaces.\
    nengo_live_output_interface import NengoLiveOutputInterface


class NengoOutputNode(nengo.Node, NengoLiveOutputInterface):

    def __init__(self, spinnaker_vertex):
        NengoLiveOutputInterface.__init__(self)
        self._spinnaker_vertex = spinnaker_vertex

    @property
    def size_in(self):
        return self._spinnaker_vertex.size_in

    @overrides(NengoLiveOutputInterface.output)
    def output(self, t, x):
        """ enforced by the nengo_spinnaker_gfe duck typing

        :param t: 
        :param x:
        :return: 
        """
        return self._spinnaker_vertex.output(t, x)
