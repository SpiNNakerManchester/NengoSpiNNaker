from spinn_utilities.overrides import overrides

import nengo
from nengo_spinnaker_gfe.nengo_implicit_interfaces.\
    nengo_live_output_interface import NengoLiveOutputInterface


class NengoOutputNode(nengo.Node, NengoLiveOutputInterface):

    def __init__(self, spinnaker_vertex, label):
        NengoLiveOutputInterface.__init__(self)
        self._spinnaker_vertex = spinnaker_vertex
        self._label = label
        self._size_in = spinnaker_vertex.size_in
        self._size_out = 0

    @property
    def label(self):
        return self._label

    @property
    def size_in(self):
        return self._size_in

    @property
    def size_out(self):
        return self._size_out

    @overrides(NengoLiveOutputInterface.output)
    def output(self, t, x):
        """ enforced by the nengo_spinnaker_gfe duck typing

        :param t: a unused time value.
        :param x: the value to send to the output vertex
        :return: 
        """
        result = self._spinnaker_vertex.output(t, x)
        return result

    @overrides(nengo.Node.__getstate__)
    def __getstate__(self):
        raise NotImplementedError("Nengo objects do not support pickling")

    @overrides(nengo.Node.__setstate__)
    def __setstate__(self, state):
        raise NotImplementedError("Nengo objects do not support pickling")
