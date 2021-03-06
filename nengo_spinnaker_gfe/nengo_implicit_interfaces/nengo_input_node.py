from spinn_utilities.overrides import overrides

import nengo
from nengo_spinnaker_gfe.nengo_implicit_interfaces.\
    nengo_live_input_interface import NengoLiveInputInterface


class NengoInputNode(nengo.Node, NengoLiveInputInterface):

    def __init__(self, spinnaker_vertex):
        nengo.Node.__init__(self)
        NengoLiveInputInterface.__init__(self)
        self._spinnaker_vertex = spinnaker_vertex

    @property
    def size_in(self):
        return self._spinnaker_vertex.size_in

    @overrides(NengoLiveInputInterface.output)
    def output(self, t):
        """ enforced by the nengo_spinnaker_gfe duck typing

        :param t: pointless parameter
        :return: 
        """
        return self._spinnaker_vertex.output

    @overrides(nengo.Node.__getstate__)
    def __getstate__(self):
        raise NotImplementedError("Nengo objects do not support pickling")

    @overrides(nengo.Node.__setstate__)
    def __setstate__(self, state):
        raise NotImplementedError("Nengo objects do not support pickling")
