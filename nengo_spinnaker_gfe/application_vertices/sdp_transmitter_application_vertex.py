import threading

import numpy
from spinn_utilities.overrides import overrides
from nengo_spinnaker_gfe. \
    abstracts.abstract_nengo_application_vertex import \
    AbstractNengoApplicationVertex
from nengo_spinnaker_gfe.machine_vertices. \
    sdp_transmitter_machine_vertex import \
    SDPTransmitterMachineVertex

from nengo_spinnaker_gfe.nengo_implicit_interfaces.nengo_live_input_interface\
    import NengoLiveInputInterface


class SDPTransmitterApplicationVertex(
        AbstractNengoApplicationVertex, NengoLiveInputInterface):
    """
    LPG equiv vertex (but includes filtering and some routing stuff)
    """

    __slots__ = [
        #
        '_size_in',
        #
        '_vertex',
        #
        '_output',
        #
        '_lock'
    ]

    def __init__(self, size_in, label, rng, seed):
        AbstractNengoApplicationVertex.__init__(
            self, label=label, rng=rng, seed=seed)
        NengoLiveInputInterface.__init__(self)
        self._size_in = size_in
        self._vertex = None
        self._output = numpy.zeros(self._size_out)
        self._lock = threading.Lock()

    @property
    def size_in(self):
        return self._size_in

    @overrides(NengoLiveInputInterface.output)
    def output(self, t):
        """This is a interface used by the nengo_spinnaker_gfe
        """
        with self._lock:
            return self._output

    def set_output(self, new_output):
        with self._lock:
            self._output = new_output

    @overrides(AbstractNengoApplicationVertex.create_machine_vertices)
    def create_machine_vertices(self, resource_tracker, nengo_partitioner):
        """Create vertices that will simulate the SDPTransmitter."""
        return SDPTransmitterMachineVertex(self._size_in)
