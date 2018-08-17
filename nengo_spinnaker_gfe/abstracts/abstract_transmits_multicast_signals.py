from six import add_metaclass

from spinn_utilities.abstract_base import AbstractBase, abstractmethod


@add_metaclass(AbstractBase)
class AbstractTransmitsMulticastSignals(object):
    __slots__ = []

    def __init__(self):
        pass

    @abstractmethod
    def transmits_multicast_signals(self, transmission_params):
        """ method to verify if the object transmit the type of multicast \
        signals

        :param transmission_params: the transmission param of the multicast edge
        :return: bool
        """
