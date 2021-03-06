from six import add_metaclass

from spinn_utilities.abstract_base import AbstractBase, abstractmethod


@add_metaclass(AbstractBase)
class AbstractFilter(object):

    __slots__ = [
        #
        '_width',
        #
        '_latching'
    ]

    def __init__(self, width, latching):
        self._width = width
        self._latching = latching

    @property
    def width(self):
        return self._width

    @property
    def latching(self):
        return self._latching

    @abstractmethod
    def __eq__(self, other):
        """ equals method
        
        :param other: 
        :return: 
        """

    @abstractmethod
    def size_words(self):  # pragma: no cover
        """Get the number of words used to store the connection_parameters for this
        filter.
        """

    @abstractmethod
    def write_spec(self, spec, dt, width):
        """ pack the data into the dsg file        
        :param spec: 
        :param dt: 
        :return: 
        """

    @staticmethod
    @abstractmethod
    def build_filter(requires_latching, reception_params, width=None):
        """ builds a filter based off these params
        
        :param requires_latching: 
        :param reception_params: 
        :param width: 
        :return: a filter object
        """