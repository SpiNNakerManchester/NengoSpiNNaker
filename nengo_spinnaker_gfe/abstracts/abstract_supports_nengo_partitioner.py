from six import add_metaclass

from spinn_utilities.abstract_base import AbstractBase, abstractmethod


@add_metaclass(AbstractBase)
class AbstractSupportNengoPartitioner(object):

    __slots__ = []

    def __init__(self):
        pass

    @abstractmethod
    def get_resources_for_slices(self, slices, n_cores):
        """ compute the resoruces used as if it was the worse loaded core
        
        :param slices: a list of either the number of neurons, or a set of 
        slices, specific to the vertex type.  
        :param n_cores: the number of cores its expected to partition over
        :return: a resource container
        """
