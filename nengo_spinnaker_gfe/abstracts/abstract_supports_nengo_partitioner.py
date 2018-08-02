from six import add_metaclass

from spinn_utilities.abstract_base import AbstractBase, abstractmethod


@add_metaclass(AbstractBase)
class AbstractSupportNengoPartitioner(object):

    def __init__(self):
        pass

    @abstractmethod
    def dtcm_usage_for_slice(self, neuron_slice, n_cores):
        """ compute the dtcm usage as if it was the worse loaded core
        
        :param neuron_slice: 
        :param n_cores: 
        :return: 
        """

    @abstractmethod
    def cpu_usage_for_slice(self, neuron_slice, n_cores):
        """ Compute the cpu usage as if its was the worse loaded core.
        :param neuron_slice: slice of neurons
        :param n_cores: how many cores are on the chip we're being 
        partitioned for
        :return: cpu usage resource
        """
