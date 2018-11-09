from six import add_metaclass

from spinn_utilities.abstract_base import AbstractBase, abstractmethod


@add_metaclass(AbstractBase)
class AbstractProbeable(object):

    __slots__ = []

    def __init__(self):
        pass

    @abstractmethod
    def can_probe_variable(self, variable):
        """
        
        :param variable: 
        :return: 
        """

    @abstractmethod
    def is_set_probeable_variable(self, variable):
        """ informs user if the variable is set to be recorded
        
        :param variable: the variable to check
        :return: bool
        """


    @abstractmethod
    def set_probeable_variable(self, variable):
        """
        
        :param variable: 
        :return: 
        """

    @abstractmethod
    def get_data_for_variable(
            self, variable, run_time, placements, graph_mapper, buffer_manager):
        """
        
        :param variable: 
        :param run_time: 
        :param placements: 
        :param graph_mapper: 
        :param buffer_manager: 
        :return: 
        """
        # pylint: disable=too-many-arguments

    @abstractmethod
    def get_possible_probeable_variables(self):
        """
        
        :return: 
        """