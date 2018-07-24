from pacman.model.graphs import AbstractVertex
from pacman.model.graphs.common import ConstrainedObject
from six import add_metaclass
from spinn_utilities.abstract_base import AbstractBase, abstractmethod
from spinn_utilities.overrides import overrides

from nengo_spinnaker_gfe.abstracts.abstract_nengo_object import AbstractNengoObject


@add_metaclass(AbstractBase)
class AbstractNengoApplicationVertex(
        AbstractNengoObject, AbstractVertex, ConstrainedObject):

    #__slots__ = [
        # the label of this vertex
    #    '_label',
    #]

    def __init__(self, label, rng, seed, constraints=None):
        if constraints is None:
            constraints = []
        ConstrainedObject.__init__(self, constraints)
        AbstractVertex.__init__(self)
        AbstractNengoObject.__init__(self, rng, seed)
        self._label = label

    @property
    @overrides(ConstrainedObject.constraints)
    def constraints(self):
        return self._constraints

    @overrides(ConstrainedObject.add_constraint)
    def add_constraint(self, constraint):
        self._constraints.add_constraints(constraint)

    @property
    @overrides(AbstractVertex.label)
    def label(self):
        return self._label

    def __str__(self):
        return self.label

    def __repr__(self):
        return "ApplicationVertex(label={}, constraints={}, seed={}".format(
            self.label, self.constraints, self._seed)

    @abstractmethod
    def create_machine_vertices(self, resource_tracker):
        """
        
        :return: 
        """
