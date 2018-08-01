from spinn_utilities.overrides import overrides
from nengo_spinnaker_gfe.connection_parameters. \
    abstract_transmission_parameters import \
    AbstractTransmissionParameters
from nengo_spinnaker_gfe.utility_objects.parameter_transform import \
    ParameterTransform

from nengo_spinnaker_gfe.connection_parameters.\
    transmission_parameters_impl import TransmissionParametersImpl


class NodeTransmissionParameters(
        TransmissionParametersImpl, AbstractTransmissionParameters):

    __slots__ = [
        #
        "_pre_slice",
        #
        "_parameter_function"]

    def __init__(self, transform, pre_slice=slice(None),
                 parameter_function=None):
        """
        """
        TransmissionParametersImpl.__init__(self, transform)
        AbstractTransmissionParameters.__init__(self)
        self._pre_slice = pre_slice
        self._parameter_function = parameter_function

    @property
    def pre_slice(self):
        return self._pre_slice

    @property
    def parameter_function(self):
        return self._parameter_function

    def __repr__(self):
        return "{}:{}:{}".format(
            self._transform, self._pre_slice, self._parameter_function)

    def __str__(self):
        return self.__repr__()

    @overrides(TransmissionParametersImpl.__eq__)
    def __eq__(self, other):
        return (super(NodeTransmissionParameters, self).__eq__(other) and
                self._pre_slice == other.pre_slice and
                self._parameter_function is other.parameter_function)

    @overrides(TransmissionParametersImpl.__hash__)
    def __hash__(self):
        return hash((type(self), self._parameter_function, self._transform))

    @overrides(AbstractTransmissionParameters.concat)
    def concat(self, other):
        """Create new connection connection_parameters which are the result of
        concatenating this connection another.

        Parameters
        ----------
        other : NodeTransmissionParameters
            Connection connection_parameters to add to the end of this connection.

        Returns
        -------
        NodeTransmissionParameters or None
            Either a new set of transmission connection_parameters, or None if the
            resulting transform contained no non-zero values.
        """
        # Get the outgoing transformation
        new_transform = self._transform.concat(other._transform)

        # Create a new connection (unless the resulting transform is empty,
        # in which case don't)
        if new_transform is not None:
            return NodeTransmissionParameters(
                    new_transform, self._pre_slice, self._parameter_function)
        else:
            # The transform consisted entirely of zeros so return None.
            return None

    @property
    @overrides(TransmissionParametersImpl.as_global_inhibition_connection)
    def as_global_inhibition_connection(self):
        """Construct a copy of the connection with the optimisation for global
        inhibition applied.
        """
        assert self.supports_global_inhibition
        transform = self.full_transform(slice_out=False)[0, :]

        return NodeTransmissionParameters(
            ParameterTransform(
                size_in=self.size_in, size_out=1, transform=transform,
                slice_in=self.slice_in),
            pre_slice=self._pre_slice,
            parameter_function=self._parameter_function)
