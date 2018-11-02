import numpy

from nengo_spinnaker_gfe.nengo_filters.basic_filter_impl import BasicFilterImpl
from spinn_utilities.overrides import overrides
from nengo_spinnaker_gfe import helpful_functions


class LowPassFilter(BasicFilterImpl):

    __slots__ = [
        #
        '_time_constant'
    ]

    SIZE_OF_WORDS = 2
    FIXED_TIME_CONSTANT = 0.0
    SECOND_CO_EFFICIENT = 1.0

    def __init__(self, width, latching, time_constant):
        BasicFilterImpl.__init__(self, width, latching)
        self._time_constant = time_constant

    @property
    def time_constant(self):
        return self._time_constant

    @overrides(BasicFilterImpl.__eq__)
    def __eq__(self, other):
        return (isinstance(other, LowPassFilter) and
                self._width == other.width and
                self._latching == other.latching and
                self._time_constant == other.time_constant)

    @overrides(BasicFilterImpl.size_words)
    def size_words(self):
        return BasicFilterImpl.size_words(self) + self.SIZE_OF_WORDS

    @staticmethod
    @overrides(BasicFilterImpl.build_filter)
    def build_filter(requires_latching, reception_params, width=None):
        if width is None:
            width = reception_params.width
        return LowPassFilter(
            width, requires_latching, reception_params.parameter_filter.tau)

    @overrides(BasicFilterImpl.write_spec)
    def write_spec(self, spec, dt, width):

        BasicFilterImpl.write_basic_spec(self, spec, width)

        """Pack the struct describing the filter into the buffer."""
        # Compute the coefficients
        if self.time_constant != self.FIXED_TIME_CONSTANT:
            a = numpy.exp(-dt / self.time_constant)
        else:
            a = self.FIXED_TIME_CONSTANT

        b = self.SECOND_CO_EFFICIENT - a

        print "low pass a {}".format(a)
        spec.write_value(helpful_functions.convert_numpy_array_to_s16_15(a))
        print "low pass b {}".format(b)
        spec.write_value(helpful_functions.convert_numpy_array_to_s16_15(b))
