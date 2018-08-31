import numpy
from nengo.utils.filter_design import cont2discrete
from nengo_spinnaker_gfe.nengo_filters.basic_filter_impl import BasicFilterImpl
from spinn_utilities.overrides import overrides
from nengo_spinnaker_gfe import helpful_functions


class LinearFilter(BasicFilterImpl):
    __slots__ = [
        #
        '_num',
        #
        '_den',
        #
        '_order'
    ]

    def __init__(self, width, latching, num, den):
        BasicFilterImpl.__init__(self, width, latching)
        self._num = numpy.array(num)
        self._den = numpy.array(den)
        self._order = len(den) - 1

    @property
    def num(self):
        return self._num

    @property
    def den(self):
        return self._den

    @property
    def order(self):
        return self._order

    @overrides(BasicFilterImpl.__eq__)
    def __eq__(self, other):
        return (isinstance(other, LinearFilter) and
                self._width == other.width and
                self._latching == other.latching and
                self._num == other.num and self._den == other.den and
                self._order == other.order)

    @overrides(BasicFilterImpl.size_words)
    def size_words(self):
        return BasicFilterImpl.size_words(self) + 1 + self._order * 2

    @staticmethod
    @overrides(BasicFilterImpl.build_filter)
    def build_filter(requires_latching, reception_params, width=None):
        if width is None:
            width = reception_params.width
        return LinearFilter(
            width, requires_latching, reception_params.filter.num,
            reception_params.filter.den)

    @overrides(BasicFilterImpl.write_spec)
    def write_spec(self, spec, dt, width):
        BasicFilterImpl.write_basic_spec(self, spec, width)

        """Pack the struct describing the filter into the buffer."""
        # Compute the filter coefficients
        b, a, _ = cont2discrete((self.num, self.den), dt)
        b = b.flatten()

        # Strip out the first values
        # `a` is negated so that it can be used with a multiply-accumulate
        # instruction on chip.
        assert b[0] == 0.0  # Oops!
        ab = numpy.vstack((-a[1:], b[1:])).T.flatten()

        # Convert the values to fixpoint and write into a data buffer
        spec.write_value(self.order)
        spec.write_array(helpful_functions.convert_numpy_array_to_s16_15(ab))
