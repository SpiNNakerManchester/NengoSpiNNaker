import numpy
from nengo.utils.filter_design import cont2discrete
from spinn_utilities.overrides import overrides
from nengo_spinnaker_gfe import helpful_functions

from nengo_spinnaker_gfe.abstracts.abstract_filter import \
    AbstractFilter


class LinearFilter(AbstractFilter):
    __slots__ = [
        #
        '_num',
        #
        '_den',
        #
        '_order'
    ]

    def __init__(self, width, latching, num, den):
        AbstractFilter.__init__(self, width, latching)
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

    @overrides(AbstractFilter.__eq__)
    def __eq__(self, other):
        return (isinstance(other, LinearFilter) and
                self._width == other.width and
                self._latching == other.latching and
                self._num == other.num and self._den == other.den and
                self._order == other.order)

    @overrides(AbstractFilter.size_words)
    def size_words(self):
        return 1 + self._order * 2

    @staticmethod
    @overrides(AbstractFilter.build_filter)
    def build_filter(requires_latching, reception_params, width=None):
        if width is None:
            width = reception_params.width
        return LinearFilter(
            width, requires_latching, reception_params.filter.num,
            reception_params.filter.den)

    @overrides(AbstractFilter.pack_into)
    def pack_into(self, spec, dt):
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