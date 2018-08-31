from spinn_utilities.overrides import overrides

from nengo_spinnaker_gfe.abstracts.abstract_filter import AbstractFilter


class NoneFilter(AbstractFilter):

    __slots__ = []

    def __init__(self, width, latching):
        AbstractFilter.__init__(self, width, latching)

    @overrides(AbstractFilter.__eq__)
    def __eq__(self, other):
        return (isinstance(other, NoneFilter) and
                self._width == other.width and
                self._latching == other.latching)

    @overrides(AbstractFilter.size_words)
    def size_words(self):
        return 0

    @overrides(AbstractFilter.write_spec)
    def write_spec(self, spec, dt, width):
        pass

    @staticmethod
    @overrides(AbstractFilter.build_filter)
    def build_filter(requires_latching, reception_params, width=None):
        if width is None:
            width = reception_params.width
        return NoneFilter(width, requires_latching)
