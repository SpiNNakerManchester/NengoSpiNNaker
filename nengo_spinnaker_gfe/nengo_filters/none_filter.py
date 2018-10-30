from nengo_spinnaker_gfe.nengo_filters.basic_filter_impl import BasicFilterImpl
from spinn_utilities.overrides import overrides

from nengo_spinnaker_gfe.abstracts.abstract_filter import AbstractFilter


class NoneFilter(BasicFilterImpl):

    __slots__ = []

    def __init__(self, width, latching):
        BasicFilterImpl.__init__(self, width, latching)

    @overrides(BasicFilterImpl.__eq__)
    def __eq__(self, other):
        return (isinstance(other, NoneFilter) and
                self._width == other.width and
                self._latching == other.latching)

    @overrides(BasicFilterImpl.size_words)
    def size_words(self):
        return BasicFilterImpl.size_words(self)

    @overrides(BasicFilterImpl.write_spec)
    def write_spec(self, spec, dt, width):
        BasicFilterImpl.write_basic_spec(self, spec, width)

    @staticmethod
    @overrides(BasicFilterImpl.build_filter)
    def build_filter(requires_latching, reception_params, width=None):
        if width is None:
            width = reception_params.width
        return NoneFilter(width, requires_latching)
