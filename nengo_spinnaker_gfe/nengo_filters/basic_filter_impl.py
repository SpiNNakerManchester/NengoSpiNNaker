from nengo_spinnaker_gfe.abstracts.abstract_filter import AbstractFilter


class BasicFilterImpl(AbstractFilter):

    BASIC_FILTER_N_WORDS = 3

    def __init__(self, width, latching):
        AbstractFilter.__init__(self, width, latching)

    def write_basic_spec(self, spec, width):
        spec.write_value(width or self.width)
        spec.write_value(0x1 if self.latching else 0x0)

    def size_words(self):
        return self.BASIC_FILTER_N_WORDS
