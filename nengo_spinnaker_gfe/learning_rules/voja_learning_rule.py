

class VojaLearningRule(object):

    __slots__ = [
        "_learning_rate",
        "_encoder_offset",
        "_learning_signal_filter_index",
        "_decoded_input_filter_index"
    ]

    def __init__(
            self, learning_rate, encoder_offset, learning_signal_filter_index,
            decoded_input_filter_index):
        self._learning_rate = learning_rate
        self._encoder_offset = encoder_offset
        self._learning_signal_filter_index = learning_signal_filter_index
        self._decoded_input_filter_index = decoded_input_filter_index

    @property
    def learning_signal_filter_index(self):
        return self._learning_signal_filter_index

    @property
    def decoded_input_filter_index(self):
        return self._decoded_input_filter_index

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def encoder_offset(self):
        return self._encoder_offset
