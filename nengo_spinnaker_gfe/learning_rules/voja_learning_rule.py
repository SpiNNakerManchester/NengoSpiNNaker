

class VojaLearningRule(object):

    __slots__ = [
        "_learning_rate",
        "_encoder_offset",
    ]

    def __init__(self, learning_rate, encoder_offset):
        self._learning_rate = learning_rate
        self._encoder_offset = encoder_offset

    @property
    def learning_rule(self):
        return self._learning_rate

    @property
    def encoder_offset(self):
        return self._encoder_offset
