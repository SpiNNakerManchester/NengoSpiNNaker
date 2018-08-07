

class PESLearningRule(object):

    __slots__ = [
        "_learning_rate",
        "_decoder_start",
        "_decoder_stop"
    ]

    def __init__(self, learning_rate, decoder_start, decoder_stop):
        self._learning_rate = learning_rate
        self._decoder_start = decoder_start
        self._decoder_stop = decoder_stop

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def decoder_start(self):
        return self._decoder_start

    @property
    def decoder_stop(self):
        return self._decoder_stop
