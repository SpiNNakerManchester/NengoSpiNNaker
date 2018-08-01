

class DestinationInputPortLearningRule(object):

    __slots__ = [
        "_learning_rule",
        "_destination_input_port"
    ]

    def __init__(self, destination_input_port, learning_rule=None):
        self._learning_rule = learning_rule
        self._destination_input_port = destination_input_port

    @property
    def learning_rule(self):
        return self._learning_rule

    @property
    def destination_input_port(self):
        return self._destination_input_port

    @destination_input_port.setter
    def destination_input_port(self, new_value):
        self._destination_input_port = new_value
