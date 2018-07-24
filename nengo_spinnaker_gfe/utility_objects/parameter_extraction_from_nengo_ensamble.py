import nengo
from nengo import builder as nengo_builder
from nengo.builder import ensemble
from nengo.utils import numpy as nengo_numpy
import numpy


class ParameterExtractionFromNengoEnsemble(object):

    __slots__ = [
        "_eval_points",
        "_encoders",
        "_scaled_encoders",
        "_max_rates",
        "_intercepts",
        "_gain",
        "_bias"]

    def __init__(self, nengo_ensemble, random_number_generator):
        """ goes through the nengo_spinnaker_gfe ensemble object and extracts the 
        connection_parameters for the lif neurons

        :param nengo_ensemble: the ensemble handed down by nengo_spinnaker_gfe
        :param random_number_generator: the random number generator 
        controlling all random in this nengo_spinnaker_gfe run
        :return: dict of params with names.
        """

        eval_points = nengo_builder.ensemble.gen_eval_points(
            nengo_ensemble, nengo_ensemble.eval_points,
            rng=random_number_generator)

        # Get the encoders
        if isinstance(nengo_ensemble.encoders, nengo.dists.Distribution):
            encoders = nengo_ensemble.encoders.sample(
                nengo_ensemble.n_neurons, nengo_ensemble.dimensions,
                rng=random_number_generator)
            encoders = numpy.asarray(encoders, dtype=numpy.float64)
        else:
            encoders = nengo_numpy.array(
                nengo_ensemble.encoders, min_dims=2, dtype=numpy.float64)
        encoders /= nengo_numpy.norm(encoders, axis=1, keepdims=True)

        # Get correct sample function (seems dists.get_samples not in nengo
        # dists in some versions, so has to be a if / else)
        # TODO figure out which one we're following. having both is crazy
        if hasattr(ensemble, 'sample'):
            sample_function = ensemble.sample
        else:
            sample_function = nengo.dists.get_samples

        # Get maximum rates and intercepts
        max_rates = sample_function(
            nengo_ensemble.max_rates, nengo_ensemble.n_neurons,
            rng=random_number_generator)
        intercepts = sample_function(
            nengo_ensemble.intercepts, nengo_ensemble.n_neurons,
            rng=random_number_generator)

        # Build the neurons
        if nengo_ensemble.gain is None and nengo_ensemble.bias is None:
            gain, bias = nengo_ensemble.neuron_type.gain_bias(
                max_rates, intercepts)
        elif (nengo_ensemble.gain is not None and
                nengo_ensemble.bias is not None):
            gain = sample_function(
                nengo_ensemble.gain, nengo_ensemble.n_neurons,
                rng=random_number_generator)
            bias = sample_function(
                nengo_ensemble.bias, nengo_ensemble.n_neurons,
                rng=random_number_generator)
        else:
            raise NotImplementedError(
                "gain or bias set for {!s}, but not both. Solving for one "
                "given the other is not yet implemented.".format(
                    nengo_ensemble))

        # Scale the encoders
        scaled_encoders = \
            encoders * (gain / nengo_ensemble.radius)[:, numpy.newaxis]

        self._intercepts = intercepts
        self._gain = gain
        self._bias = bias
        self._max_rates = max_rates
        self._encoders = encoders
        self._scaled_encoders = scaled_encoders
        self._eval_points = eval_points

    @property
    def intercepts(self):
        return self._intercepts

    @property
    def gain(self):
        return self._gain

    @property
    def bias(self):
        return self._bias

    @property
    def max_rates(self):
        return self._max_rates

    @property
    def encoders(self):
        return self._encoders

    @property
    def scaled_encoders(self):
        return self._scaled_encoders

    @property
    def eval_points(self):
        return self._eval_points
