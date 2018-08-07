import nengo

from nengo.synapses import Lowpass as NengoLowPass
from nengo.synapses import LinearFilter as NengoLinearFilter
from nengo_spinnaker_gfe.nengo_exceptions import NotRecognisedFilterType
from nengo_spinnaker_gfe.nengo_filters.linear_filter import LinearFilter
from nengo_spinnaker_gfe.nengo_filters.low_pass_filter import LowPassFilter
from nengo_spinnaker_gfe.nengo_filters.none_filter import NoneFilter


class FilterAndRoutingRegionGenerator(object):

    def __init__(self):
        pass

    @staticmethod
    def add_filters(filters, outgoing_partition, learning_rule, minimise=False,
                    width=None):
        """Add signals and connections to existing lists
        of nengo_filters and keyspace routes

        :param filters: 
        :param outgoing_partition: 
        :param minimise: 
        :param learning_rule
        :param width:  It is possible to reduce the amount of memory and \
        computation required to simulate nengo_filters by combining \ 
        equivalent nengo_filters together. If minimise is `True` then this \
        is done, otherwise not.
        :return: 
        """

        for edge in outgoing_partition.edges:
            reception_params = edge.reception_parameters
            if isinstance(reception_params.parameter_filter, NengoLowPass):
                new_parameter_filter = LowPassFilter.build_filter(
                    outgoing_partition.identifier.latching_required,
                    reception_params, width)
            elif isinstance(reception_params.parameter_filter,
                            NengoLinearFilter):
                new_parameter_filter = LinearFilter.build_filter(
                    outgoing_partition.identifier.latching_required,
                    reception_params, width)
            elif reception_params.parameter_filter is None:
                new_parameter_filter = NoneFilter.build_filter(
                    outgoing_partition.identifier.latching_required,
                    reception_params, width)
            else:
                raise NotRecognisedFilterType(
                    "Do not recognise the filter type {}".format(type(
                        reception_params.parameter_filter)))

            # Store the filter
            for already_built_param_filter in filters[learning_rule]:
                if (new_parameter_filter == already_built_param_filter and
                        minimise):
                    break
            else:
                filters[learning_rule].append(new_parameter_filter)