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
    def add_filters(filters, edge, outgoing_partition, minimise=False,
                    width=None):
        """Add signals and connections to existing lists
        of nengo_filters and key space routes

        :param filters: 
        :param edge: 
        :param outgoing_partition:
        :param minimise: 
        :param width:  It is possible to reduce the amount of memory and \
        computation required to simulate nengo_filters by combining \ 
        equivalent nengo_filters together. If minimise is `True` then this \
        is done, otherwise not.
        :return: 
        """

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
        new_parameter_filter = FilterAndRoutingRegionGenerator.locate_filter(
                filters[outgoing_partition], new_parameter_filter, minimise)
        filters[outgoing_partition].append(new_parameter_filter)

    @staticmethod
    def locate_filter(filters, new_filter, minimise):
        for already_built_param_filter in filters:
            if (new_filter == already_built_param_filter) and minimise:
                return already_built_param_filter
        return new_filter
