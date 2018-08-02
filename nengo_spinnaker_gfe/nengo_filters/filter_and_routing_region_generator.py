import nengo

from nengo.synapses import Lowpass as NengoLowPass
from nengo.synapses import LinearFilter as NengoLinearFilter
from nengo_spinnaker_gfe.nengo_exceptions import NotRecognisedFilterType
from nengo_spinnaker_gfe.nengo_filters.linear_filter import LinearFilter
from nengo_spinnaker_gfe.nengo_filters.low_pass_filter import LowPassFilter
from nengo_spinnaker_gfe.nengo_filters.none_filter import NoneFilter


class FilterAndRoutingRegionGenerator(object):

    def __init__(self):
        self._supported_filter_types = {
            None: NoneFilter,
            nengo.synapses.Lowpass: LowPassFilter,
            nengo.synapses.LinearFilter: LinearFilter,
        }

    def generate_filter_and_filter_routing_regions(
            self, spec, machine_graph, filter_region_id,
            filter_routing_region_id,
            minimise=False, filter_routing_tag="filter_routing",
            index_field="index", width=None):
        filters, signal_routes = self._make_filters(
            machine_graph, minimise=minimise, width=width)

        # Create the regions
        filter_region = FilterRegion(filters, dt)
        routing_region = FilterRoutingRegion(signal_routes, filter_routing_tag,
                                             index_field)

    @staticmethod
    def add_filters(filters, outgoing_partition, minimise=False, width=None):
        """Add signals and connections to existing lists
        of nengo_filters and keyspace routes

        :param filters: 
        :param outgoing_partition: 
        :param minimise: 
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
                        outgoing_partition.identifier, reception_params)
            elif isinstance(reception_params.parameter_filter,
                            NengoLinearFilter):
                new_parameter_filter = LinearFilter.build_filter(
                    outgoing_partition.indentifer, reception_params)
            elif isinstance(reception_params.parameter_filter, None):
                new_parameter_filter = NoneFilter.build_filter(
                    outgoing_partition.indentifer, reception_params)
            else:
                raise NotRecognisedFilterType(
                    "Do not recognise the filter type {}".format(type(
                        reception_params.parameter_filter)))

            # Store the filter
            for already_built_param_filter in filters:
                if (new_parameter_filter == already_built_param_filter and
                        minimise):
                    break
            else:
                filters.append(new_parameter_filter)
        return filters

    def _make_filters(self, machine_graph, minimise, width):
        """Create a list of nengo_filters and keyspace routes from the given
        signals and connections.

        Parameters
        ----------
        specs : [ReceptionSpec, ...]
            List of reception specs (as generated by `get_signals_to`) to build the
            filter regions for.

        Other Parameters
        ----------------
        minimise : bool
            It is possible to reduce the amount of memory and computation required
            to simulate nengo_filters by combining equivalent nengo_filters together.  If
            minimise is `True` then this is done, otherwise not.
        """
        # Create new lists of nengo_filters and the routing entries
        filters = list()
        signal_routes = list()

        # Add signals and connections to lists
        return self.add_filters(filters, signal_routes, specs, minimise, width)
