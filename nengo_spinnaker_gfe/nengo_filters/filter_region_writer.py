from nengo_spinnaker_gfe.nengo_exceptions import NotRecognisedFilterType
from nengo_spinnaker_gfe.nengo_filters.linear_filter import LinearFilter
from nengo_spinnaker_gfe.nengo_filters.low_pass_filter import LowPassFilter
from nengo_spinnaker_gfe.nengo_filters.none_filter import NoneFilter

import itertools


def write_filter_region(
        spec, machine_time_step_in_seconds, input_slice, filters):
    # write how many filters there are
    low_pass_filters, none_type_filters, linear_filters = \
        _separate_filter_types(filters)
    filter_to_index_map = dict()

    # write sizes
    spec.write_value(len(low_pass_filters))
    spec.write_value(len(none_type_filters))
    spec.write_value(len(linear_filters))

    # count tracker
    written_order = 0

    # process filters
    for a_filter in itertools.chain(
            low_pass_filters, none_type_filters, linear_filters):
        a_filter.write_spec(
            spec, machine_time_step_in_seconds, input_slice.n_atoms)
        filter_to_index_map[a_filter] = written_order
        written_order += 1
    return filter_to_index_map


def _separate_filter_types(filters):
    low_pass = list()
    none_type = list()
    linear = list()
    for outgoing_partition in filters:
        for filter_to_categorise in filters[outgoing_partition]:
            if isinstance(filter_to_categorise, NoneFilter):
                none_type.append(filter_to_categorise)
            elif isinstance(filter_to_categorise, LowPassFilter):
                low_pass.append(filter_to_categorise)
            elif isinstance(filter_to_categorise, LinearFilter):
                linear.append(filter_to_categorise)
            else:
                raise NotRecognisedFilterType(
                    "don't know how to separate this filter")
    return low_pass, none_type, linear
