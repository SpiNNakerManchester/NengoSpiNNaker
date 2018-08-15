from data_specification.enums import DataType
import numpy

from nengo_spinnaker_gfe import constants


def get_seed(nengo_object):
    if hasattr(nengo_object, "seed"):
        return nengo_object.seed
    else:
        return None


def convert_numpy_array_to_s16_15(values):
    """Convert the given NumPy array of values into fixed point format."""
    # Scale and cast to appropriate int types
    scaled_values = values * DataType.S1615.scale

    # Saturate the values
    clipped_values = numpy.clip(scaled_values, DataType.S1615.min,
                                DataType.S1615.max)

    # **NOTE** for some reason just casting resulted in shape
    # being zeroed on some indeterminate selection of OSes,
    # architectures, Python and Numpy versions"
    return numpy.array(clipped_values, copy=True, dtype=numpy.int32)


def convert_transform_to_machine_vertex_level(
        transform, matrix_slice, sliced_dimension):
    fixed_point_transform = convert_numpy_array_to_s16_15(transform)
    sliced_transform = fixed_point_transform[
        expand_slice(
            matrix_slice, sliced_dimension, fixed_point_transform.ndim)]
    return sliced_transform


def expand_slice(matrix_slice, sliced_dimension, n_dim):
    if sliced_dimension is None:
        return slice(None)

    return (
        tuple(slice(None) for _ in range(sliced_dimension.value)) +
        (matrix_slice,) +
        tuple(slice(None) for _ in range(sliced_dimension.value + 1, n_dim)))


def sdram_size_in_bytes_for_filter_region(filters):
    total = 0
    total_n_filters = 0
    for outgoing_partition in filters:
        for input_filter in filters[outgoing_partition]:
            total += input_filter.size_words()
            total_n_filters += 1
    total += (
        (constants.FILTER_N_FILTERS_SIZE + (
            constants.FILTER_PARAMETERS_SIZE * total_n_filters)) *
        constants.BYTE_TO_WORD_MULTIPLIER)
    return total


def sdram_size_in_bytes_for_routing_region(n_keys):
    return ((constants.ROUTING_N_ROUTES_SIZE + (
        constants.ROUTING_ENTRIES_PER_ROUTE * n_keys)) *
            constants.BYTE_TO_WORD_MULTIPLIER)
