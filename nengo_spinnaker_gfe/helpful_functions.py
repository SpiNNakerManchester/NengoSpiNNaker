from data_specification.enums import DataType
import numpy

from nengo_spinnaker_gfe import constants


def get_seed(nengo_object):
    """ generated a seed for a nengo object
    
    :param nengo_object:  the nengo object
    :return: either the seed or None is the object has no seed
    """
    if hasattr(nengo_object, "seed"):
        return nengo_object.seed
    else:
        return None


def convert_numpy_array_to_s16_15(values):
    """Convert the given NumPy array of values into fixed point format.
    
    :param values: the values to convert
    :return: s1615 array
    """
    # Scale and cast to appropriate int types
    scaled_values = values * DataType.S1615.scale

    # Saturate the values
    clipped_values = numpy.clip(scaled_values, DataType.S1615.min,
                                DataType.S1615.max)

    # **NOTE** for some reason just casting resulted in shape
    # being zeroed on some indeterminate selection of OSes,
    # architectures, Python and Numpy versions"
    return numpy.array(clipped_values, copy=True, dtype=numpy.int32)


def convert_s16_15_to_numpy_array(values):
    """Convert the given fixed point array of values into a numpy array format.
    
    :param values: the values to convert
    :return: a numpy array
    """
    return values / DataType.S1615.scale


def convert_matrix_to_machine_vertex_level(
        transform, matrix_slice, sliced_dimension):
    """ converts a matrix from a application vertex, inside the matrix for the
    machine vertex
    
    :param transform: the matrix to convert 
    :param matrix_slice: the machine vertex slice
    :param sliced_dimension:  the sliced dimension (rows / columns)
    :return: the sliced matrix
    """
    sliced_transform = transform[_expand_slice(
        matrix_slice, sliced_dimension, transform.ndim)]
    return sliced_transform


def _expand_slice(matrix_slice, sliced_dimension, n_dim):
    """ takes a slice and converts it into a form that the numpy array can 
    use for generating a machine slice from a application matrix
    
    :param matrix_slice: the slice to convert
    :param sliced_dimension: the dimension of the slice
    :param n_dim: the number of dimensions the numpy array has
    :return: a slicing object the numpy array can recognise
    """
    if sliced_dimension is None:
        return slice(None)

    return (
        tuple(slice(None) for _ in range(sliced_dimension.value)) +
        (matrix_slice,) +
        tuple(slice(None) for _ in range(sliced_dimension.value + 1, n_dim)))


def sdram_size_in_bytes_for_filter_region(filters):
    """ generates the number of bytes a filter region requires
    
    :param filters: the filters needed to be stored in sdram
    :return: the size in bytes
    """
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
    """ calculates the sdram usage in bytes for a routing region
    
    :param n_keys: the number of incoming keys expected to be stored in this 
    region
    :return: the number of bytes this region requires 
    """
    return ((constants.ROUTING_N_ROUTES_SIZE + (
        constants.ROUTING_ENTRIES_PER_ROUTE * n_keys)) *
            constants.BYTE_TO_WORD_MULTIPLIER)


def locate_all_incoming_edges_of_type(operator_graph, app_vertex, input_type):
    """ locates all the incoming edges of type given.
    
    :param operator_graph: the application graph
    :param app_vertex: the vertex within the app graph to find incoming 
    connections
    :param input_type: the source port (input port) that we're looking for 
    connections of. 
    :return:  the list of connections with this input port that go into the 
    app vertex.
    """
    # create the filters and n_keys
    incoming_valid_edges = list()
    incoming_edges = operator_graph.get_edges_ending_at_vertex(app_vertex)
    for in_edge in incoming_edges:
        if in_edge.input_port.destination_input_port == input_type:
            incoming_valid_edges.append(in_edge)
    return incoming_valid_edges
