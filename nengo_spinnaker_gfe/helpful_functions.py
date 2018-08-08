from data_specification.enums import DataType
import numpy

from pacman.model.graphs.common import Slice


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


def expand_slice(vertex_slice, partition_index, n_dim):
    """ function used to expand a slice into ........
    
    :param vertex_slice: 
    :param partition_index: 
    :param n_dim: 
    :return: 
    """
    if partition_index is None:
        return slice(None)

    the_translated_slice = slice(vertex_slice.lo_atom, vertex_slice.hi_atom)

    thing = (
        tuple(slice(None) for _ in range(partition_index)) +
        (the_translated_slice,) +
        tuple(slice(None) for _ in range(partition_index + 1, n_dim)))
    return thing
