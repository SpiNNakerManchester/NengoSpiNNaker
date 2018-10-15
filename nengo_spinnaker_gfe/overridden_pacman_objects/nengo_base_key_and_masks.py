from nengo_spinnaker_gfe import constants
from pacman.model.routing_info import BaseKeyAndMask
from spinn_utilities.overrides import overrides

import numpy


class NengoBaseKeysAndMasks(BaseKeyAndMask):

    __slots__ = [
        "_key_space",
        "_neuron_mask"
    ]

    def __init__(self, key_space):
        self._key_space = key_space
        BaseKeyAndMask.__init__(
            self, base_key=key_space.get_value(constants.ROUTING_TAG),
            mask=key_space.get_mask(constants.ROUTING_TAG))
        self._neuron_mask = key_space.get_mask(constants.INDEX_FIELD_ID)

    @property
    def neuron_mask(self):
        """ returns the mask for locating neuron id / dimension
        
        :return: the neuron mask
        """
        return self._neuron_mask

    @overrides(BaseKeyAndMask.get_keys)
    def get_keys(self, key_array=None, offset=0, n_keys=None):
        """ Get the ordered list of keys that the combination allows

        :param key_array: \
            Optional array into which the returned keys will be placed
        :type key_array: array-like of int
        :param offset: \
            Optional offset into the array at which to start placing keys
        :type offset: int
        :param n_keys: \
            Optional limit on the number of keys returned. If less than this\
            number of keys are available, only the keys available will be added
        :type n_keys: int
        :return: A tuple of an array of keys and the number of keys added to\
            the array
        :rtype: tuple(array-like of int, int)
        """
        if self._key_space.user != constants.USER_FIELDS.NENGO.value:
            return BaseKeyAndMask.get_keys(key_array, offset, n_keys)
        else:
            # Get the position of the zeros in the mask - assume 32-bits
            unwrapped_mask = numpy.unpackbits(
                numpy.asarray([self._mask], dtype=">u4").view(dtype="uint8"))
            zeros = numpy.where(unwrapped_mask == 0)[0]

            # If there are no zeros, there is only one key in the range, so
            # return that
            if len(zeros) == 0:
                if key_array is None:
                    key_array = numpy.zeros(1, dtype=">u4")
                key_array[offset] = self._base_key
                return key_array, 1

            # We now know how many values there are - 2^len(zeros)
            max_n_keys = 2 ** len(zeros)
            if key_array is not None and len(key_array) < max_n_keys:
                max_n_keys = len(key_array)
            if n_keys is None or n_keys > max_n_keys:
                n_keys = max_n_keys
            if key_array is None:
                key_array = numpy.zeros(n_keys, dtype=">u4")

            # get keys
            keys = list()
            for index in range(0, n_keys):
                args = {constants.INDEX_FIELD_ID: index}
                keys.append(self._key_space(**args))

            # for each key, create its key with the idea of a neuron ID being
            # continuous and live at an offset position from the bottom of
            # the key
            for key, index in enumerate(keys):
                key = numpy.unpackbits(numpy.asarray([key], dtype=">u4").view(
                    dtype="uint8"))
                key = numpy.copy(key)
                unwrapped_value = numpy.unpackbits(
                    numpy.asarray([index], dtype=">u4")
                         .view(dtype="uint8"))[-len(zeros):]
                key[zeros] = unwrapped_value
                key_array[index + offset] = \
                    numpy.packbits(key).view(dtype=">u4")[0].item()
            return key_array, n_keys
