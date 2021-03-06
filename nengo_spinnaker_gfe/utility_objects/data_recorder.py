from __future__ import division
from collections import OrderedDict
import logging
import math
import numpy
from six import iteritems, raise_from
from six.moves import range, xrange

from data_specification.enums import DataType
from spinn_front_end_common.utilities.exceptions import ConfigurationException
from spinn_front_end_common.utilities import globals_variables
from spinn_utilities.progress_bar import ProgressBar
from spynnaker.pyNN.models.neural_properties import NeuronParameter

from nengo_spinnaker_gfe import constants

logger = logging.getLogger(__name__)

SPIKES = "spikes"


class DataRecorder(object):

    __slots__ = [
        "_sampling_rates",
        "_indexes",
        "_n_neurons",
        "_matrix_variable_sizes",
        "_n_recorded_regions",
        "_machine_vertex_to_slice"
    ]

    N_BYTES_FOR_TIMESTAMP = 4
    N_WORDS_FOR_TIMESTAMP = 1
    N_BYTES_PER_INDEX = 1
    N_BYTES_PER_SIZE = 4
    N_BYTES_PER_RATE = 4
    N_BYTES_PER_NUM_NEURONS = 4
    N_CPU_CYCLES_PER_NEURON = 8
    N_BYTES_PER_POINTER = 4

    N_BITS_IN_A_BYTE = 8

    N_DTCM_CYCLES_PER_BYTE_PR_POINTER = 2
    N_DTCM_CYCLES_PER_OUTSTANDING_RECORDING = 4

    # the overflow bit is used to cover all over neurons that spiked but are
    # not being recorded, it allows easier usage within the c code.
    N_BITS_FOR_OVERFLOW = 1

    ALL_FLAG = "all"

    MAX_RATE = 2 ** 32 - 1  # To allow a unit32_t to be used to store the rate

    def __init__(self, allowed_variables, n_neurons, matrix_variable_sizes):
        self._sampling_rates = OrderedDict()
        self._indexes = dict()
        self._n_neurons = n_neurons
        self._matrix_variable_sizes = matrix_variable_sizes
        self._n_recorded_regions = len(allowed_variables)
        self._machine_vertex_to_slice = dict()
        for variable in allowed_variables:
            self._sampling_rates[variable] = 0
            self._indexes[variable] = None

    def _count_recording_per_slice(self, variable, vertex_slice):
        if self._sampling_rates[variable] == 0:
            return 0
        if self._indexes[variable] is None:
            return vertex_slice.n_atoms
        return sum(vertex_slice.lo_atom <= index <= vertex_slice.hi_atom
                   for index in self._indexes[variable])

    def add_machine_vertex_mapping(self, variable, machine_vertex, atom_slice):
        self._machine_vertex_to_slice[(variable, machine_vertex)] = atom_slice

    def _bits_recording(self, variable, vertex_slice):
        if self._sampling_rates[variable] == 0:
            return []
        if self._indexes[variable] is None:
            return range(vertex_slice.lo_atom, vertex_slice.hi_atom+1)
        recording = []
        indexes = self._indexes[variable]
        for index in xrange(vertex_slice.lo_atom, vertex_slice.hi_atom+1):
            if index in indexes:
                recording.append(index)
        return recording

    def get_neuron_sampling_interval(self, variable):
        """ Return the current sampling interval for this variable

        :param variable: name of the variable
        :return: Sampling interval in micro seconds
        """
        step = (globals_variables.get_simulator().machine_time_step /
                constants.CONVERT_MILLISECONDS_TO_SECONDS)
        return self._sampling_rates[variable] * step

    def get_sorted_matrix_data(
            self,  label, buffer_manager, region, placements, graph_mapper,
            application_vertex, variable, n_machine_time_steps):
        data, indices, sampling_interval = self.get_matrix_data(
            label, buffer_manager, region, placements, graph_mapper,
            application_vertex, variable, n_machine_time_steps)
        return data[numpy.argsort(indices)], indices, sampling_interval

    def get_matrix_data(
            self, label, buffer_manager, region, placements, graph_mapper,
            application_vertex, variable, n_machine_time_steps):
        """ Read a uint32 mapped to time and neuron IDs from the SpiNNaker\
            machine.

        :param label: vertex label
        :param buffer_manager: the manager for buffered data
        :param region: the DSG region ID used for this data
        :param placements: the placements object
        :param graph_mapper: \
            the mapping between application and machine vertices
        :param application_vertex:
        :param variable: PyNN name for the variable (V, gsy_inh etc.)
        :type variable: str
        :param n_machine_time_steps:
        :return:
        """
        if variable == SPIKES:
            msg = "Variable {} is not supported use get_spikes".format(SPIKES)
            raise ConfigurationException(msg)
        vertices = graph_mapper.get_machine_vertices(application_vertex)
        progress = ProgressBar(
            vertices, "Getting {} for {}".format(variable, label))
        sampling_rate = self._sampling_rates[variable]
        expected_rows = int(math.ceil(n_machine_time_steps / sampling_rate))
        missing_str = ""
        data = None
        indexes = []
        for vertex in progress.over(vertices):
            placement = placements.get_placement_of_vertex(vertex)
            vertex_slice = self._machine_vertex_to_slice[(variable, vertex)]
            if vertex_slice.n_atoms == 0:
                continue
            indexes.extend(self._bits_recording(variable, vertex_slice))
            # for buffering output info is taken form the buffer manager
            recording_region_data_pointer, missing_data = \
                buffer_manager.get_data_for_vertex(placement, region)
            record_raw = recording_region_data_pointer.read_all()
            record_length = len(record_raw)

            row_length = (
                self.N_BYTES_FOR_TIMESTAMP + (
                    vertex_slice.n_atoms *
                    self._matrix_variable_sizes[variable]))

            # There is one column for time and one for each neuron recording
            n_rows = record_length // row_length

            # Converts bytes to ints and make a matrix
            format_string = "<i{}".format(self._matrix_variable_sizes[variable])
            record = (
                numpy.asarray(record_raw, dtype="uint8").view(
                    dtype=format_string)).reshape(
                (n_rows, (vertex_slice.n_atoms + 1)))

            # Check if you have the expected data
            if not missing_data and n_rows == expected_rows:
                # Just cut the timestamps off to get the fragment
                fragment = (record[:, 1:] / float(DataType.S1615.scale))
            else:
                missing_str += "({}, {}, {}); ".format(
                    placement.x, placement.y, placement.p)
                # Start the fragment for this slice empty
                fragment = numpy.empty((expected_rows, vertex_slice.n_atoms))
                for i in xrange(0, expected_rows):
                    time = i * sampling_rate
                    # Check if there is data for this timestep
                    local_indexes = numpy.where(record[:, 0] == time)
                    if len(local_indexes[0]) > 0:
                        # Set row to data for that timestep
                        fragment[i] = (record[local_indexes[0], 1:] /
                                       float(DataType.S1615.scale))[i]
                    else:
                        # Set row to nan
                        fragment[i] = numpy.full(
                            vertex_slice.n_atoms, numpy.nan)
            if data is None:
                data = fragment
            else:
                # Add the slice fragment on axis 1 which is IDs/channel_index
                data = numpy.append(data, fragment, axis=1)
        if len(missing_str) > 0:
            logger.warn(
                "Population {} is missing recorded data in region {} from the"
                " following cores: {}".format(label, region, missing_str))
        sampling_interval = self.get_neuron_sampling_interval(variable)

        return data, indexes, sampling_interval

    def get_bools(
            self, label, buffer_manager, region, placements, graph_mapper,
            application_vertex, machine_time_step):
        """ 
        
        :param label: 
        :param buffer_manager: 
        :param region: 
        :param placements: 
        :param graph_mapper: 
        :param application_vertex: 
        :param machine_time_step: 
        :return: 
        """

        recorded_times = list()
        recorded_ids = list()
        ms_per_tick = \
            machine_time_step / constants.CONVERT_MILLISECONDS_TO_SECONDS

        vertices = graph_mapper.get_machine_vertices(application_vertex)
        missing_str = ""
        progress = ProgressBar(vertices, "Getting spikes for {}".format(label))
        for vertex in progress.over(vertices):
            placement = placements.get_placement_of_vertex(vertex)
            vertex_slice = self._machine_vertex_to_slice[(SPIKES, vertex)]

            if self._indexes[SPIKES] is None:
                things_recording = vertex_slice.n_atoms
            else:
                things_recording = sum(
                    (vertex_slice.lo_atom <= index <= vertex_slice.hi_atom)
                    for index in self._indexes[SPIKES])
                if things_recording == 0:
                    continue
                if things_recording < vertex_slice.n_atoms:
                    # For spikes the overflow position is also returned
                    things_recording += self.N_BITS_FOR_OVERFLOW
            # Read the spikes
            n_words = int(math.ceil(
                things_recording / constants.WORD_TO_BIT_CONVERSION))
            n_bytes = n_words * constants.BYTE_TO_WORD_MULTIPLIER
            n_words_with_timestamp = n_words + self.N_WORDS_FOR_TIMESTAMP

            # for buffering output info is taken form the buffer manager
            recording_region_data_pointer, data_missing = \
                buffer_manager.get_data_for_vertex(placement, region)
            if data_missing:
                missing_str += "({}, {}, {}); ".format(
                    placement.x, placement.y, placement.p)
            record_raw = recording_region_data_pointer.read_all()

            raw_data = (
                numpy.asarray(record_raw, dtype="uint8").view(
                    dtype="<i4")).reshape([-1, n_words_with_timestamp])

            if len(raw_data) > 0:
                record_time = raw_data[:, 0] * float(ms_per_tick)
                bools = raw_data[:, 1:].byteswap().view("uint8")
                bits = numpy.fliplr(numpy.unpackbits(bools).reshape(
                    (-1, constants.WORD_TO_BIT_CONVERSION))).reshape(
                    (-1, n_bytes * self.N_BITS_IN_A_BYTE))
                time_indices, local_indices = numpy.where(bits == 1)
                if self._indexes[SPIKES] is None:
                    indices = local_indices + vertex_slice.lo_atom
                    times = record_time[time_indices].reshape((-1))
                    recorded_ids.extend(indices)
                    recorded_times.extend(times)
                else:
                    things_recording = self._bits_recording(
                        SPIKES, vertex_slice)
                    n_things = len(things_recording)
                    for time_indice, local in zip(time_indices, local_indices):
                        if local < n_things:
                            recorded_ids.append(things_recording[local])
                            recorded_times.append(record_time[time_indice])

        if len(missing_str) > 0:
            logger.warn(
                "Population {} is missing bool data in region {} from the"
                " following cores: {}".format(label, region, missing_str))

        if len(recorded_ids) == 0:
            return numpy.zeros((0, 2), dtype="float")

        result = numpy.column_stack((recorded_ids, recorded_times))
        return result[numpy.lexsort((recorded_times, recorded_ids))]

    def get_recordable_variables(self):
        return self._sampling_rates.keys()

    def is_recording(self, variable):
        try:
            return self._sampling_rates[variable] > 0
        except KeyError as e:
            msg = "Variable {} is not supported. Supported variables are {}" \
                  "".format(variable, self.get_recordable_variables())
            raise_from(ConfigurationException(msg), e)

    @property
    def recording_variables(self):
        results = list()
        for key in self._sampling_rates:
            if self.is_recording(key):
                results.append(key)
        return results

    def _compute_rate(self, sampling_interval):
        """ Convert a sampling interval into a rate. \
            Remember, machine time step is in nanoseconds

        :param sampling_interval: interval between samples in microseconds
        :return: rate
        """
        if sampling_interval is None:
            return 1

        step = (
            globals_variables.get_simulator().machine_time_step /
            constants.CONVERT_MILLISECONDS_TO_SECONDS)
        rate = int(sampling_interval / step)
        if sampling_interval != rate * step:
            msg = "sampling_interval {} is not an an integer multiple of the "\
                  "simulation timestep {}".format(sampling_interval, step)
            raise ConfigurationException(msg)
        if rate > self.MAX_RATE:
            msg = "sampling_interval {} higher than max allowed which is {}" \
                  "".format(sampling_interval, step * self.MAX_RATE)
            raise ConfigurationException(msg)
        return rate

    def check_indexes(self, indexes):
        if indexes is None:
            return

        if len(indexes) == 0:
            raise ConfigurationException("Empty indexes list")

        found = False
        warning = None
        for index in indexes:
            if index < 0:
                raise ConfigurationException(
                    "Negative indexes are not supported")
            elif index >= self._n_neurons:
                warning = "Ignoring indexes greater than population size."
            else:
                found = True
            if warning is not None:
                logger.warning(warning)
        if not found:
            raise ConfigurationException(
                "All indexes larger than population size")

    def _turn_off_recording(self, variable, sampling_interval, remove_indexes):
        if self._sampling_rates[variable] == 0:
            # Already off so ignore other parameters
            return

        if remove_indexes is None:
            # turning all off so ignoring sampling interval
            self._sampling_rates[variable] = 0
            self._indexes[variable] = None
            return

        # No good reason to specify_interval when turning off
        if sampling_interval is not None:
            rate = self._compute_rate(sampling_interval)
            # But if they do make sure it is the same as before
            if rate != self._sampling_rates[variable]:
                raise ConfigurationException(
                    "Illegal sampling_interval parameter while turning "
                    "off recording")

        if self._indexes[variable] is None:
            # start with all indexes
            self._indexes[variable] = range(self._n_neurons)

        # remove the indexes not recording
        self._indexes[variable] = \
            [index for index in self._indexes[variable]
                if index not in remove_indexes]

        # Check is at least one index still recording
        if len(self._indexes[variable]) == 0:
            self._sampling_rates[variable] = 0
            self._indexes[variable] = None

    def _check_complete_overwrite(self, variable, indexes):
        if indexes is None:
            # overwriting all OK!
            return
        if self._indexes[variable] is None:
            if set(set(range(self._n_neurons))).issubset(set(indexes)):
                # overwriting all previous so OK!
                return
        else:
            if set(self._indexes[variable]).issubset(set(indexes)):
                # overwriting all previous so OK!
                return
        raise ConfigurationException(
            "Current implementation does not support multiple "
            "sampling_intervals for {} on one population.".format(
                variable))

    def _turn_on_recording(self, variable, sampling_interval, indexes):

        rate = self._compute_rate(sampling_interval)
        if self._sampling_rates[variable] == 0:
            # Previously not recording so OK
            self._sampling_rates[variable] = rate
        elif rate != self._sampling_rates[variable]:
            self._check_complete_overwrite(variable, indexes)
        # else rate not changed so no action

        if indexes is None:
            # previous recording indexes does not matter as now all (None)
            self._indexes[variable] = None
        else:
            # make sure indexes is not a generator like range
            indexes = list(indexes)
            self.check_indexes(indexes)
            if self._indexes[variable] is None:
                # just use the new indexes
                self._indexes[variable] = indexes
            else:
                # merge the two indexes
                self._indexes[variable] = \
                    list(set(self._indexes[variable] + indexes))
                self._indexes[variable].sort()

    def set_recording(self, variable, new_state, sampling_interval=None,
                      indexes=None):
        if variable == self.ALL_FLAG:
            for key in self._sampling_rates.keys():
                self.set_recording(key, new_state, sampling_interval, indexes)
        elif variable in self._sampling_rates:
            if new_state:
                self._turn_on_recording(variable, sampling_interval, indexes)
            else:
                self._turn_off_recording(variable, sampling_interval, indexes)
        else:
            raise ConfigurationException(
                "Variable {} is not supported".format(variable))

    def get_buffered_sdram_per_record(self, variable, vertex_slice):
        """ Return the SDRAM used per record

        :param variable:
        :param vertex_slice:
        :return:
        """
        n_neurons = self._count_recording_per_slice(variable, vertex_slice)
        if n_neurons == 0:
            return 0
        if variable == SPIKES:
            if n_neurons < vertex_slice.n_atoms:
                # Indexing is used rather than gating to determine recording
                # Non recoding neurons write to an extra slot
                n_neurons += 1
            out_spike_words = int(
                math.ceil(n_neurons / constants.WORD_TO_BIT_CONVERSION))
            out_spike_bytes = (
                out_spike_words * constants.BYTE_TO_WORD_MULTIPLIER)
            return self.N_BYTES_FOR_TIMESTAMP + out_spike_bytes
        else:
            return (self.N_BYTES_FOR_TIMESTAMP +
                    n_neurons * self._matrix_variable_sizes[variable])

    def get_buffered_sdram_per_timestep(self, variable, vertex_slice):
        """ Return the SDRAM used per timestep.

        In the case where sampling is used it returns the average\
        for recording and none recording based on the recording rate

        :param variable:
        :param vertex_slice:
        :return:
        """
        rate = self._sampling_rates[variable]
        if rate == 0:
            return 0

        data_size = self.get_buffered_sdram_per_record(variable, vertex_slice)
        if rate == 1:
            return data_size
        else:
            return data_size // rate

    def get_sampling_overflow_sdram(self, vertex_slice):
        """ Get the extra SDRAM that should be reserved if using per_timestep

        This is the extra that must be reserved if per_timestep is an average\
        rather than fixed for every timestep.

        When sampling the average * time_steps may not be quite enough.\
        This returns the extra space in the worst case\
        where time_steps is a multiple of sampling rate + 1,\
        and recording is done in the first and last time_step

        :param vertex_slice:
        :return: Highest possible overflow needed
        """
        overflow = 0
        for variable, rate in iteritems(self._sampling_rates):
            # If rate is 0 no recording so no overflow
            # If rate is 1 there is no overflow as average is exact
            if rate > 1:
                data_size = self.get_buffered_sdram_per_record(
                    variable,  vertex_slice)
                overflow += data_size // rate * (rate - 1)
        return overflow

    def get_buffered_sdram(self, variable, vertex_slice, n_machine_time_steps):
        """ Return the SDRAM used per timestep

        In the case where sampling is used it returns the average\
        for recording and none recording based on the recording rate

        :param variable:
        :param vertex_slice:
        :param n_machine_time_steps:
        :return:
        """
        rate = self._sampling_rates[variable]
        if rate == 0:
            return 0
        data_size = self.get_buffered_sdram_per_record(variable, vertex_slice)
        records = n_machine_time_steps // rate
        if n_machine_time_steps % rate > 0:
            records = records + 1
        return data_size * records

    def get_sdram_usage_in_bytes(self, vertex_slice):
        n_words_for_n_neurons = int(math.ceil(
            (vertex_slice.n_atoms * self.N_BYTES_PER_INDEX) /
            constants.BYTE_TO_WORD_MULTIPLIER))
        n_bytes_for_n_neurons = (
            n_words_for_n_neurons * constants.BYTE_TO_WORD_MULTIPLIER)
        return (self.N_BYTES_PER_RATE + self.N_BYTES_PER_NUM_NEURONS +
                n_bytes_for_n_neurons) * self._n_recorded_regions

    def get_dtcm_usage_in_bytes(self, vertex_slice):
        # *_rate + n_neurons_recording_* + *_indexes
        usage = self.get_sdram_usage_in_bytes(vertex_slice)
        # *_count + *_increment
        usage += (self._n_recorded_regions * self.N_BYTES_PER_POINTER *
                  self.N_DTCM_CYCLES_PER_BYTE_PR_POINTER)
        # out_spikes, *_values
        for variable in self._sampling_rates:
            if variable == SPIKES:
                out_spike_words = int(
                    math.ceil(vertex_slice.n_atoms /
                              constants.WORD_TO_BIT_CONVERSION))
                out_spike_bytes = (
                    out_spike_words * constants.BYTE_TO_WORD_MULTIPLIER)
                usage += self.N_BYTES_FOR_TIMESTAMP + out_spike_bytes
            else:
                usage += (
                    self.N_BYTES_FOR_TIMESTAMP +
                    (vertex_slice.n_atoms *
                     self._matrix_variable_sizes[variable]))
        # *_size
        usage += self._n_recorded_regions * self.N_BYTES_PER_SIZE

        # n_recordings_outstanding
        usage += (
            constants.BYTE_TO_WORD_MULTIPLIER *
            self.N_DTCM_CYCLES_PER_OUTSTANDING_RECORDING)
        return usage

    def get_n_cpu_cycles(self, n_neurons):
        return n_neurons * self.N_CPU_CYCLES_PER_NEURON * \
                len(self.recording_variables)

    def get_initialization_data_array(self, vertex_slice):
        data = list()
        n_words_for_n_neurons = int(math.ceil(
            vertex_slice.n_atoms / constants.BYTE_TO_WORD_MULTIPLIER))
        n_bytes_for_n_neurons = (
            n_words_for_n_neurons * constants.BYTE_TO_WORD_MULTIPLIER)
        for variable in self._sampling_rates:
            rate = self._sampling_rates[variable]
            n_recording = self._count_recording_per_slice(
                variable, vertex_slice)
            data.append(numpy.array([rate, n_recording], dtype="uint32"))
            if rate == 0:
                data.append(numpy.zeros(n_words_for_n_neurons, dtype="uint32"))
            elif self._indexes[variable] is None:
                data.append(numpy.arange(
                    n_bytes_for_n_neurons, dtype="uint8").view("uint32"))
            else:
                indexes = self._indexes[variable]
                local_index = 0
                local_indexes = list()
                for index in xrange(n_bytes_for_n_neurons):
                    if index + vertex_slice.lo_atom in indexes:
                        local_indexes.append(local_index)
                        local_index += 1
                    else:
                        # write to one beyond recording range
                        local_indexes.append(n_recording)
                data.append(
                    numpy.array(local_indexes, dtype="uint8").view("uint32"))
        return numpy.concatenate(data)

    def get_global_parameters(self, vertex_slice):
        params = []
        for variable in self._sampling_rates:
            params.append(NeuronParameter(
                self._sampling_rates[variable], DataType.UINT32))
        for variable in self._sampling_rates:
            n_recording = self._count_recording_per_slice(
                variable, vertex_slice)
            params.append(NeuronParameter(n_recording, DataType.UINT8))
        return params
