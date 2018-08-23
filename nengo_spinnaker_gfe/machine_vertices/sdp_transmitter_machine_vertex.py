import threading
import numpy
from enum import Enum

from nengo_spinnaker_gfe import constants, helpful_functions
from nengo_spinnaker_gfe.abstracts.abstract_accepts_multicast_signals import \
    AbstractAcceptsMulticastSignals
from pacman.model.graphs.machine import MachineVertex
from pacman.model.resources import ResourceContainer, SDRAMResource, \
    IPtagResource
from spinn_front_end_common.abstract_models import AbstractHasAssociatedBinary
from spinn_front_end_common.abstract_models.impl import \
    MachineDataSpecableVertex
from spinn_front_end_common.interface.simulation import simulation_utilities
from spinn_front_end_common.utilities.utility_objs import ExecutableType
from spinn_front_end_common.utilities import constants as fec_constants
from spinn_utilities.overrides import overrides


class SDPTransmitterMachineVertex(
        MachineVertex, MachineDataSpecableVertex, AbstractHasAssociatedBinary,
        AbstractAcceptsMulticastSignals):

    __slots__ = [
        #
        "_size_in",
        #
        "_input_filters",
        #
        "_input_n_keys",
        #
        "_hostname",
        #
        "_port",
        #
        "_output_lock",
        #
        "_output"
    ]

    DATA_REGIONS = Enum(
        value="DATA_REGIONS",
        names=[('SYSTEM', 0),
               ('TRANSMITTER', 1),
               ('FILTER', 2),
               ('ROUTING', 3)])

    TRANSMITTER_REGION_ELEMENTS = 2
    TRANSMISSION_DELAY = 1
    IPTAG_TRAFFIC_IDENTIFIER = "SDP_RECEIVER_FEED"
    USE_IPTAG = False

    def __init__(self, size_in, input_filters, inputs_n_keys, hostname):
        MachineVertex.__init__(self)
        MachineDataSpecableVertex.__init__(self)
        AbstractHasAssociatedBinary.__init__(self)
        AbstractAcceptsMulticastSignals.__init__(self)
        self._size_in = size_in
        self._input_filters = input_filters
        self._input_n_keys = inputs_n_keys
        self._hostname = hostname
        self._output_lock = threading.Lock()
        self._output = numpy.zeros(self._size_in)

    @overrides(AbstractHasAssociatedBinary.get_binary_file_name)
    def get_binary_file_name(self):
        return "sdp_transmitter.aplx"

    @overrides(AbstractHasAssociatedBinary.get_binary_start_type)
    def get_binary_start_type(self):
        return ExecutableType.USES_SIMULATION_INTERFACE

    @property
    @overrides(MachineVertex.resources_required)
    def resources_required(self):
        return self.get_static_resources(
            self._input_filters, self._input_n_keys,
            self._hostname)

    @staticmethod
    def get_static_resources(input_filters, n_routing_keys, hostname):
        """ generates resource calculation so that it can eb called from 
        outside and not instantiated
        
        :param input_filters: the input filters going into this vertex
        :param n_routing_keys: the n keys from the input filters
        :param hostname: The hostname of the host machine we're running on
        :return: A resource container containing the resources used by this 
        vertex for those inputs. 
        """
        iptags = list()
        if SDPTransmitterMachineVertex.USE_IPTAG:
            iptags.append(
                IPtagResource(
                    ip_address=hostname, port=None, strip_sdp=False,
                    tag=None, traffic_identifier=(
                        SDPTransmitterMachineVertex.IPTAG_TRAFFIC_IDENTIFIER)))

        return ResourceContainer(
            sdram=SDRAMResource(
                fec_constants.SYSTEM_BYTES_REQUIREMENT +
                helpful_functions.sdram_size_in_bytes_for_filter_region(
                    input_filters) +
                helpful_functions.sdram_size_in_bytes_for_routing_region(
                    n_routing_keys) +
                SDPTransmitterMachineVertex._transmitter_region()),
            iptags=iptags)

    @staticmethod
    def _transmitter_region():
        """ determines the size of the transmitter region in bytes
        
        :return: the size in bytes
        :rtype: int
        """
        return (SDPTransmitterMachineVertex.TRANSMITTER_REGION_ELEMENTS *
                constants.BYTE_TO_WORD_MULTIPLIER)

    @overrides(AbstractAcceptsMulticastSignals.accepts_multicast_signals)
    def accepts_multicast_signals(self, transmission_params):
        return True

    @overrides(MachineDataSpecableVertex.generate_machine_data_specification)
    def generate_machine_data_specification(
            self, spec, placement, machine_graph, routing_info, iptags,
            reverse_iptags, machine_time_step, time_scale_factor):
        self._reserve_memory_regions(spec)

        # create system region
        spec.switch_write_focus(self.DATA_REGIONS.SYSTEM.value)
        spec.write_array(simulation_utilities.get_simulation_header_array(
            self.get_binary_file_name(), machine_time_step,
            time_scale_factor))

        # fill in filter region
        spec.switch_write_focus(self.DATA_REGIONS.FILTER.value)
        self._write_filter_region(spec)

        # fill in routing region
        spec.switch_write_focus(self.DATA_REGIONS.ROUTING.value)
        self._write_routing_region(spec)

        # fill in transmitter region
        spec.switch_write_focus(self.DATA_REGIONS.TRANSMITTER.value)
        spec.write_value(self._size_in)
        spec.write_value(self.TRANSMISSION_DELAY)
        spec.end_specification()

    def _write_filter_region(self, spec):
        pass

    def _write_routing_region(self, spec):
        pass

    def _reserve_memory_regions(self, spec):
        """ reserve the memory region sizes
        
        :param spec: the dsg spec
        :rtype: None
        """
        spec.reserve_memory_region(
            self.DATA_REGIONS.SYSTEM.value,
            fec_constants.SYSTEM_BYTES_REQUIREMENT, label="system region")
        spec.reserve_memory_region(
            self.DATA_REGIONS.FILTER.value,
            helpful_functions.sdram_size_in_bytes_for_filter_region(
                self._input_n_keys),
            label="filter region")
        spec.reserve_memory_region(
            self.DATA_REGIONS.ROUTING.value,
            helpful_functions.sdram_size_in_bytes_for_routing_region(
                self._input_n_keys),
            label="routing region")
        spec.reserve_memory_region(
            self.DATA_REGIONS.TRANSMITTER.value,
            self._transmitter_region(), label="transmitter region")

    @property
    def output(self):
        with self._output_lock:
            # return a copy of the array
            return self._output[:]

    @output.setter
    def output(self, new_value):
        with self._output_lock:
            self._output = new_value
