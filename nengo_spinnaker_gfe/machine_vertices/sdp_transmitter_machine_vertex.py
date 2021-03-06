import struct
import threading

import numpy
from enum import Enum

from nengo_spinnaker_gfe import constants, helpful_functions
from nengo_spinnaker_gfe.abstracts.abstract_accepts_multicast_signals import \
    AbstractAcceptsMulticastSignals
from nengo_spinnaker_gfe.abstracts.abstract_nengo_machine_vertex import \
    AbstractNengoMachineVertex
from nengo_spinnaker_gfe.nengo_filters import filter_region_writer
from pacman.executor.injection_decorator import inject_items
from pacman.model.resources import ResourceContainer, SDRAMResource, \
    IPtagResource
from spinn_front_end_common.abstract_models import AbstractHasAssociatedBinary
from spinn_front_end_common.abstract_models.impl import \
    MachineDataSpecableVertex
from spinn_front_end_common.interface.provenance import \
    ProvidesProvenanceDataFromMachineImpl
from spinn_front_end_common.interface.simulation import simulation_utilities
from spinn_front_end_common.utilities import constants as fec_constants
from spinn_front_end_common.utilities.utility_objs import ExecutableType
from spinn_utilities.overrides import overrides


class SDPTransmitterMachineVertex(
        AbstractNengoMachineVertex, MachineDataSpecableVertex,
        AbstractHasAssociatedBinary, AbstractAcceptsMulticastSignals,
        ProvidesProvenanceDataFromMachineImpl):

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
               ('ROUTING', 3),
               ('PROVENANCE_DATA', 4)])

    TRANSMITTER_REGION_ELEMENTS = 4
    TRANSMISSION_DELAY = 1
    N_LOCAL_PROVENANCE_ITEMS = 0
    IPTAG_TRAFFIC_IDENTIFIER = "SDP_RECEIVER_FEED"
    _ONE_SHORT = struct.Struct("<H")
    _TWO_BYTES = struct.Struct("<BB")

    def __init__(self, size_in, input_filters, inputs_n_keys, hostname, label):
        AbstractNengoMachineVertex.__init__(self, label=label)
        MachineDataSpecableVertex.__init__(self)
        AbstractHasAssociatedBinary.__init__(self)
        AbstractAcceptsMulticastSignals.__init__(self)
        ProvidesProvenanceDataFromMachineImpl.__init__(self)

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
    @overrides(AbstractNengoMachineVertex.resources_required)
    def resources_required(self):
        return self.get_static_resources(
            self._input_filters, self._input_n_keys,
            self._hostname, self.N_LOCAL_PROVENANCE_ITEMS)

    @staticmethod
    def get_static_resources(
            input_filters, n_routing_keys, hostname, n_provenance_items):
        """ generates resource calculation so that it can eb called from 
        outside and not instantiated
        
        :param input_filters: the input filters going into this vertex
        :param n_routing_keys: the n keys from the input filters
        :param hostname: The hostname of the host machine we're running on
        :param n_provenance_items: n provenance data items 
        :return: A resource container containing the resources used by this 
        vertex for those inputs. 
        """
        iptags = list()
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
                ProvidesProvenanceDataFromMachineImpl.get_provenance_data_size(
                    n_provenance_items) +
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

    @inject_items({
        "machine_time_step_in_seconds": "MachineTimeStepInSeconds",
        "graph_mapper": "NengoGraphMapper",
        "nengo_graph": "NengoOperatorGraph"})
    @overrides(
        MachineDataSpecableVertex.generate_machine_data_specification,
        additional_arguments=[
            "machine_time_step_in_seconds", "graph_mapper", "nengo_graph"])
    def generate_machine_data_specification(
            self, spec, placement, machine_graph, routing_info, iptags,
            reverse_iptags, machine_time_step, time_scale_factor,
            machine_time_step_in_seconds, graph_mapper, nengo_graph):
        print "trasnmitter at {}".format(placement)
        self._reserve_memory_regions(spec)

        # create system region
        spec.switch_write_focus(self.DATA_REGIONS.SYSTEM.value)
        spec.write_array(simulation_utilities.get_simulation_header_array(
            self.get_binary_file_name(), machine_time_step,
            time_scale_factor))

        # fill in filter region
        spec.switch_write_focus(self.DATA_REGIONS.FILTER.value)
        filter_to_index_map = filter_region_writer.write_filter_region(
            spec, machine_time_step_in_seconds, self._input_slice,
            self._input_filters)

        # fill in routing region
        spec.switch_write_focus(self.DATA_REGIONS.ROUTING.value)
        helpful_functions.write_routing_region(
            spec, routing_info, machine_graph.get_edges_ending_at_vertex(self),
            filter_to_index_map, self._input_filters, graph_mapper,
            nengo_graph)

        # fill in transmitter region
        spec.switch_write_focus(self.DATA_REGIONS.TRANSMITTER.value)
        spec.write_value(self._size_in)
        spec.write_value(self.TRANSMISSION_DELAY)
        spec.write_value(iptags[0].tag)
        spec.write_value(self._ONE_SHORT.unpack(self._TWO_BYTES.pack(
            iptags[0].destination_y, iptags[0].destination_x))[0])

        spec.end_specification()

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
        self.reserve_provenance_data_region(spec)

    @property
    def output(self):
        with self._output_lock:
            # return a copy of the array
            return self._output[:]

    @output.setter
    def output(self, new_value):
        with self._output_lock:
            self._output = new_value

    @property
    @overrides(ProvidesProvenanceDataFromMachineImpl._n_additional_data_items)
    def _n_additional_data_items(self):
        return self.N_LOCAL_PROVENANCE_ITEMS

    @property
    @overrides(ProvidesProvenanceDataFromMachineImpl._provenance_region_id)
    def _provenance_region_id(self):
        return self.DATA_REGIONS.PROVENANCE_DATA.value
