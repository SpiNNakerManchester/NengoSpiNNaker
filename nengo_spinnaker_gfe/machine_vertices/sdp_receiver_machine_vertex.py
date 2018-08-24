import numpy
from data_specification.enums import DataType
from enum import Enum

from nengo_spinnaker_gfe.abstracts.abstract_transmits_multicast_signals import \
    AbstractTransmitsMulticastSignals
from nengo_spinnaker_gfe.graph_components.nengo_machine_vertex import \
    NengoMachineVertex
from pacman.model.resources import ResourceContainer, SDRAMResource, \
    DTCMResource, CPUCyclesPerTickResource, ReverseIPtagResource
from spinn_front_end_common.abstract_models import \
    AbstractHasAssociatedBinary, AbstractProvidesNKeysForPartition
from spinn_front_end_common.abstract_models.impl import \
    MachineDataSpecableVertex
from spinn_front_end_common.interface.simulation import simulation_utilities
from spinn_front_end_common.utilities import constants as fec_constants
from spinn_front_end_common.utilities.utility_objs import ExecutableType
from spinn_utilities.overrides import overrides
from nengo_spinnaker_gfe import constants
from nengo_spinnaker_gfe import helpful_functions
from spinnman.messages.sdp import SDPMessage, SDPHeader


class SDPReceiverMachineVertex(
        NengoMachineVertex, MachineDataSpecableVertex,
        AbstractHasAssociatedBinary, AbstractProvidesNKeysForPartition,
        AbstractTransmitsMulticastSignals):

    __slots__ = [
        # keys to transmit with i think
        '_n_keys',

        #
        "_managing_outgoing_partition",

    ]

    DATA_REGIONS = Enum(
        value="DATA_REGIONS",
        names=[('SYSTEM', 0),
               ('N_KEYS', 1),
               ('KEYS', 2)])
    N_KEYS_REGION_ITEMS = 2
    BYTES_PER_FIELD = 4
    USE_REVERSE_IPTAGS = False

    # TODO THIS LIMIT IS BECAUSE THE C CODE ASSUMES 1 SDP Message contains
    # the next timer ticks worth of changes. future could be modded to remove
    # this limitation.
    MAX_N_KEYS_SUPPORTED = 64
    TRANSFORM_SLICE_OUT = False

    def __init__(self, outgoing_partition):
        NengoMachineVertex.__init__(self)
        MachineDataSpecableVertex.__init__(self)
        AbstractHasAssociatedBinary.__init__(self)
        AbstractTransmitsMulticastSignals.__init__(self)
        AbstractProvidesNKeysForPartition.__init__(self)

        # TODO WHY DO WE PARTITION OVER OUTGOING PARTITIONS!!!
        self._managing_outgoing_partition = outgoing_partition

        transform = self._managing_outgoing_partition.identifier\
            .transmission_parameter.full_transform(
                slice_out=self.TRANSFORM_SLICE_OUT)
        self._n_keys = transform.shape[0]

        # Check n keys size
        if self._n_keys > self.MAX_N_KEYS_SUPPORTED:
            raise NotImplementedError(
                "Connection is too wide to transmit to SpiNNaker. "
                "Consider breaking the connection up or making the "
                "originating node a function of time Node.")

    @overrides(AbstractProvidesNKeysForPartition.get_n_keys_for_partition)
    def get_n_keys_for_partition(self, partition, graph_mapper):
        if partition.identifier != self._managing_outgoing_partition.identifier:
            raise Exception("don't recognise this partition")
        else:
            return self._n_keys

    @property
    @overrides(NengoMachineVertex.resources_required)
    def resources_required(self):
        return self.get_static_resources(self._n_keys)

    @staticmethod
    def get_static_resources(keys):
        if SDPReceiverMachineVertex.USE_REVERSE_IPTAGS:
            reverse_ip_tags = [ReverseIPtagResource()]
        else:
            reverse_ip_tags = None
        return ResourceContainer(
            sdram=SDRAMResource(
                fec_constants.SYSTEM_BYTES_REQUIREMENT +
                (SDPReceiverMachineVertex.N_KEYS_REGION_ITEMS *
                 constants.BYTE_TO_WORD_MULTIPLIER) +
                SDPReceiverMachineVertex._calculate_sdram_for_keys(keys)),
            dtcm=DTCMResource(0),
            cpu_cycles=CPUCyclesPerTickResource(0),
            reverse_iptags=reverse_ip_tags)

    @staticmethod
    def _calculate_sdram_for_keys(keys):
        return SDPReceiverMachineVertex.BYTES_PER_FIELD * keys

    @overrides(AbstractHasAssociatedBinary.get_binary_file_name)
    def get_binary_file_name(self):
        return "sdp_receiver.aplx"

    @overrides(AbstractHasAssociatedBinary.get_binary_start_type)
    def get_binary_start_type(self):
        return ExecutableType.USES_SIMULATION_INTERFACE

    @overrides(AbstractTransmitsMulticastSignals.transmits_multicast_signals)
    def transmits_multicast_signals(self, transmission_params):
        return (
            transmission_params ==
            self._managing_outgoing_partition.identifier.transmission_parameter)

    @overrides(MachineDataSpecableVertex.generate_machine_data_specification)
    def generate_machine_data_specification(
            self, spec, placement, machine_graph, routing_info,
            iptags, reverse_iptags, machine_time_step, time_scale_factor):
        self._reserve_memory_regions(spec)
        spec.switch_write_focus(self.DATA_REGIONS.SYSTEM.value)
        spec.write_array(simulation_utilities.get_simulation_header_array(
            self.get_binary_file_name(), machine_time_step,
            time_scale_factor))
        spec.switch_write_focus(self.DATA_REGIONS.N_KEYS.value)
        spec.write_value(self._n_keys, DataType.UINT32)
        spec.switch_write_focus(self.DATA_REGIONS.KEYS.value)
        self._write_keys_region(spec, routing_info)
        spec.end_specification()

    def _write_keys_region(self, spec, routing_info):
        partition_routing_info = routing_info.get_routing_info_from_partition(
            self._managing_outgoing_partition)
        if partition_routing_info is None:
            spec.write_value(1)
            spec.write_value(0)
        else:
            spec.write_value(0)
            for key in partition_routing_info.get_keys():
                spec.write_value(key)

    def _reserve_memory_regions(self, spec):
        spec.reserve_memory_region(
            self.DATA_REGIONS.SYSTEM.value,
            fec_constants.SYSTEM_BYTES_REQUIREMENT, label="system region")
        spec.reserve_memory_region(
            self.DATA_REGIONS.N_KEYS.value,
            (self.N_KEYS_REGION_ITEMS * constants.BYTE_TO_WORD_MULTIPLIER),
            label="n_keys region")
        spec.reserve_memory_region(
            self.DATA_REGIONS.KEYS.value,
            self._calculate_sdram_for_keys(self._n_keys),
            label="keys region")

    def send_output_to_spinnaker(self, value, placement, transceiver):
        # Apply the pre-slice, the connection function and the transform.
        c_value = value[self._pre_slice]

        # locate required transforms and functions
        partition_transmission_function = \
            self._managing_outgoing_partition.identifier\
                .transmission_parameter.function
        partition_transmission_transform = \
            self._managing_outgoing_partition.identifier\
                .transmission_parameter.full_transform(slice_out=False)

        # execute function if required
        if partition_transmission_function is not None:
            c_value = partition_transmission_function(c_value)

        # do transform
        c_value = numpy.dot(partition_transmission_transform, c_value)

        # create SCP packet
        # c_value is converted to S16.15
        data = helpful_functions.convert_numpy_array_to_s16_15(c_value)
        packet = SDPMessage(
            sdp_header=SDPHeader(
                destination_port=constants.SDP_PORTS.SDP_RECEIVER,
                destination_cpu=placement.p, destination_chip_x=placement.x,
                destination_chip_y=placement.y),
            data=data)
        transceiver.send_sdp_message(packet)
