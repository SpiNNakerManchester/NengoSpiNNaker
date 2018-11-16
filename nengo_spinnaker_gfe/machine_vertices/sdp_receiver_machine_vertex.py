import numpy
from enum import Enum
import random

from nengo_spinnaker_gfe import constants
from nengo_spinnaker_gfe import helpful_functions
from nengo_spinnaker_gfe.abstracts.abstract_nengo_machine_vertex import \
    AbstractNengoMachineVertex
from nengo_spinnaker_gfe.abstracts.abstract_transmits_multicast_signals import \
    AbstractTransmitsMulticastSignals
from pacman.executor.injection_decorator import inject_items
from pacman.model.resources import ResourceContainer, SDRAMResource, \
    DTCMResource, CPUCyclesPerTickResource, ReverseIPtagResource
from spinn_front_end_common.abstract_models import \
    AbstractHasAssociatedBinary, AbstractProvidesNKeysForPartition, \
    AbstractRecordable
from spinn_front_end_common.abstract_models.impl import \
    MachineDataSpecableVertex
from spinn_front_end_common.interface.provenance import \
    ProvidesProvenanceDataFromMachineImpl
from spinn_front_end_common.interface.simulation import simulation_utilities
from spinn_front_end_common.utilities import constants as fec_constants
from spinn_front_end_common.utilities.utility_objs import ExecutableType
from spinn_utilities.overrides import overrides
from spinnman.messages.sdp import SDPMessage, SDPHeader, SDPFlag


class SDPReceiverMachineVertex(
        AbstractNengoMachineVertex, MachineDataSpecableVertex,
        AbstractHasAssociatedBinary, AbstractProvidesNKeysForPartition,
        AbstractTransmitsMulticastSignals, AbstractRecordable,
        ProvidesProvenanceDataFromMachineImpl):

    __slots__ = [
        # keys to transmit with i think
        '_n_keys',

        # the outgoing partition this sdp receiver is managing for the host
        "_managing_app_outgoing_partition",
    ]

    DATA_REGIONS = Enum(
        value="DATA_REGIONS",
        names=[('SYSTEM', 0),
               ('SDP_PORT', 1),
               ('KEYS', 2),
               ('MC_TRANSMISSION_PARAMS', 3),
               ('PROVENANCE_DATA', 4)])

    BYTES_PER_FIELD = 4
    SDP_PORT_SIZE = 1
    USE_REVERSE_IPTAGS = False
    SDP_PORT = 6
    MC_TRANSMISSION_REGION_ITEMS = 2
    N_LOCAL_PROVENANCE_ITEMS = 0

    # TODO THIS LIMIT IS BECAUSE THE C CODE ASSUMES 1 SDP Message contains
    # the next timer ticks worth of changes. future could be modded to remove
    # this limitation.
    MAX_N_KEYS_SUPPORTED = 64
    TRANSFORM_SLICE_OUT = False

    def __init__(self, outgoing_partition, label):
        AbstractNengoMachineVertex.__init__(self, label)
        MachineDataSpecableVertex.__init__(self)
        AbstractHasAssociatedBinary.__init__(self)
        AbstractTransmitsMulticastSignals.__init__(self)
        AbstractProvidesNKeysForPartition.__init__(self)
        AbstractRecordable.__init__(self)
        ProvidesProvenanceDataFromMachineImpl.__init__(self)

        # TODO WHY DO WE PARTITION OVER OUTGOING PARTITIONS!!!
        self._managing_app_outgoing_partition = outgoing_partition

        transform = self._managing_app_outgoing_partition.identifier\
            .transmission_parameter.full_transform(
                slice_out=self.TRANSFORM_SLICE_OUT)
        self._n_keys = transform.shape[0]

        # Check n keys size
        if self._n_keys > self.MAX_N_KEYS_SUPPORTED:
            raise NotImplementedError(
                "Connection is too wide to transmit to SpiNNaker. "
                "Consider breaking the connection up or making the "
                "originating node a function of time Node.")

    @overrides(AbstractRecordable.is_recording)
    def is_recording(self):
        return True

    @overrides(AbstractProvidesNKeysForPartition.get_n_keys_for_partition)
    def get_n_keys_for_partition(self, partition, graph_mapper):
        if (partition.identifier !=
                self._managing_app_outgoing_partition.identifier):
            raise Exception("don't recognise this partition")
        else:
            return self._n_keys

    @property
    @overrides(AbstractNengoMachineVertex.resources_required)
    def resources_required(self):
        return self.get_static_resources(
            self._n_keys, self.N_LOCAL_PROVENANCE_ITEMS)

    @staticmethod
    def get_static_resources(keys, local_provenance_items):
        if SDPReceiverMachineVertex.USE_REVERSE_IPTAGS:
            reverse_ip_tags = [ReverseIPtagResource()]
        else:
            reverse_ip_tags = None
        return ResourceContainer(
            sdram=SDRAMResource(
                fec_constants.SYSTEM_BYTES_REQUIREMENT +
                (SDPReceiverMachineVertex.SDP_PORT_SIZE *
                 constants.BYTE_TO_WORD_MULTIPLIER) +
                (SDPReceiverMachineVertex.MC_TRANSMISSION_REGION_ITEMS *
                 constants.BYTE_TO_WORD_MULTIPLIER) +
                ProvidesProvenanceDataFromMachineImpl.get_provenance_data_size(
                    local_provenance_items) +
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
            transmission_params == (
                self._managing_app_outgoing_partition.
                identifier.transmission_parameter))

    @inject_items({"graph_mapper": "NengoGraphMapper"})
    @overrides(
        MachineDataSpecableVertex.generate_machine_data_specification,
        additional_arguments=["graph_mapper"])
    def generate_machine_data_specification(
            self, spec, placement, machine_graph, routing_info, iptags,
            reverse_iptags, machine_time_step, time_scale_factor, graph_mapper):
        self._reserve_memory_regions(spec)
        spec.switch_write_focus(self.DATA_REGIONS.SYSTEM.value)
        spec.write_array(simulation_utilities.get_simulation_header_array(
            self.get_binary_file_name(), machine_time_step,
            time_scale_factor))
        spec.switch_write_focus(self.DATA_REGIONS.KEYS.value)
        self._write_keys_region(spec, routing_info, graph_mapper, machine_graph)
        spec.switch_write_focus(self.DATA_REGIONS.SDP_PORT.value)
        spec.write_value(self.SDP_PORT)
        spec.switch_write_focus(self.DATA_REGIONS.MC_TRANSMISSION_PARAMS.value)
        self._write_mc_transmission_params(
            spec, graph_mapper, machine_time_step, time_scale_factor)
        spec.end_specification()

    def _write_mc_transmission_params(
            self, spec, graph_mapper, machine_time_step, time_scale_factor):
        # Write the random back off value
        app_vertex = graph_mapper.get_application_vertex(self)
        spec.write_value(random.randint(0, min(
            app_vertex.n_sdp_receiver_machine_vertices,
            constants.MICROSECONDS_PER_SECOND // machine_time_step)))

        # avoid a possible division by zero / small number (which may
        # result in a value that doesn't fit in a uint32) by only
        # setting time_between_spikes if spikes_per_timestep is > 1
        time_between_spikes = 0.0
        if self._n_keys > 1:
            time_between_spikes = (
                (machine_time_step * time_scale_factor) /
                (self._n_keys * 2.0))
        spec.write_value(data=int(time_between_spikes))

    def _write_keys_region(
            self, spec, routing_info, graph_mapper, machine_graph):
        app_edge = self._managing_app_outgoing_partition.edges.peek()
        machine_edge = graph_mapper.get_machine_edges(app_edge).peek()
        machine_outgoing_partition = \
            machine_graph.get_outgoing_partition_for_edge(machine_edge)
        partition_routing_info = routing_info.get_routing_info_from_partition(
            machine_outgoing_partition)
        spec.write_value(self._n_keys)
        for key in partition_routing_info.get_keys(n_keys=self._n_keys):
            spec.write_value(key)

    def _reserve_memory_regions(self, spec):
        spec.reserve_memory_region(
            self.DATA_REGIONS.SYSTEM.value,
            fec_constants.SYSTEM_BYTES_REQUIREMENT, label="system region")
        spec.reserve_memory_region(
            self.DATA_REGIONS.SDP_PORT.value,
            (self.SDP_PORT_SIZE * constants.BYTE_TO_WORD_MULTIPLIER),
            label="n_keys region")
        spec.reserve_memory_region(
            self.DATA_REGIONS.KEYS.value,
            self._calculate_sdram_for_keys(self._n_keys),
            label="keys region")
        spec.reserve_memory_region(
            self.DATA_REGIONS.MC_TRANSMISSION_PARAMS.value,
            (self.MC_TRANSMISSION_REGION_ITEMS *
             constants.BYTE_TO_WORD_MULTIPLIER),
            label="mc_transmission data")
        self.reserve_provenance_data_region(spec)

    def send_output_to_spinnaker(self, value, placement, transceiver):

        # Apply the pre-slice, the connection function and the transform.
        c_value = value[(
            self._managing_app_outgoing_partition.identifier.
            transmission_parameter.pre_slice)]

        # locate required transforms and functions
        partition_transmission_function = \
            self._managing_app_outgoing_partition.identifier\
                .transmission_parameter.parameter_function
        partition_transmission_transform = \
            self._managing_app_outgoing_partition.identifier\
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
                destination_port=constants.SDP_PORTS.SDP_RECEIVER.value,
                destination_cpu=placement.p, destination_chip_x=placement.x,
                destination_chip_y=placement.y,
                flags=SDPFlag.REPLY_NOT_EXPECTED),
            data=bytes(data.data))
        transceiver.send_sdp_message(packet)

    @property
    @overrides(ProvidesProvenanceDataFromMachineImpl._n_additional_data_items)
    def _n_additional_data_items(self):
        return self.N_LOCAL_PROVENANCE_ITEMS

    @property
    @overrides(ProvidesProvenanceDataFromMachineImpl._provenance_region_id)
    def _provenance_region_id(self):
        return self.DATA_REGIONS.PROVENANCE_DATA.value
