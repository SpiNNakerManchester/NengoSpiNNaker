from enum import Enum

from nengo_spinnaker_gfe import constants, helpful_functions
from nengo_spinnaker_gfe.abstracts.abstract_accepts_multicast_signals import \
    AbstractAcceptsMulticastSignals
from nengo_spinnaker_gfe.abstracts.abstract_nengo_machine_vertex import \
    AbstractNengoMachineVertex
from nengo_spinnaker_gfe.abstracts.abstract_transmits_multicast_signals import \
    AbstractTransmitsMulticastSignals
from pacman.model.resources import ResourceContainer, SDRAMResource
from spinn_front_end_common.abstract_models import AbstractHasAssociatedBinary
from spinn_front_end_common.abstract_models.impl import \
    MachineDataSpecableVertex
from spinn_front_end_common.utilities.utility_objs import ExecutableType
from spinn_utilities.overrides import overrides


class InterposerMachineVertex(
        AbstractNengoMachineVertex, MachineDataSpecableVertex,
        AbstractHasAssociatedBinary, AbstractAcceptsMulticastSignals,
        AbstractTransmitsMulticastSignals):

    __slots__ = [
        "_size_in",
        "_output_slice",
        "_transform_data",
        "_n_keys",
        "_filter_keys",
        "_output_slices",
        "_machine_time_step",
        "_filters",
        "_transmission_params"
    ]

    """Portion of the rows of the transform assigned to a parallel filter
    group, represents the load assigned to a single processing core.
    """

    DATA_REGIONS = Enum(
        value="DATA_REGIONS",
        names=[('SYSTEM', 0),
               ('KEYS', 1),
               ('INPUT_FILTERS', 2),
               ('INPUT_ROUTING', 3),
               ('TRANSFORM', 4)])

    SYSTEM_DATA_ITEMS = 4

    def __init__(
            self, size_in, output_slice, transform_data, n_keys, filter_keys,
            output_slices, machine_time_step, filters, label, constraints):
        AbstractNengoMachineVertex.__init__(
            self, label=label, constraints=constraints)
        AbstractHasAssociatedBinary.__init__(self)
        AbstractAcceptsMulticastSignals.__init__(self)
        MachineDataSpecableVertex.__init__(self)
        AbstractTransmitsMulticastSignals.__init__(self)

        self._size_in = size_in
        self._output_slice = output_slice
        self._transform_data = transform_data
        self._n_keys = n_keys
        self._filter_keys = filter_keys
        self._output_slices = output_slices
        self._machine_time_step = machine_time_step
        self._filters = filters

        # Store which signal parameter slices we contain
        self._transmission_params = self._filter_transmission_params()

    def _filter_transmission_params(self):
        transmission_params = set()
        out_set = set(range(self._output_slice.start, self._output_slice.stop))
        for transmission_params, outs in self._output_slices:
            # If there is an intersection between the outs and the set of outs
            # we're responsible for then store transmission parameters.
            if out_set & outs:
                transmission_params.add(transmission_params)
        return out_set

    @overrides(MachineDataSpecableVertex.generate_machine_data_specification)
    def generate_machine_data_specification(self, spec, placement,
                                            machine_graph, routing_info,
                                            iptags,
                                            reverse_iptags,
                                            machine_time_step,
                                            time_scale_factor):
        raise Exception()

    @overrides(AbstractAcceptsMulticastSignals.accepts_multicast_signals)
    def accepts_multicast_signals(self, transmission_params):
        return transmission_params.projects_to(self._column_slice)

    @overrides(AbstractHasAssociatedBinary.get_binary_file_name)
    def get_binary_file_name(self):
        return "interposer.aplx"  # this was filter in mundy code

    @property
    @overrides(AbstractNengoMachineVertex.resources_required)
    def resources_required(self):
        return self.generate_static_resources(
            self._transform_data, self._n_keys, self._filters)

    @staticmethod
    def generate_static_resources(transform_data, n_keys, filters):
        sdram = (
            (InterposerMachineVertex.SYSTEM_DATA_ITEMS *
             constants.BYTE_TO_WORD_MULTIPLIER) +
            transform_data.nbytes + (constants.BYTES_PER_KEY * n_keys) +
            helpful_functions.sdram_size_in_bytes_for_filter_region(filters) +
            helpful_functions.sdram_size_in_bytes_for_routing_region(n_keys))
        return ResourceContainer(sdram=SDRAMResource(sdram))

    @overrides(AbstractHasAssociatedBinary.get_binary_start_type)
    def get_binary_start_type(self):
        return ExecutableType.USES_SIMULATION_INTERFACE

    @overrides(AbstractAcceptsMulticastSignals.accepts_multicast_signals)
    def accepts_multicast_signals(self, transmission_params):
        return transmission_params.projects_to(self._size_in)

    @overrides(AbstractTransmitsMulticastSignals.transmits_multicast_signals)
    def transmits_multicast_signals(self, transmission_params):
        return transmission_params in self._transmission_params
