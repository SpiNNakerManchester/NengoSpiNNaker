from enum import Enum

from nengo_spinnaker_gfe import constants
from nengo_spinnaker_gfe.abstracts.abstract_accepts_multicast_signals import \
    AbstractAcceptsMulticastSignals
from nengo_spinnaker_gfe.abstracts.abstract_nengo_machine_vertex import \
    AbstractNengoMachineVertex
from nengo_spinnaker_gfe.abstracts.abstract_transmits_multicast_signals import \
    AbstractTransmitsMulticastSignals

from spinn_front_end_common.abstract_models import AbstractHasAssociatedBinary
from spinn_front_end_common.abstract_models.impl import \
    MachineDataSpecableVertex
from spinn_front_end_common.interface.simulation import simulation_utilities
from spinn_front_end_common.utilities.utility_objs import ExecutableType
from spinn_front_end_common.utilities import constants as fec_constants

from spinn_utilities.overrides import overrides


class LIFMachineVertex(
        AbstractNengoMachineVertex, MachineDataSpecableVertex,
        AbstractHasAssociatedBinary, AbstractAcceptsMulticastSignals,
        AbstractTransmitsMulticastSignals):

    __slots__ = [
        "_resources",
        "_neuron_slice",
        "_input_slice",
        "_output_slice",
        "_learnt_slice",
        "_n_profiler_samples",
        "_ensemble_size_in",
        "_encoders_with_gain"
    ]

    DATA_REGIONS = Enum(
        value="DATA_REGIONS",
        names=[
            ('SYSTEM', 0),
            ('ENSEMBLE_PARAMS', 1),  # Ensemble, NEURON, POP length
            ('NEURON', 2),
            ('ENCODER', 3),
            ('BIAS', 4),
            ('GAIN', 5),
            ('DECODER', 6),
            ('LEARNT_DECODER', 7),
            ('KEYS', 8),  # KEYS and LEARNT KEYS
            ('FILTERS', 9),  # INPUT FILTERS, INHIB FILTERS, MODULATORY  FILTERS, LEARNT ENDCODER FILTERS
            ('ROUTING', 10),  # INPUT ROUTING, INHIB ROUTING, MOD ROUTING, LEANT ENCODER ROUTING
            ('PES', 11),
            ('VOJA', 12),
            ('FILTERED_ACTIVITY', 13),
            ('PROFILER', 14),
            ('RECORDING', 15)  # only one for SPike Voltage Encoder
           ])  # 26

    ENSEMBLE_PARAMS_ITEMS = 17
    NEURON_PARAMS_ITEMS = 0
    POP_LENGTH_ITEMS = 0

    def __init__(self, vertex_index, neuron_slice, input_slice, output_slice,
                learnt_slice, resources, n_profiler_samples, encoders_with_gain,
                 ensemble_size_in, label):
        AbstractNengoMachineVertex.__init__(self, label=label)
        MachineDataSpecableVertex.__init__(self)
        AbstractHasAssociatedBinary.__init__(self)
        AbstractAcceptsMulticastSignals.__init__(self)
        AbstractTransmitsMulticastSignals.__init__(self)
        self._resources = resources
        self._neuron_slice = neuron_slice
        self._input_slice = input_slice
        self._output_slice = output_slice
        self._learnt_slice = learnt_slice
        self._n_profiler_samples = n_profiler_samples
        self._ensemble_size_in = ensemble_size_in
        self._encoders_with_gain = encoders_with_gain

    @property
    def neuron_slice(self):
        return self._neuron_slice

    @property
    def input_slice(self):
        return self._input_slice

    @property
    def output_slice(self):
        return self._output_slice

    @property
    def learnt_slice(self):
        return self._learnt_slice

    @overrides(AbstractAcceptsMulticastSignals.accepts_multicast_signals)
    def accepts_multicast_signals(self, transmission_params):
        return True

    @overrides(AbstractTransmitsMulticastSignals.transmits_multicast_signals)
    def transmits_multicast_signals(self, transmission_params):
        return True

    @overrides(MachineDataSpecableVertex.generate_machine_data_specification)
    def generate_machine_data_specification(
            self, spec, placement, machine_graph, routing_info, iptags,
            reverse_iptags, machine_time_step, time_scale_factor):
        self._allocate_memory_regions(spec)
        spec.switch_write_focus(self.DATA_REGIONS.SYSTEM.value)
        spec.write_array(simulation_utilities.get_simulation_header_array(
            self.get_binary_file_name(), machine_time_step,
            time_scale_factor))
        self._write_params(spec)

        raise Exception()

    def _write_params(self, spec):
        spec.write_value(self._neuron_slice.n_atoms)
        spec.write_value(self._ensemble_size_in)
        spec.write_value(self._encoders_with_gain.shape[1])


    def _allocate_memory_regions(self, spec):
        spec.reserve_memory_region(
            self.DATA_REGIONS.SYSTEM.value,
            fec_constants.SYSTEM_BYTES_REQUIREMENT, label="system region")
        spec.reserve_memory_region(
            self.DATA_REGIONS.ENSEMBLE_PARAMS.value,
            (self.ENSEMBLE_PARAMS_ITEMS + self.NEURON_PARAMS_ITEMS +
             self.POP_LENGTH_ITEMS) * constants.BYTE_TO_WORD_MULTIPLIER)



    @overrides(AbstractHasAssociatedBinary.get_binary_start_type)
    def get_binary_start_type(self):
        return ExecutableType.USES_SIMULATION_INTERFACE

    @property
    @overrides(AbstractNengoMachineVertex.resources_required)
    def resources_required(self):
        return self._resources

    @overrides(AbstractHasAssociatedBinary.get_binary_file_name)
    def get_binary_file_name(self):
        if self._n_profiler_samples > 0:
            return "lif_profiled.aplx"
        else:
            return "lif.aplx"
