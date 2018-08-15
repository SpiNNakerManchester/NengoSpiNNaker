from nengo_spinnaker_gfe.abstracts.abstract_transmits_multicast_signals import \
    TransmitsMulticastSignals
from pacman.model.graphs.machine import MachineVertex
from spinn_front_end_common.abstract_models import AbstractHasAssociatedBinary
from spinn_front_end_common.abstract_models.impl import \
    MachineDataSpecableVertex
from spinn_front_end_common.utilities.utility_objs import ExecutableType
from spinn_utilities.overrides import overrides

from nengo_spinnaker_gfe.abstracts.abstract_accepts_multicast_signals import \
    AcceptsMulticastSignals


class LIFMachineVertex(
        MachineVertex, MachineDataSpecableVertex, AbstractHasAssociatedBinary,
        AcceptsMulticastSignals, TransmitsMulticastSignals):

    __slots__ = [
        "_resources",
        "_neuron_slice",
        "_input_slice",
        "_output_slice",
        "_learnt_slice",
        "_n_profiler_samples"
    ]

    def __init__(self, vertex_index, neuron_slice, input_slice, output_slice,
                learnt_slice, resources, n_profiler_samples):
        MachineVertex.__init__(self)
        MachineDataSpecableVertex.__init__(self)
        AbstractHasAssociatedBinary.__init__(self)
        AcceptsMulticastSignals.__init__(self)
        self._resources = resources
        self._neuron_slice = neuron_slice
        self._input_slice = input_slice
        self._output_slice = output_slice
        self._learnt_slice = learnt_slice
        self._n_profiler_samples = n_profiler_samples

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

    @overrides(AcceptsMulticastSignals.accepts_multicast_signals)
    def accepts_multicast_signals(self, transmission_params):
        return True

    @overrides(TransmitsMulticastSignals.transmits_multicast_signals)
    def transmits_multicast_signals(self, transmission_params):
        return True

    @overrides(MachineDataSpecableVertex.generate_machine_data_specification)
    def generate_machine_data_specification(
            self, spec, placement, machine_graph, routing_info, iptags,
            reverse_iptags, machine_time_step, time_scale_factor):
        pass

    @overrides(AbstractHasAssociatedBinary.get_binary_start_type)
    def get_binary_start_type(self):
        return ExecutableType.USES_SIMULATION_INTERFACE

    @property
    @overrides(MachineVertex.resources_required)
    def resources_required(self):
        return self._resources

    @overrides(AbstractHasAssociatedBinary.get_binary_file_name)
    def get_binary_file_name(self):
        if self._n_profiler_samples > 0:
            return "lif_profiled.aplx"
        else:
            return "lif.aplx"
