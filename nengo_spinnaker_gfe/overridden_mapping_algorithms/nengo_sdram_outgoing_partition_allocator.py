from nengo_spinnaker_gfe.graph_components.\
    constant_sdram_machine_partition import \
    ConstantSDRAMMachinePartition
from nengo_spinnaker_gfe.graph_components.\
    segmented_input_sdram_machine_partition import \
    SegmentedInputSDRAMMachinePartition
from nengo_spinnaker_gfe.graph_components.\
    segmented_spikes_sdram_machine_partition import \
    SegmentedSpikesSDRAMMachinePartition
from spinn_utilities.progress_bar import ProgressBar


class NengoSDRAMOutgoingPartitionAllocator(object):

    def __call__(self, machine_graph, transceiver, placements, app_id):

        progress_bar = ProgressBar(
            total_number_of_things_to_do=len(
                machine_graph.outgoing_edge_partitions),
            string_describing_what_being_progressed=(
                "Allocating SDRAM for SDRAM outgoing egde partitions"))

        for outgoing_edge_partition in \
                progress_bar.over(machine_graph.outgoing_edge_partitions):
            if (isinstance(outgoing_edge_partition,
                           SegmentedInputSDRAMMachinePartition) or
                isinstance(outgoing_edge_partition,
                           SegmentedSpikesSDRAMMachinePartition) or
                isinstance(outgoing_edge_partition,
                           ConstantSDRAMMachinePartition)):
                placement = placements.get_placement_of_vertex(
                    outgoing_edge_partition.pre_vertex)
                sdram_base_address = transceiver.malloc_sdram(
                    placement.x, placement.y,
                    outgoing_edge_partition.total_sdram_requirements(), app_id)

                outgoing_edge_partition.sdram_base_address = sdram_base_address
