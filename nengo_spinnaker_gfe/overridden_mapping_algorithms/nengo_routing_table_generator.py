from pacman.exceptions import PacmanAlreadyExistsException
from pacman.model.routing_tables import MulticastRoutingTable
from pacman.operations.routing_table_generators.\
    basic_routing_table_generator import BasicRoutingTableGenerator
from spinn_machine import MulticastRoutingEntry
from spinn_utilities.overrides import overrides


class NengoRoutingTableGenerator(BasicRoutingTableGenerator):

    def __init__(self):
        BasicRoutingTableGenerator.__init__(self)

    @overrides(BasicRoutingTableGenerator._create_routing_table)
    def _create_routing_table(self, chip, partitions_in_table, routing_infos):
        table = MulticastRoutingTable(chip.x, chip.y)
        for partition in partitions_in_table:
            r_info = routing_infos.get_routing_info_from_partition(partition)
            entry = partitions_in_table[partition]
            for key_and_mask in r_info.keys_and_masks:
                internal_entry = \
                    table.get_multicast_routing_entry_by_routing_entry_key(
                        routing_entry_key=key_and_mask.key_combo,
                        mask=key_and_mask.mask)
                this_entry = MulticastRoutingEntry(
                        routing_entry_key=key_and_mask.key_combo,
                        defaultable=entry.defaultable, mask=key_and_mask.mask,
                        link_ids=entry.link_ids,
                        processor_ids=entry.processor_ids)
                if internal_entry is None:
                    table.add_multicast_routing_entry(this_entry)
                else:
                    if this_entry == internal_entry:
                        pass
                    else:
                        raise PacmanAlreadyExistsException(
                            "Multicast_routing_entry", str(this_entry))
        return table
