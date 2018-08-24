from nengo_spinnaker_gfe import constants
from nengo_spinnaker_gfe.constraints.nengo_key_constraint import \
    NengoKeyConstraint
from nengo_spinnaker_gfe.constraints.nengo_key_constraints import \
    NengoKeyConstraints
from pacman.model.graphs.machine import MachineVertex
from spinn_front_end_common.abstract_models import \
    AbstractProvidesOutgoingPartitionConstraints


class NengoMachineVertex(
        MachineVertex, AbstractProvidesOutgoingPartitionConstraints):

    def __init__(self, label=None, constraints=None):
        MachineVertex.__init__(self, label=label, constraints=constraints)
        AbstractProvidesOutgoingPartitionConstraints.__init__(self)

    def get_outgoing_partition_constraints(self, partition):
        return [NengoKeyConstraints(
            constraints=[
                NengoKeyConstraint(
                    field_id=constants.USER_FIELD_ID,
                    field_value=constants.USER_FIELDS.NENGO.value,
                    tags=[constants.ROUTING_TAG, constants.FILTER_ROUTING_TAG],
                    start_at=None, length=None),
                NengoKeyConstraint(
                    field_id=constants.CONNECTION_FIELD_ID,
                    field_value=None, start_at=None, length=None,
                    tags=[constants.ROUTING_TAG, constants.FILTER_ROUTING_TAG]),
                NengoKeyConstraint(
                    field_id=constants.CLUSTER_FIELD_ID, field_value=None,
                    tags=[constants.ROUTING_TAG], start_at=None, length=None),
                NengoKeyConstraint(
                    field_id=constants.INDEX_FIELD_ID, field_value=None,
                    tags=None, start_at=constants.INDEX_FIELD_START_POINT,
                    length=None)])]
