from pacman.model.constraints.key_allocator_constraints import \
    AbstractKeyAllocatorConstraint


class NengoKeyConstraints(AbstractKeyAllocatorConstraint):

    __slots__ = [
        "_constraints"
    ]

    def __init__(self, constraints):
        self._constraints = constraints

    @property
    def constraints(self):
        return self._constraints

    def set_value_for_field(self, field_id, field_value):
        for constraint in self._constraints:
            if constraint.field_id == field_id:
                constraint.field_value = field_value

    def get_value_for_field(self, field_id):
        for constraint in self._constraints:
            if constraint.field_id == field_id:
                return constraint.field_value
