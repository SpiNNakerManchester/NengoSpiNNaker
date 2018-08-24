from pacman.model.constraints.key_allocator_constraints import \
    AbstractKeyAllocatorConstraint


class NengoKeyConstraint(AbstractKeyAllocatorConstraint):

    __slots__ = [
        '_field_id',
        '_field_value',
        "_tags",
        "_start_at",
        "_length"
    ]

    def __init__(self, field_id, field_value, tags, start_at, length):
        AbstractKeyAllocatorConstraint.__init__(self)
        self._field_id = field_id
        self._field_value = field_value
        self._tags = tags
        self._start_at = start_at
        self._length = length

    @property
    def start_at(self):
        return self._start_at

    @property
    def tags(self):
        return self._tags

    @property
    def field_id(self):
        return self._field_id

    @property
    def field_value(self):
        return self._field_value

    @property
    def length(self):
        return self._length

    @field_value.setter
    def field_value(self, new_value):
        self._field_value = new_value
