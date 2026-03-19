"""
Implements an abstract base class for lists with a fixed type.
"""

class FixedTypeList(list):
    """
    """
    list_type = None

    def __init__(self, iterable=None):
        if iterable is None:
            super().__init__()
            return
        for value in iterable:
            self._validate(value)
        super().__init__(iterable)
        
    def _validate(self, value):
        # NOTE: This assumes list_type is defined!
        if not isinstance(value, self.list_type):
            raise TypeError(
                f'A {self.__class__.__name__} can only contain {self.list_type.__name__} objects; '
                f'cannot assign a {type(value).__name__} object.'
            )

    def append(self, value):
        self._validate(value)
        super().append(value)

    def extend(self, iterable):
        for value in iterable:
            self._validate(value)
        super().extend(iterable)

    def __setitem__(self, key, value):
        # Handle slice assignments (e.g., my_list[1:3] = [4, 5])
        if isinstance(key, slice):
            for v in value:
                self._validate(v)
        # Handle single item assignments (e.g., my_list[0] = 1)
        else:
            self._validate(value)
        super().__setitem__(key, value)

    def __getitem__(self, key):
        item = super().__getitem__(key)
        if isinstance(item, list):
            return self.__class__(item)
        return item
