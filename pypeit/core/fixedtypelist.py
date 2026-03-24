"""
Implements an abstract base class for lists with a fixed type.
"""

from IPython import embed

class FixedTypeList(list):
    """

    .. warning::

        This class *does not* override all the :obj:`list` dunder methods.  This
        means that some operations may return an instance of :obj:`list` instead
        of an instance of the original subclass.
    """
    list_type = None

    def __init__(self, iterable=None):
        if self.list_type is None:
            raise NotImplementedError(
                f'CODING ERROR: Implementation of {self.__class__.__name__} does not define the '
                'type for the list elements and cannot be instantiated.'
            )
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

    def __iadd__(self, iterable):
        self.extend(iterable)
        return self
    
    def __add__(self, iterable):
        return self.__class__([s for s in self] + [s for s in iterable])

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
