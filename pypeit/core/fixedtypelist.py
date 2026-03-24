"""
Implements an abstract base class for lists with a fixed type.
"""

from IPython import embed


class FixedTypeList(list):
    """
    An abstract subclass of :obj:`list` that enforces each element in the list
    have a given type.

    This class should not be instantiated itself; subclasses of this base class
    must define :attr:`list_type`.

    .. warning::

        This class *does not* override all the :obj:`list` dunder methods.  This
        means that some operations may return an instance of :obj:`list` instead
        of an instance of the original subclass.

    Parameters
    ----------
    iterable : iterable
        An iterable object with the initial contents of the list.  If None, the
        list is initially empty.

    Raises
    ------
    NotImplementedError
        Raised if :attr:`list_type` is not defined.
    """

    list_type = None
    """
    The type for elements in instances of this list.
    """

    def __init__(self, iterable=None):
        if self.list_type is None:
            raise NotImplementedError(
                f'Implementation of {self.__class__.__name__} does not define the type for the '
                'list elements and cannot be instantiated.'
            )
        if iterable is None:
            super().__init__()
            return
        for value in iterable:
            self._validate(value)
        super().__init__(iterable)
        
    def _validate(self, value):
        """
        Validate a new value to be added to the list.

        Parameters
        ----------
        value
            The new value to be included in the list.

        Raises
        ------
        TypeError
            Raised if ``value`` does not have the type defined by
            :attr:`list_type`.
        """
        # NOTE: This assumes list_type is defined!
        if not isinstance(value, self.list_type):
            raise TypeError(
                f'A {self.__class__.__name__} can only contain {self.list_type.__name__} objects; '
                f'cannot assign a {type(value).__name__} object.'
            )

    def append(self, value):
        """
        Overrides the base class method to include type checking.

        Parameters
        ----------
        value
            The new value to be included in the list.
        """
        self._validate(value)
        super().append(value)

    def extend(self, iterable):
        """
        Overrides the base class method to include type checking.

        Parameters
        ----------
        iterable
            The new set of values to be included in the list.
        """
        for value in iterable:
            self._validate(value)
        super().extend(iterable)

    def __iadd__(self, iterable):
        """
        Overrides the base class method to include type checking.

        Parameters
        ----------
        iterable
            The new set of values to be included in the list.
        """
        self.extend(iterable)
        return self
    
    def __add__(self, iterable):
        """
        Overrides the base class method to include type checking.

        Parameters
        ----------
        iterable
            The new set of values to be included in the list.
        """
        return self.__class__([s for s in self] + [s for s in iterable])

    def __setitem__(self, key, value):
        """
        Overrides the base class method to include type checking.

        Parameters
        ----------
        key : int, slice
            The item key
        value
            Values to set to a certain list element.  This should be a single
            object if ``key`` is an :obj:`int`; it should be an iterable
            otherwise.
        """
        # Handle slice assignments (e.g., my_list[1:3] = [4, 5])
        if isinstance(key, slice):
            for v in value:
                self._validate(v)
        # Handle single item assignments (e.g., my_list[0] = 1)
        else:
            self._validate(value)
        super().__setitem__(key, value)

    def __getitem__(self, key):
        """
        Overrides the base class method to ensure the returned object is an
        instance of this class, instead of a :obj:`list`, if more than one
        element is selected.
        """
        item = super().__getitem__(key)
        if isinstance(item, list):
            return self.__class__(item)
        return item
