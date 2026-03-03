"""
Implements a :class:`~pypeit.par.parset.ParSet` subclass that collects the
keyword arguments of a function.

.. include:: ../include/links.rst
"""

import inspect

import numpy as np

from pypeit import PypeItError
from pypeit import utils
from pypeit.par.parset import ParSet


class FuncPar(ParSet):
    """
    A abstract :class:`~pypeit.par.parset.ParSet` subclass that collects the
    keyword arguments of a function.

    This class cannot be instantiated directly.

    .. note::
    
        This class currently does not:

            - capture positional arguments of the function or

            - determine the options, data types, or descriptions of the
              parameters.

    Parameters
    ----------
    **kwargs:
        The initial values for the keyword arguments.  If not provided, the
        default values from the function signature will be used.  An exception
        is raised if any of the provided keywords are *not* part of the function
        argument list.

    Attributes
    ----------
    module : str
        The module where the function is defined.
    name : str
        The name of the function.

    Raises
    ------
    PypeItError:
        Raised if any of the keywords in the ``restrict_to`` or ``kwargs`` list
        are not part of the function argument list.
    """

    func = None
    """
    The callable function whose keyword arguments are to be collected.
    """

    omitted_keys = None
    """
    Keyword arguments (provided as a list of strings) that should be omitted
    from the parameter set.  Any keyword in this list that is *not* part of the
    function signature is ignored.
    """

    kw_subset = None
    """
    A subset of keyword arguments (provided as a list of strings) for this
    parameter set.  If None, all keywords are included.  Any keyword that is
    *not* part of the function argument list will lead to an exception, which
    must be fixed at the coding level (i.e., this would not be user error).
    """

    def __init__(self, **kwargs):

        func_kwargs = self.valid_default_kwargs()

        # Finally, check that the provided kwargs are valid
        if len(kwargs) > 0:
            bad_keys = [key for key in kwargs.keys() if key not in func_kwargs.keys()]
            if len(bad_keys) > 0:
                # TODO: Use a warning instead?
                raise PypeItError(
                    f'{bad_keys} are not valid keyword arguments for this instance of '
                    f'{self.__class__.__name__} for function {self.func.__name__}.'
                )

        # Overwrite the defaults with the provided values
        user_kwargs = {k: v for k, v in func_kwargs.items()}
        for k, v in kwargs.items():
            user_kwargs[k] = v

        mod = inspect.getmodule(self.func)
        module_name = mod.__name__ if mod else None
        descr = [f'Parameter for {self.func.__name__} in {module_name}.']*len(func_kwargs)

        # Instantiate the base class.  There are no options, dtypes, or
        # descriptions.  TODO: We could potentially pull those from the
        # docstrings...
        super().__init__(
            list(func_kwargs.keys()),               # Restricted list of possible keywords
            values=list(user_kwargs.values()),      # Values provided by instantiation
            defaults=list(func_kwargs.values()),    # Default values from the function signature
            descr=descr,                            # Generic description
        )

        # Add the module and function names as attributes
        self.module = module_name
        self.name = self.func.__name__

    @classmethod
    def valid_default_kwargs(cls):

        if cls.func is None:
            raise NotImplementedError(
                f'CODING ERROR: The {cls.__name__} does not define the function from which to '
                'pull the keyword arguments!'
            )

        # Extract all the keyword arguments and their default values
        func_kwargs = utils.get_func_kwargs(cls.func)

        # Next apply the restricted list of keyword arguments    
        if cls.kw_subset is not None:
            indx = [k not in func_kwargs for k in cls.kw_subset]
            if any(indx):
                raise PypeItError(
                    f'CODING ERROR: {np.asarray(cls.kw_subset)[indx]} are not keyword arguments '
                    f'for {cls.func.__name__}!'
                )
            func_kwargs = {k: v for k, v in func_kwargs.items() if k in cls.kw_subset}

        # Next remove any keys that should generally be omitted for this
        # function
        if cls.omitted_keys is not None:
            for key in cls.omitted_keys:
                func_kwargs.pop(key, None)

        return func_kwargs
    
    @classmethod
    def from_dict(cls, cfg):
        k = np.array([*cfg.keys()])

        func_kwargs = cls.valid_default_kwargs()

        badkeys = np.array([pk not in func_kwargs.keys() for pk in cfg.keys()])
        if np.any(badkeys):
            raise ValueError(f'{k[badkeys]} not recognized key(s) for {cls.__name__}.')

        return cls(**{pk : cfg[pk] for pk in func_kwargs.keys() if pk in cfg.keys()})
