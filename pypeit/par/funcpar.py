"""
Implements a :class:`~pypeit.par.parset.ParSet` subclass that collects the
keyword arguments of a function.

.. include:: ../include/links.rst
"""

import inspect

import numpy as np
from scipy import optimize

from pypeit import PypeItError
from pypeit import utils
from pypeit.core.pydl import djs_reject
from pypeit.par.parset import ParSet


class FuncPar(ParSet):
    """
    A :class:`~pypeit.par.parset.ParSet` subclass that collects the keyword
    arguments of a function.

    .. note::
    
        This class currently does not:

            - capture positional arguments of the function or

            - determine the options, data types, or descriptions of the
              parameters.

    Parameters
    ----------
    func : callable
        The function whose keyword arguments are to be collected.
    restrict_to : array-like, optional
        A restricted list of keywords to include.  If None, all keywords are
        included.  If provided, only the keywords provided will be included; an
        exception is raised if any of the provided keywords that are *not* part
        of the function argument list.
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

    omitted_keys = None
    """
    Keyword arguments that are omitted from the parameter set.
    """

    def __init__(self, func, restrict_to=None, **kwargs):

        # Extract all the keyword arguments and their default values
        func_kwargs = utils.get_func_kwargs(func)

        # Next apply the restricted list of keyword arguments    
        if restrict_to is not None:
            _rt = np.asarray(restrict_to)
            indx = [k not in func_kwargs for k in _rt]
            if any(indx):
                # TODO: Use a warning instead?
                raise PypeItError(f'{_rt[indx]} are not keyword arguments for {func.__name__}!')
            func_kwargs = {k: v for k, v in func_kwargs.items() if k in restrict_to}

        # Next remove any keys that should generally be omitted for this
        # function
        if self.omitted_keys is not None:
            for key in self.omitted_keys:
                func_kwargs.pop(key, None)

        # Finally, check that the provided kwargs are valid
        if len(kwargs) > 0:
            bad_keys = [key for key in kwargs.keys() if key not in func_kwargs.keys()]
            if len(bad_keys) > 0:
                # TODO: Use a warning instead?
                raise PypeItError(
                    f'{bad_keys} are not valid keyword arguments for this instance of '
                    f'{self.__class__.__name__} for function {func.__name__}.'
                )

        # Overwrite the defaults with the provided values
        user_kwargs = {k: v for k, v in func_kwargs.items()}
        for k, v in kwargs.items():
            user_kwargs[k] = v

        mod = inspect.getmodule(func)
        module_name = mod.__name__ if mod else None
        descr = [f'Parameter for {func.__name__} in {module_name}.']*len(func_kwargs)

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
        self.name = func.__name__


class DifferentialEvolutionPar(FuncPar):
    """
    Parameters used by :func:`scipy.optimize.differential_evolution`.
    """

    omitted_keys = ['args', 'seed']
    """
    Differential evolution parameters omit the arguments for the fitting
    function.  Also, the seed parameter is being deprecated by scipy; use rng
    instead.
    """

    def __init__(self, restrict_to=None, **kwargs):
        super().__init__(optimize.differential_evolution, restrict_to=restrict_to, **kwargs)


class DJSRejectPar(FuncPar):
    """
    Parameters used by :func:`~pypeit.core.pydl.djs_reject`.
    """

    omitted_keys = ['outmask', 'inmask', 'invvar']
    """
    Omit the data-specific parameters for :func:`~pypeit.core.pydl.djs_reject`.
    """

    def __init__(self, restrict_to=None, **kwargs):
        super().__init__(djs_reject, restrict_to=restrict_to, **kwargs)
