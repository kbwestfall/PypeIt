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
    def __init__(self, func, restrict_to=None, **kwargs):

        # Extract all the keyword arguments and their default values
        func_kwargs = utils.get_func_kwargs(func)

        # Apply the restricted list of keyword arguments    
        if restrict_to is not None:
            _rt = np.asarray(restrict_to)
            indx = [k not in func_kwargs for k in _rt]
            if any(indx):
                raise PypeItError(f'{_rt[indx]} are not keyword arguments for {func.__name__}!')
            func_kwargs = {k: v for k, v in func_kwargs.items() if k in restrict_to}

        # Check that the provided kwargs are valid
        if len(kwargs) > 0:
            bad_kwargs = [k for k in kwargs if k not in func_kwargs]
            if len(bad_kwargs) > 0:
                raise PypeItError(f'{bad_kwargs} are not keyword arguments for {func.__name__}!')

        # Overwrite the defaults
        user_values = {k: v for k, v in func_kwargs.items()}
        for k, v in kwargs.items():
            user_values[k] = v

        # Instantiate the base class.  There are no options, dtypes, or
        # descriptions.  TODO: We could potentially pull those from the
        # docstrings...
        super().__init__(
            list(func_kwargs.keys()),               # Restricted list of possible keywords
            values=list(user_values.values()),      # Values provided by instantiation
            defaults=list(func_kwargs.values()),    # Default values from the function signature
        )

        # Add the module and function names as attributes
        mod = inspect.getmodule(func)
        self.module = mod.__name__ if mod else None
        self.name = func.__name__


class DifferentialEvolutionPar(FuncPar):
    """
    Parameters used by :func:`scipy.optimize.differential_evolution`.

    .. warning::

        The ``seed`` parameter is being deprecated by
        :func:`scipy.optimize.differential_evolution`.  Use ``rng`` instead.
        This class will raise an exception if ``seed`` is provided as a
        ``kwarg``.
    """
    def __init__(self, **kwargs):
        if 'seed' in kwargs:
            raise PypeItError(
                'The seed parameter is being deprecated by differential_evolution.  Use rng '
                'instead.'
            )
        func_kwargs = utils.get_func_kwargs(optimize.differential_evolution)
        func_kwargs.pop('seed', None)   # Remove seed from the possible keywords
        super().__init__(
            optimize.differential_evolution, restrict_to=list(func_kwargs.keys()), **kwargs
        )


class DJSRejectPar(FuncPar):
    """
    Parameters used by :func:`~pypeit.core.pydl.djs_reject`.

    .. warning::

        This does *not* include the ``outmask``, ``inmask``, or ``invvar``
        parameters.

    """
    def __init__(self, **kwargs):
        keys_to_omit = ['outmask', 'inmask', 'invvar']
        found_keys = [key for key in keys_to_omit if key in kwargs]
        if len(found_keys) > 0:
            raise PypeItError(
                f'Keywords {found_keys} are not allowed in the DJSRejectPar class.'
            )
        func_kwargs = utils.get_func_kwargs(djs_reject)
        for key in keys_to_omit:
            func_kwargs.pop(key, None)   
        super().__init__(djs_reject, restrict_to=list(func_kwargs.keys()), **kwargs)
