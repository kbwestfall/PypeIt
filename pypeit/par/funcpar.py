"""
Implements a :class:`~pypeit.par.parset.ParSet` subclass that collects the
keyword arguments of a function.

.. include:: ../include/links.rst
"""

import inspect

from IPython import embed
import numpy as np

from pypeit import utils
from pypeit.par import parset


def _valid_default_kwargs(func, kw_subset, omitted_keys):
    """
    Use the signature of the provided function to extract its keyword arguments
    and their default values.

    This is a helper function for :mod:`~pypeit.par.funcpar`.

    Parameters
    ----------
    func : callable
        The callable function whose keyword arguments are to be collected.
    kw_subset : list
        A subset of keyword arguments (provided as a list of strings) to
        extract.  If None, all keywords are included.  Any keyword that is
        *not* part of the function argument list will lead to an exception,
        which must be fixed at the coding level (i.e., this would not be user
        error).
    omitted_keys : list
        Keyword arguments (provided as a list of strings) that should be
        excluded from the extracted set.  Any keyword in this list that is *not*
        part of the function signature is ignored.

    Returns
    -------
    dict
        The dictionary with the keyword parameters and their default values.
    """
    if func is None:
        raise NotImplementedError(
            f'CODING ERROR: Function from which to pull the keyword arguments is not defined!'
        )

    # Extract all the keyword arguments and their default values
    func_kwargs = utils.get_func_kwargs(func)

    # Next apply the restricted list of keyword arguments    
    if kw_subset is not None:
        indx = [k not in func_kwargs for k in kw_subset]
        if any(indx):
            raise KeyError(
                f'CODING ERROR: {np.asarray(kw_subset)[indx]} are not valid keyword arguments '
                f'for {func.__name__}!'
            )
        func_kwargs = {k: v for k, v in func_kwargs.items() if k in kw_subset}

    # Next remove any keys that should generally be omitted for this
    # function
    if omitted_keys is not None:
        for key in omitted_keys:
            func_kwargs.pop(key, None)

    return func_kwargs


def _get_module(func):
    """
    Get the parent module for the provided function.

    This is a helper function for :mod:`~pypeit.par.funcpar`.

    Parameters
    ----------
    func : callable
        The callable function to inspect.

    Returns
    -------
    str
        The string representation of the parent module of ``func``.  If
        :func:`inspect.getmodule` is unsuccessful, None is returned.
    """
    mod = inspect.getmodule(func)
    return mod.__name__ if mod else None


def _define_parameters(func, func_kwargs):
    """
    Define the parameter dictionary to use for a
    :class:`~pypeit.par.funcpar.FuncPar` subclass.

    Parameters
    ----------
    func : callable
        The callable function whose keyword arguments are to be collected into a
        parameter set.
    func_kwargs : dict
        A dictionary with the keyword parameters for ``func`` and their default
        values; see :func:`~pypeit.par.funcpar._valid_default_kwargs`.

    Returns
    -------
    dict
        The ``parameters`` dictionary to use for the function parameter set.
    """
    module_name = _get_module(func)
    descr = f'Parameter for {func.__name__} in {module_name}.'
    return {
        key : parset.set_parameter_definition(default=value, descr=descr)
        for key, value in func_kwargs.items()
    }


# NOTE: I'm not crazy about the idea of having to use a metaclass, but that's
# the only way that we can define the `parameters` dictionary dynamically.

class FuncParMetaClass(type):
    """
    The metaclass to use for :class:`~pypeit.par.funcpar.FuncPar` to enable the
    parameter dictionary to be constructed dynamically (instead of having to be
    hard-coded).
    """
    def __new__(mcs, name, bases, dic):
        # Create the class object
        cls = super().__new__(mcs, name, bases, dic)
        # Add the attribute to the newly created class object
        cls.parameters = _define_parameters(
            cls.func, _valid_default_kwargs(cls.func, cls.kw_subset, cls.omitted_keys)
        )
        return cls


class FuncPar(parset.ParSet):
    """
    A abstract :class:`~pypeit.par.parset.ParSet` subclass that collects the
    keyword arguments of a function.

    This class cannot be instantiated directly.  All subclasses must also use
    :class:`~pypeit.par.funcpar.FuncParMetaClass` for the class to be fully
    defined.

    Attributes listed below are in addition to those provided by the base class.

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
        argument list.  Note that subclasses can limit the keywords that are
        ingested as parameters; setting a value for any of the excluded
        parameters will raise an exception.

    Attributes
    ----------
    module : str
        The module where the function is defined.
    name : str
        The name of the function.
    """

    func = None
    """
    The callable function whose keyword arguments are to be collected.
    """

    kw_subset = None
    """
    A subset of keyword arguments (provided as a list of strings) for this
    parameter set.  If None, all keywords are included.  Any keyword that is
    *not* part of the function argument list will lead to an exception, which
    must be fixed at the coding level (i.e., this would not be user error).
    """

    omitted_keys = None
    """
    Keyword arguments (provided as a list of strings) that should be omitted
    from the parameter set.  Any keyword in this list that is *not* part of the
    function signature is ignored.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.module = _get_module(self.func)
        self.name = self.func.__name__

