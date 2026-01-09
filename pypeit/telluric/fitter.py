"""
Module for fitting a source + telluric model to an observed spectrum.
"""

from IPython import embed

import numpy as np
from scipy import optimize
from scipy import special
from scipy.stats import qmc

from pypeit import log
from pypeit import PypeItError
from pypeit import utils
from pypeit.core import spectrum


class ObservedSourceModel:

    def __init__(self, src_model, tell_model):
        """
        Class to perform the source + telluric model fit to an observed spectrum.

        Parameters
        ----------
        src_model : :class`~pypeit.telluric.source.AdjustedSpectrumModel`
            The class to use when modeling the source spectrum.  The object
            cannot be ``None`` and the model must be fully initialized (i.e.,
            ``src_model.sample()`` should not fail).
        tell_model : :class:`~pypeit.telluric.model.TelluricModel`
            The class to use when modeling the telluric spectrum.  The object
            cannot be ``None``, the model must be fully initialized (i.e.,
            ``tell_model.sample()`` should not fail), and the wavelength array
            of the model must match the source spectrum model.
        """
        self.src_model = src_model
        self.tell_model = tell_model

        if not np.allclose(self.src_model.wave, self.tell_model.wave):
            raise PypeItError('Source and telluric model wavelength arrays do not match.')

        # Kept during fitting
        self.obs_spec = None

    @property
    def wave(self):
        """
        Wavelength array of the source + telluric model.
        """
        return self.src_model.wave

    @property
    def npar(self):
        """
        Total number of parameters in the source + telluric model.
        """
        return self.src_model.npar + self.tell_model.npar

    def par_guess(self, obs_spec):
        """
        Provide an initial guess for all the model parameters.

        Parameters
        ----------
        obs_spec : :class:`~pypeit.core.spectrum.Spectrum`
            Spectrum to be fit.

        Returns
        -------
        `numpy.ndarray`_
            Guess parameters
        """
        # Guess the telluric parameters first.  Note that the current telluric
        # model classes do not use the flux vector of the observed spectrum.
        # They only use the wavelength vector to guess the spectral resolution.
        tell_par = self.tell_model.par_guess(obs_spec.wave)

        # Use the guess parameters to generate an initial telluric model
        tell_wave, tell_spec, tell_gpm = self.tell_model.sample(tell_par)
        tell_spec_inv = spectrum.Spectrum(tell_wave, tell_spec, gpm=tell_gpm)
        tell_spec_inv.inverse()
        # Divide the observed spectrum by the initial telluric model
        corr_spec = obs_spec.resample(tell_wave)
        corr_spec.multiply(tell_spec_inv)

        # The source spectrum parameters can be None, although they should
        # effectively never be none because that means there's no overall
        # normalization.  I.e., in general, the order of the polynomial included
        # in the source model should always be >= 0.
        src_par = self.src_model.par_guess(corr_spec)

        return tell_par if src_par is None else np.append(src_par, tell_par)

    def par_bounds(self,
        guess_par, rel_coeff_bounds=(-20.0, 20.0), abs_coeff_bounds=(-5.0, 5.0),
        resolution_frac_bounds=(0.3, 1.5), pix_shift_bounds=(-5.0,5.0),
        pix_stretch_bounds=(0.98,1.02)
    ):
        """
        Provide the bounds for all model parameters.

        Parameters
        ----------
        guess_par : list, `numpy.ndarray`_
            Initial guess for the model parameters.  Length must be :attr:`npar`.
        rel_coeff_bounds : tuple, optional
            The lower and upper boundary of each coefficient relative to the
            value of the guess value.  For example, (0.5, 2,0) means that every
            coefficient must be within a factor of 2 of the guess value.   These
            limits are used for *all* polynomial coefficients.  See description above.
        abs_coeff_bounds : tuple, optional
            The absolute lower and upper boundaries for the coefficients; i.e.,
            this is *not* relative to the guess value.  These limits are used
            for *all* polynomial coefficients.  See description above.
        resolution_frac_bounds : :obj:`tuple`, optional
            Lower and upper bounds for the spectral resolution expressed as a
            fraction of the guess resolution.
        pix_shift_bounds : :obj:`tuple`, optional
            Lower and upper bounds for the pixel shift.
        pix_stretch_bounds : :obj:`tuple`, optional
            Lower and upper bounds for the pixel stretch.

        Returns
        -------
        list
            A list of two-tuples providing the lower and upper bounds for each
            model parameter.  Length is :attr:`npar`.
        """
        src_bounds = self.src_model.par_bounds(
            guess_par[:self.src_model.npar], rel_coeff_bounds, abs_coeff_bounds
        )
        tell_bounds = self.tell_model.par_bounds(
            guess_par[self.src_model.npar:], resolution_frac_bounds=resolution_frac_bounds,
            pix_shift_bounds=pix_shift_bounds, pix_stretch_bounds=pix_stretch_bounds
        )
        return tell_bounds if src_bounds is None else src_bounds + tell_bounds
    
    def sample(self, theta):
        """
        Sample the combined source + telluric model.

        Parameters
        ----------
        theta : `numpy.ndarray`_
            Model parameters.  The length mush be :attr:`npar`.

        Returns
        -------
        wave : `numpy.ndarray`_
            Model wavelength array.
        flux : `numpy.ndarray`_
            Model flux array.
        gpm : `numpy.ndarray`_, boolean
            Good pixel mask.
        """
        src_spec, src_gpm = self.src_model.sample(
            theta[:self.src_model.npar] if self.src_model.npar > 0 else None
        )
        tell_wave, tell_spec, tell_gpm = self.tell_model.sample(theta[self.src_model.npar:])
        # TODO: Have this return a Spectrum object so that it can be easily
        # resampled.
        return tell_wave, src_spec * tell_spec, src_gpm & tell_gpm
    
    def _waves_match(self, obs_spec):
        """
        Confirm that the wavelength arrays of the observed spectrum to fit and
        the model spectra are the same.

        Parameters
        ----------
        obs_spec : :class:`~pypeit.core.spectrum.Spectrum`
            Spectrum to be fit.

        Returns
        -------
        bool
            True if the wavelength arrays match, False otherwise.
        """
        return obs_spec.wave.size == self.wave.size and np.allclose(obs_spec.wave, self.wave)

    def fit_fom(self, theta):
        """
        Compute the fit figure-of-merit (FOM) that is minimized when fitting the
        observed spectrum.

        This function uses a Huber loss function with a transition from squared
        to absolute loss at an error-weighted (if errors are available) residual
        of 2.

        The observed spectrum should be available via :attr:`obs_spec` *before*
        calling this function, and it must have the same wavelength grid as
        :attr:`src_model` and :attr:`tell_model`.

        Parameters
        ----------
        theta : `numpy.ndarray`_
            Model parameters.  The length mush be :attr:`npar`.

        Returns
        -------
        float
            Value of the fit figure-of-merit.
        """
        if self.obs_spec is None:
            raise PypeItError('Observed spectrum not set.  Cannot compute fit metric.')
        if not self._waves_match(self.obs_spec):
            raise PypeItError('Observed spectrum wavelength array does not match model spectra.')

        # Get the model spectrum
        _, model_flux, model_gpm = self.sample(theta)

        # Check if everything will be masked, and return infinity if so.
        resid_gpm = self.obs_spec.gpm & model_gpm
        if not np.any(resid_gpm):
            return np.inf
        
        # Impose a penalty if the model is masked anywhere that the data is not
        penalty_gpm = self.obs_spec.gpm & np.logical_not(model_gpm)
        if np.any(penalty_gpm):
            model_flux[penalty_gpm] = 0.0
            resid_gpm[penalty_gpm] = True   # This effectively makes resid_gpm equal to self.obs_spec.gpm

        # Compute the vector of error-normalized residuals and the fit metric
        resid = self.obs_spec.flux - model_flux
        if self.obs_spec.ivar is not None:
            resid *= np.sqrt(self.obs_spec.ivar)
        # TODO: Consider using pseudo_huber for a smooth derivative
        return np.sum(special.huber(2.0, resid[resid_gpm]))
#        fom = np.sum(special.huber(2.0, resid[resid_gpm]))
#        print(f'npix: {np.sum(resid_gpm)}; fom: {fom:0.4e}')
#        return fom
    
    def _init_fit_pop(self, bounds, guess_par, popsize, ballsize, rng):
        """
        Helper function used to initialize the population for the differential
        evolution optimizer.

        This function should only be called if at least one of the parameters
        have a provided guess value.  The ``guess_par`` object can have ``None``
        elements, indicating that there is no guess.  For these parameters, the
        population follows a latin hypercube distribution over the space defined
        by the parameter boundaries.  The population distribution for all the
        remaining parameters is a multivariate Normal distribution centered on
        the guess value and with a sigma set by the ``ballsize`` and the
        parameter bounds.

        Parameters
        ----------
        bounds : list
            A list of two-tuples providing the lower and upper bounds for each
            model parameter.  Length must be :attr:`npar`.  Cannot be ``None``.
        guess_par : list, `numpy.ndarray`_, optional
            Initial guess for the model parameters.  Length must be
            :attr:`npar`.  Cannot be ``None``, but individual elements in the
            vector can be.  See description above for treatment of ``None``
            elements.
        popsize : int, optional
            The population size, where the number of samples is always ``popsize
            * npar``.
        ballsize : float, optional
            When constructing the population as a multivariate Gaussian
            distribution centered on the guess parameters, this is the scale
            (1-sigma) of the distribution as a fraction of the separation
            between the parameter bounds.
        rng : int, `numpy.random.Generator`, optional
            Random-number generator object or seed used for drawing samples for
            the population.

        Returns
        -------
        `numpy.ndarray`_
            Initial population for the differential evolution optimizer.  Shape
            is (``popsize * npar``, ``npar``).

        """
        if guess_par is None or all(use_lhs:=[p is None for p in guess_par]):
            raise PypeItError('Must provide at least one guess parameter!')
        if len(guess_par) != self.npar:
            raise PypeItError('Length of guess_par does not match number of model parameters!')

        # Number of samples for the population
        npop = popsize * self.npar

        # Cast to array for slicing
        _guess_par = np.asarray(guess_par)

        # Isolate the lower and upper bounds
        lb, ub = np.asarray(bounds).T
        db = ub - lb

        # Setup the generator.  If rng is already a Generator, default_rng just
        # returns it.
        _rng = np.random.default_rng(rng)

        # Initialize the array to hold the random samples
        init = np.empty((npop, self.npar), dtype=float)

        # Get the latin hypercube samples
        nlhs = np.sum(use_lhs)
        if nlhs > 0:
            init[:,use_lhs] = (
                qmc.LatinHypercube(d=nlhs, seed=_rng).random(npop) * db[None,use_lhs]
                + lb[None,use_lhs]
            )

        # Get the (uncorrelated) multivariate Normal samples, and clip the
        # distribution to ensure the samples are within the bounds.
        use_mvn = np.logical_not(use_lhs)
        nmvn = np.sum(use_mvn)
        init[:,use_mvn] = np.clip(
            _rng.normal(size=(npop, nmvn)) * ballsize * db[None,use_mvn] + _guess_par[None,use_mvn],
            a_min=lb[use_mvn], a_max=ub[use_mvn]
        )

        # Done
        return init

    # TODO: Does this need the airmass?
    #   airmass : float, optional
    #       Airmass of the observation.  This is only needed if the telluric
    #       model requires it (e.g., for grid models).
    def fit(
        self, obs_spec, bounds, guess_par=None, popsize=30, ballsize=5e-4, rng=None,
        init='latinhypercube', **kwargs
    ):
        """
        Fit an observed spectrum using a parameterized source spectrum and a
        telluric transmission spectrum.

        The optimization algorithm used is
        `scipy.optimize.differential_evolution`.

        Parameters
        ----------
        obs_spec : :class:`~pypeit.core.spectrum.Spectrum`
            Spectrum to be fit.  If the wavelength array does *not* match the
            wavelength arrays of the source and telluric model objects, the
            spectrum will be resampled such that it does.
        bounds : list
            A list of two-tuples providing the lower and upper bounds for each
            model parameter.  Length must be :attr:`npar`.
        guess_par : list, `numpy.ndarray`_, optional
            Initial guess for the model parameters.  If ``None``, ``init`` must
            provide the mode used to construct the initial sample population
            used by `scipy.optimize.differential_evolution`.  If not ``None``,
            the length must be :attr:`npar`.  Values in the vector that are
            ``None`` indicate that there is no guess value and the population
            samples are determined using a latin hypercube distribution.
        popsize : int, optional
            Specify the population size for the differential evolution
            optimizer.  Note that ``popsize`` is *ignored* if ``init`` provides
            the initial population directly. See the
            `scipy.optimize.differential_evolution` documentation for details.
        ballsize : float, optional
            When constructing the population as a multivariate Gaussian
            distribution about the the guess parameters, this is the scale of
            the distribution as a fraction of the separation between the
            parameter bounds.
        rng : int, `numpy.random.Generator`, optional
            Random-number generator object or seed used for drawing samples for
            the population.  This is provided to allow the algorithm to be
            reproducible.
        init : str, `numpy.ndarray`_, optional
            The method of initializing the sample population for the
            differential evolution optimizer, or the sample population itself.
            See the `scipy.optimize.differential_evolution` documentation for
            details.
        **kwargs : dict, optional
            Keywords passed directly to `scipy.optimize.differential_evolution`.
        """
        # Setup the generator.  If rng is already a Generator, default_rng just
        # returns it.
        _rng = np.random.default_rng(rng)

        # Get the sample population if the guess parameters are provided
        _init = (
            init if guess_par is None
            else self._init_fit_pop(bounds, guess_par, popsize, ballsize, _rng)
        )

        # If the wavelength arrays do not match, resample the observed spectrum
        # TODO: We should resample the *model*, not the data
        self.obs_spec = (
            obs_spec if self._waves_match(obs_spec)
            else obs_spec.resample(self.src_model.wave)
        )

        # Perform the fit
        result = optimize.differential_evolution(
            self.fit_fom, bounds, rng=_rng, init=_init, popsize=popsize, **kwargs
        )
        
        # Return the best fit parameters
        return result.x

