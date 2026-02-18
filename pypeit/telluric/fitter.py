"""
Module for fitting a source + telluric model to an observed spectrum.
"""

import copy
import inspect
from IPython import embed

import numpy as np
from scipy import optimize
from scipy import special
from scipy.stats import qmc

from pypeit import log
from pypeit import PypeItError
from pypeit import utils
from pypeit.core import coadd
from pypeit.core import pydl
from pypeit.core import spectrum
from pypeit.par import funcpar


class ObservedSourceModelFitter:
    """
    Class to perform the source + telluric model fit to an observed spectrum.

    Parameters
    ----------
    src_model : :class`~pypeit.telluric.source.AdjustedSpectrumModel`
        The class to use when modeling the source spectrum.  The object cannot
        be ``None`` and the model must be fully initialized (i.e.,
        ``src_model.sample()`` should not fail).
    tel_model : :class:`~pypeit.telluric.model.TelluricModel`
        The class to use when modeling the telluric spectrum.  The object cannot
        be ``None``, the model must be fully initialized (i.e.,
        ``tel_model.sample()`` should not fail), and the wavelength array of
        the model must match the source spectrum model.

    Attributes
    ----------
    src_model : :class`~pypeit.telluric.source.AdjustedSpectrumModel`
        Source model spectrum
    tel_model : :class`~pypeit.telluric.model.TelluricModel`
        Telluric model spectrum
    obs_spec : :class:`~pypeit.core.spectrum.Spectrum`
        The observed spectrum to be fit.
    """
    def __init__(self, src_model, tel_model):
        self.src_model = src_model
        self.tel_model = tel_model

        if not np.allclose(self.src_model.wave, self.tel_model.wave):
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
        return self.src_model.npar + self.tel_model.npar

    def par_guess(self, obs_spec, resolution_guess=None):
        """
        Provide an initial guess for all the model parameters.

        Parameters
        ----------
        obs_spec : :class:`~pypeit.core.spectrum.Spectrum`
            Spectrum to be fit.
        resolution_guess : float, optional
            Initial guess for the spectral resolution (R = lambda/delta_lambda)
            of the telluric model.  If None, 
            :func:`~pypeit.core.wavecal.wvutils.get_sampling` is used to
            estimate the resolution using the wavelength vector for
            ``obs_spec``.

        Returns
        -------
        :class:`numpy.ndarray`
            Guess parameters
        """
        # Guess the telluric parameters first.  Note that the current telluric
        # model classes do not use the flux vector of the observed spectrum.
        # They only use the wavelength vector to guess the spectral resolution.
        tell_par = self.tel_model.par_guess(
            obs_wave=obs_spec.wave, resolution_guess=resolution_guess
        )

        # Use the guess parameters to generate an initial telluric model
        tell_wave, tell_spec, tell_gpm = self.tel_model.sample(tell_par)
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
        guess_par : list, :class:`numpy.ndarray`
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
        tell_bounds = self.tel_model.par_bounds(
            guess_par[self.src_model.npar:], resolution_frac_bounds=resolution_frac_bounds,
            pix_shift_bounds=pix_shift_bounds, pix_stretch_bounds=pix_stretch_bounds
        )
        return tell_bounds if src_bounds is None else src_bounds + tell_bounds
    
    def sample(self, theta):
        """
        Sample the combined source + telluric model.

        Parameters
        ----------
        theta : :class:`numpy.ndarray`
            Model parameters.  The length mush be :attr:`npar`.

        Returns
        -------
        wave : :class:`numpy.ndarray`
            Model wavelength array.
        flux : :class:`numpy.ndarray`
            Model flux array.
        gpm : :class:`numpy.ndarray`
            Good pixel mask.
        """
        src_spec, src_gpm = self.src_model.sample(
            theta[:self.src_model.npar] if self.src_model.npar > 0 else None
        )
        tell_wave, tell_spec, tell_gpm = self.tel_model.sample(theta[self.src_model.npar:])
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
        :attr:`src_model` and :attr:`tel_model`.

        Parameters
        ----------
        theta : :class:`numpy.ndarray`
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
            # This will set the value of the residual to the observed spectrum
            model_flux[penalty_gpm] = 0.0
            # This makes resid_gpm equal to self.obs_spec.gpm
            # TODO: Remove this and just use self.obs_spec.gpm directly below?
            resid_gpm[penalty_gpm] = True

        # Compute the vector of error-normalized residuals and the fit metric
        resid = self.obs_spec.flux - model_flux
        if self.obs_spec.ivar is not None:
            resid *= np.sqrt(self.obs_spec.ivar)
        # TODO: Consider using pseudo_huber for a smooth derivative
        return np.sum(special.huber(2.0, resid[resid_gpm]))
#        fom = np.sum(special.huber(2.0, resid[resid_gpm]))
#        print(f'npix: {np.sum(resid_gpm)}; fom: {fom:0.4e}')
#        return fom
    
    def init_fit_pop(self, bounds, guess_par, popsize, ballsize, rng):
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
        guess_par : list, :class:`numpy.ndarray`, optional
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
        rng : int, :class:`numpy.random.Generator`, optional
            Random-number generator object or seed used for drawing samples for
            the population.

        Returns
        -------
        :class:`numpy.ndarray`
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
#        popsize=30, rng=None,
#        init='latinhypercube', 
#        popsize : int, optional
#            Specify the population size for the differential evolution
#            optimizer.  Note that ``popsize`` is *ignored* if ``init`` provides
#            the initial population directly. See the
#            `scipy.optimize.differential_evolution` documentation for details.
#        rng : int, `numpy.random.Generator`, optional
#            Random-number generator object or seed used for drawing samples for
#            the population.  This is provided to allow the algorithm to be
#            reproducible.
#        init : str, `numpy.ndarray`_, optional
#            The method of initializing the sample population for the
#            differential evolution optimizer, or the sample population itself.
#            See the `scipy.optimize.differential_evolution` documentation for
#            details.
    def fit(self, obs_spec, bounds, guess_par=None, ballsize=5e-4, de_par=None):
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
        guess_par : list, :class:`numpy.ndarray`, optional
            Initial guess for the model parameters.  If ``None``, the ``init``
            parameter used by `scipy.optimize.differential_evolution` (passed as
            a kwarg) must provide the mode used to construct the initial sample
            population.  If not ``None``, the length must be :attr:`npar`.
            Values in the vector that are ``None`` indicate that there is no
            guess value and the population samples are determined using a latin
            hypercube distribution; see :func:`init_fit_pop`.
        ballsize : float, optional
            When constructing the population as a multivariate Gaussian
            distribution about the the guess parameters, this is the scale of
            the distribution as a fraction of the separation between the
            parameter bounds.
        de_par : :class:`~pypeit.par.funcpar.DifferentialEvolutionPar`, optional
            The keyword arguments to use for the
            `scipy.optimize.differential_evolution` optimizer.  If ``None``, the
            default values are used.  See the
            `scipy.optimize.differential_evolution` documentation for details.
        """
        # Get the differential_evolution parameters
        _de_par = funcpar.DifferentialEvolutionPar() if de_par is None else de_par
        if not isinstance(_de_par, funcpar.DifferentialEvolutionPar):
            raise PypeItError(
                'de_par must be an instance of pypeit.par.funcpar.DifferentialEvolutionPar!'
            )

        # TODO: I don't like this deepcopy, but I'm not sure there's a way
        # around it.
        de_kwargs = copy.deepcopy(_de_par.data)

        # If the guess parameters are provided, use them to construct the
        # initial population used by differential evolution
        if guess_par is not None:
            # Setup the generator and add it to the optimizer kwargs.  If rng is
            # already a Generator, default_rng just returns it.
            rng = np.random.default_rng(de_kwargs.pop('rng', _de_par.default['rng']))

            # Get the initial population and add it to the optimizer kwargs
            de_kwargs.pop('init', None)  # Remove any existing 'init' entry
            init = self.init_fit_pop(
                bounds, guess_par, de_kwargs.pop('popsize', _de_par.default['popsize']),
                ballsize, rng
            )

        # If the wavelength arrays do not match, resample the observed spectrum
        # TODO: We should resample the *model*, not the data
        self.obs_spec = (
            obs_spec if self._waves_match(obs_spec)
            else obs_spec.resample(self.src_model.wave)
        )

        embed(header='fitter')
        exit()

        # Perform the fit
        result = optimize.differential_evolution(
            self.fit_fom, bounds, rng=rng, init=init, **de_kwargs
        )
        
        # Return the best fit parameters
        return result.x

    def iter_fit(self, obs_spec, bounds, guess_par=None, ballsize=5e-4, max_rej_iter=1,
                 de_par=None, rej_par=None):
        """
        Fit an observed spectrum using a parameterized source spectrum and a
        telluric transmission spectrum with rejection iterations.

        Parameters
        ----------
        obs_spec : :class:`~pypeit.core.spectrum.Spectrum`
            Spectrum to be fit.  If the wavelength array does *not* match the
            wavelength arrays of the source and telluric model objects, the
            spectrum will be resampled such that it does.
        bounds : list
            A list of two-tuples providing the lower and upper bounds for each
            model parameter.  Length must be :attr:`npar`.
        guess_par : list, :class:`numpy.ndarray`, optional
            Initial guess for the model parameters.  If ``None``, the ``init``
            parameter used by `scipy.optimize.differential_evolution` (passed as
            a kwarg) must provide the mode used to construct the initial sample
            population.  If not ``None``, the length must be :attr:`npar`.
            Values in the vector that are ``None`` indicate that there is no
            guess value and the population samples are determined using a latin
            hypercube distribution; see :func:`init_fit_pop`.
        ballsize : float, optional
            When constructing the population as a multivariate Gaussian
            distribution about the the guess parameters, this is the scale of
            the distribution as a fraction of the separation between the
            parameter bounds.
        max_rej_iter : int, optional
            Maximum number of rejection iterations to perform.  Must be >= 1.
            If you do not want to perform any rejection iterations, use
            :func:`fit`.  The maximum number of *fitting* iterations is
            ``max_rej_iter+1``.
        de_par : :class:`~pypeit.par.funcpar.DifferentialEvolutionPar`, optional
            The keyword arguments to use for the
            `scipy.optimize.differential_evolution` optimizer.  If ``None``, the
            default values are used.  See the
            `scipy.optimize.differential_evolution` documentation for details.
        rej_par : :class:`~pypeit.par.funcpar.DJSRejectPar`, optional
            The keyword arguments to use for the `pypeit.core.pydl.djs_reject`
            function used during the rejection iterations.  If ``None``, the
            default values are used.  See the
            :func:`~pypeit.core.pydl.djs_reject` documentation for details.
        """
        if max_rej_iter < 1:
            raise PypeItError(
                'When fitting observed source model, max_rej_iter must be >= 1!  For a fit '
                'without rejections, use the fit() method.'
            )

        # Get the rejection parameters
        _rej_par = funcpar.DJSRejectPar() if rej_par is None else rej_par
        if not isinstance(_rej_par, funcpar.DJSRejectPar):
            raise PypeItError(
                'rej_par must be an instance of pypeit.par.funcpar.DJSRejectPar!'
            )

        # If the wavelength arrays do not match, resample the observed spectrum
        # TODO: We should resample the *model*, not the data
        self.obs_spec = (
            obs_spec if self._waves_match(obs_spec)
            else obs_spec.resample(self.src_model.wave)
        )

        # Rejection iteration setup
        orig_gpm = self.obs_spec.gpm.copy()
        qdone = False
        i = 0
        _guess_par = None if guess_par is None else np.asarray(guess_par).copy()

        # Iteration loop
        while not qdone and i < max_rej_iter+1:

            # Get the best-fit parameters.  Note the fit uses:
            # - the inverse variances provided by self.obs_spec.  These are
            #   *not* rescaled between iterations.
            # - the current rejection mask

            print(f'Fitting iteration {i+1} of {max_rej_iter+1}...')
            best_fit_par = self.fit(
                self.obs_spec, bounds, guess_par=_guess_par, ballsize=ballsize, de_par=de_par
            )
            print(f'Best fit par: {best_fit_par}')

            if i == max_rej_iter:
                # No more fits will be done, so skip the rejection
                break

            # Get the best-fit model
            _, bf_model, bf_gpm = self.sample(best_fit_par)
            self.obs_spec.gpm &= bf_gpm

            # Get the error renormalization factor
            chi = (self.obs_spec.flux - bf_model) * np.sqrt(self.obs_spec.ivar)
            err_corr, _ = coadd.renormalize_errors(chi, gpm=self.obs_spec.gpm)

            # Perform the rejection using the rescaled inverse variance data
            rej_gpm, qdone = pydl.djs_reject(
                self.obs_spec.flux, bf_model, outmask=self.obs_spec.gpm, inmask=orig_gpm,
                invvar=self.obs_spec.ivar / err_corr**2, **_rej_par.data 
            )
            print(f'Number of pixels rejected in iteration {i+1}: '
                  f'{np.sum(orig_gpm) - np.sum(rej_gpm)}')
            if not np.any(rej_gpm):
                log.warning('All pixels have been rejected during the telluric fit iterations.  '
                            'Returning the fit before the most recent rejection iteration.')
                break

            # Prep for the next iteration
            self.obs_spec.gpm = rej_gpm
            _guess_par = best_fit_par
            i += 1

        log.info(
            f'Telluric fit completed after {i+1} iterations; '
            f'{np.sum(np.logical_not(self.obs_spec.gpm) & orig_gpm)} of {np.sum(orig_gpm)} pixels '
            'rejected during the fit.'
        )

        return best_fit_par, rej_gpm

        















