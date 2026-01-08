"""
Module for fitting a telluric + object model to an observed spectrum.
"""

from IPython import embed

import numpy as np
from scipy import optimize
from scipy import special

from pypeit import log
from pypeit import PypeItError
from pypeit import utils
from pypeit.core import spectrum

#result = scipy.optimize.differential_evolution(tellfit_chi2, bounds, args=(flux, thismask, arg_dict,), seed=rng,
#                                                   init = init, updating='immediate', popsize=popsize,
#                                                   recombination=arg_dict['recombination'], maxiter=arg_dict['diff_evol_maxiter'],
#                                                   polish=arg_dict['polish'], disp=arg_dict['disp'])

class ObservedSourceModel:

    def __init__(self, obj_model, tell_model):
        """
        Class to perform the object + telluric model fit to an observed spectrum.

        Parameters
        ----------
        obj_model : :class`~pypeit.telluric.object.AdjustedSpectrumModel`
            The class to use when modeling the object spectrum.  The object
            cannot be ``None`` and the model must be fully initialized (i.e.,
            ``obj_model.sample()`` should not fail).
        tell_model : :class:`~pypeit.telluric.model.TelluricModel`
            The class to use when modeling the telluric spectrum.  The object
            cannot be ``None``, the model must be fully initialized (i.e.,
            ``tell_model.sample()`` should not fail), and the wavelength array
            of the model must match the object model.
        """
        self.obj_model = obj_model
        self.tell_model = tell_model

        if not np.allclose(self.obj_model.wave, self.tell_model.wave):
            raise PypeItError('Object and telluric model wavelength arrays do not match.')

        # Kept during fitting
        self._reset_fit()

    def _reset_fit(self):
        """
        Reset the attributes used during fitting.
        """
        self.obs_spec = None

    @property
    def npar(self):
        """
        Total number of parameters in the object + telluric model.
        """
        return self.obj_model.npar + self.tell_model.npar

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
        # Guess the telluric parameters first.  The object spectrum is passed to
        # the guess function, but the current models don't actually use the flux
        # vector.  They only use the wavelength vector to guess the spectral
        # resolution.
        tell_par = self.tell_model.par_guess(obs_spec)

        # Use the guess parameters to generate an initial telluric model
        tell_wave, tell_spec, tell_gpm = self.tell_model.sample(tell_par)
        tell_spec_inv = spectrum.Spectrum(tell_wave, tell_spec, gpm=tell_gpm)
        tell_spec_inv.inverse()
        # Divide the observed spectrum by the initial telluric model
        corr_spec = obs_spec.resample(tell_wave)
        corr_spec.multiply(tell_spec_inv)

        # The object spectrum parameters can be None, although they should
        # effectively never be none because that means there's no overall
        # normalization.  I.e., in general, the order of the polynomial included
        # in the object model should always be >= 0.
        obj_par = self.obj_model.par_guess(corr_spec)

        return tell_par if obj_par is None else np.append(obj_par, tell_par)

    def par_bounds(self,
        guess_par, rel_coeff_bounds=(-20.0, 20.0), abs_coeff_bounds=(-5.0, 5.0),
        resolution_frac_bounds=(0.3, 1.5), pix_shift_bounds=(-5.0,5.0),
        pix_stretch_bounds=(0.98,1.02)
    ):
        obj_bounds = self.obj_model.par_bounds(
            guess_par[:self.obj_model.npar], rel_coeff_bounds, abs_coeff_bounds
        )
        tell_bounds = self.tell_model.par_bounds(
            guess_par[self.obj_model.npar:], resolution_frac_bounds=resolution_frac_bounds,
            pix_shift_bounds=pix_shift_bounds, pix_stretch_bounds=pix_stretch_bounds
        )
        return tell_bounds if obj_bounds is None else obj_bounds + tell_bounds
    
    def sample(self, theta):
        """
        Sample the combined object + telluric model.

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
        src_spec, src_gpm = self.obj_model.sample(
            theta[:self.obj_model.npar] if self.obj_model.npar > 0 else None
        )
        tell_wave, tell_spec, tell_gpm = self.tell_model.sample(theta[self.obj_model.npar:])
        return tell_wave, src_spec * tell_spec, src_gpm & tell_gpm
    
    def fit_metric(self, theta):
        """
        Compute the fit metric that is minimized when fitting the observed
        spectrum.

        This function uses a Huber loss function with a transition from squared
        to absolute loss at an error-weighted (if errors are available) residual
        of 2.

        The observed spectrum should be available via :attr:`obs_spec` *before*
        calling this function, and it must have the same wavelength grid as
        :attr:`obj_model` and :attr:`tell_model`.

        Parameters
        ----------
        theta : `numpy.ndarray`_
            Model parameters.  The length mush be :attr:`npar`.

        Returns
        -------
        float
            Value of the fit metric.
        """
        if self.obs_spec is None:
            raise PypeItError('Observed spectrum not set.  Cannot compute fit metric.')
        if not np.allclose(self.obs_spec.wave, self.obj_model.wave):
            raise PypeItError('Observed spectrum wavelength array does not match model spectra.')

        # Get the model spectrum
        model_wave, model_flux, model_gpm = self.sample(theta)

        # Check if everything will be masked, and return infinity if so.
        resid_gpm = self.obs_spec.gpm & model_gpm
        if not np.any(resid_gpm):
            return np.inf

        # Compute the vector of error-normalized residuals and the fit metric
        resid = self.obs_spec.flux - model_flux
        if self.obs_spec.ivar is not None:
            resid *= np.sqrt(self.obs_spec.ivar)
        # TODO: Consider using pseudo_huber for a smooth derivative
        return np.sum(special.huber(2.0, resid[resid_gpm]))

    def fit(
        self, obs_spec, guess_par, bounds, airmass=None, ballsize=5e-4, diff_evol_maxiter=1000,
        seed=None, init=None, updating='immediate', popsize=30, recombination=0.7, maxiter=1,
        polish=True, disp=False, 
    ):
        """
        Fit an observed spectrum using a parameterized source spectrum and a
        telluric transmission spectrum.

        Parameters
        ----------
        obs_spec : :class:`~pypeit.core.spectrum.Spectrum`
            Spectrum to be fit.
        guess_par : `numpy.ndarray`_
            Initial guess for the model parameters.  Length must be :attr:`npar`.
        bounds : list
            A list of two-tuples providing the lower and upper bounds for each
            model parameter.  Length must be :attr:`npar`.
        airmass : float, optional
            Airmass of the observation.  This is only needed if the telluric
            model requires it (e.g., for grid models).
        ballsize : float, optional
            This parameter governs how the differential evolution random
            population is initialized for the object model and for subsequent
            iterations.  See the `scipy.optimize.differential_evolution`
            documentation for details.
        diff_evol_maxiter : int, optional
            Maximum number of iterations for the differential evolution
            optimizer.
        seed : int, optional
            Seed to be used to initialize the random number generator for the
            differential evolution optimizer.  A specific seed is used because
            otherwise the random number generator will use the time for the seed
            and the results will not be reproducible.
        init : str or `numpy.ndarray`_, optional
            Specify the population initialization for the differential evolution
            optimizer. See the `scipy.optimize.differential_evolution`
            documentation for details.
        updating : str, optional
            Specify the updating strategy for the differential evolution
            optimizer. See the `scipy.optimize.differential_evolution`
            documentation for details.
        popsize : int, optional
            Specify the population size for the differential evolution
            optimizer. See the `scipy.optimize.differential_evolution`
            documentation for details.
        recombination : float, optional
            Specify the recombination constant for the differential evolution
            optimizer. See the `scipy.optimize.differential_evolution`
            documentation for details.
        maxiter : int, optional
            Specify the maximum number of iterations for the differential
            evolution optimizer. See the `scipy.optimize.differential_evolution`
            documentation for details.
        polish : bool, optional
            Specify whether to polish the best solution at the end of the
            differential evolution optimization. See the
            `scipy.optimize.differential_evolution` documentation for details.
        disp : bool, optional
            Specify whether to display the progress of the differential
            evolution optimization. See the `scipy.optimize.differential_evolution`
            documentation for details.

        """
        pass



def tellfit(flux, thismask, arg_dict, init_from_last=None):
    """
    Routine to perform the object + telluric model fitting for telluric
    corrections. This is a general abstracted routine that performs the
    fits for any object model that the user provides.

    Args:
        flux (`numpy.ndarray`_):
            The flux of the object being fit
        thismask (`numpy.ndarray`_, boolean):
            A mask indicating which values are to be fit. This is a good
            pixel mask, i.e. True=Good
        arg_dict (dict):
            A dictionary containing the parameters needed to evaluate
            the telluric model and the object model.  The required keys
            are:

                - ``arg_dict['flux_ivar']``:  Inverse variance for the
                  flux array
                - ``arg_dict['tell_dict']``: Dictionary containing the
                  telluric model and its parameters read in by
                  read_telluric_pca or read_telluric_grid
                - ``arg_dict['ind_lower']``: Lower index into the
                  telluric model wave_grid to trim down the telluric
                  model.
                - ``arg_dict['ind_upper']``: Upper index into the
                  telluric model wave_grid to trim down the telluric
                  model.
                - ``arg_dict['obj_model_func']``: User provided function
                  for evaluating the object model
                - ``arg_dict['obj_dict']``:  Dictionary containing the
                  object model arguments which is passed to the
                  obj_model_func

        init_from_last (object, optional):
             Optional. Result object returned by the differential
             evolution optimizer for the last iteration. If this is passed the code
             will initialize from the previous best-fit for faster convergence.


    Returns:
        tuple:  Returns three objects:

            - result (obj): Result object returned by the differential
              evolution optimizer
            - modelfit (`numpy.ndarray`_): Modelfit to the input flux.
              This has the same size as the flux
            - ivartot (`numpy.ndarray`_): Corrected inverse variances
              for the flux. This has the same size as the flux. The
              errors are renormalized using the renormalize_errors
              function by a correction factor, i.e. ivartot =
              flux_ivar/sigma_corr**2

    """

    # Unpack arguments
    obj_model_func = arg_dict['obj_model_func'] # Evaluation function
    flux_ivar = arg_dict['ivar'] # Inverse variance of flux or counts
    bounds = arg_dict['bounds']  # bounds for differential evolution optimization
    rng = arg_dict['rng']      # Seed for differential evolution optimizaton
    maxiter = arg_dict['diff_evol_maxiter'] # Maximum number of iterations
    ballsize = arg_dict['ballsize'] # Ballsize for initialization from a previous optimum
    nparams = len(bounds) # Number of parameters in the model
    popsize = arg_dict['popsize'] # Note this does nothing if the init is done from a previous iteration or optimum
    nsamples = arg_dict['popsize']*nparams
    teltype = arg_dict['tell_dict']['teltype']
    # FD: Assumes shift and stretch are turned on.
    if teltype == 'pca':
        ntheta_tell = arg_dict['tell_npca']+3 # Total number of telluric model parameters in PCA mode
    elif teltype == 'grid':
        ntheta_tell = 4+3 # Total number of telluric model parameters in grid mode

    # Decide how to initialize
    if init_from_last is not None:
        # Use a Gaussian ball about the optimum from a previous iteration
        init = np.array([[np.clip(param + ballsize*(bounds[i][1] - bounds[i][0]) * rng.standard_normal(1)[0],
                                  bounds[i][0], bounds[i][1])
                                  for i, param in enumerate(init_from_last.x)] for jsamp in range(nsamples)])
    elif 'init_obj_opt_theta' in arg_dict['obj_dict']:
        # Initialize from  the object parameters. Use a Gaussian ball about the best object model, and latin hypercube
        # for the telluric parameters
        bounds_obj = arg_dict['obj_dict']['bounds_obj']
        init_obj = np.array([[np.clip(param + ballsize*(bounds_obj[i][1] - bounds_obj[i][0]) * rng.standard_normal(1)[0],
                                      bounds_obj[i][0], bounds_obj[i][1]) for i, param in enumerate(arg_dict['obj_dict']['init_obj_opt_theta'])]
                             for jsamp in range(nsamples)])
        tell_lhs = utils.lhs(ntheta_tell, samples=nsamples)
        init_tell = np.array([[bounds[-idim][0] + tell_lhs[isamp, idim] * (bounds[-idim][1] - bounds[-idim][0])
                               for idim in range(ntheta_tell)] for isamp in range(nsamples)])
        init = np.hstack((init_obj, init_tell))
    else:
        # If this is the first iteration and no object model optimum is presented, use a latin hypercube which is the default
        init = 'latinhypercube'

    result = scipy.optimize.differential_evolution(tellfit_chi2, bounds, args=(flux, thismask, arg_dict,), seed=rng,
                                                   init = init, updating='immediate', popsize=popsize,
                                                   recombination=arg_dict['recombination'], maxiter=arg_dict['diff_evol_maxiter'],
                                                   polish=arg_dict['polish'], disp=arg_dict['disp'])
                                        
    theta_obj  = result.x[:-ntheta_tell]
    theta_tell = result.x[-ntheta_tell:]
    tell_model = eval_telluric(theta_tell, arg_dict['tell_dict'],
                                 ind_lower=arg_dict['ind_lower'], ind_upper=arg_dict['ind_upper'])
    obj_model, modelmask = obj_model_func(theta_obj, arg_dict['obj_dict'])
    totalmask = thismask & modelmask
    chi_vec = totalmask*(flux - tell_model*obj_model)*np.sqrt(flux_ivar)

    debug = arg_dict['debug'] if 'debug' in arg_dict else False

    # Name of function for title in case QA requested
    obj_model_func_name = getattr(obj_model_func, '__name__', repr(obj_model_func))
    sigma_corr, maskchi = coadd.renormalize_errors(chi_vec, mask=totalmask, title = obj_model_func_name,
                                                   debug=debug)
    ivartot = flux_ivar/sigma_corr**2

    return result, tell_model*obj_model, ivartot

