"""
Module for fitting a telluric + object model to an observed spectrum.
"""

from IPython import embed
import numpy as np

from pypeit import utils
from pypeit.core import spectrum

#result = scipy.optimize.differential_evolution(tellfit_chi2, bounds, args=(flux, thismask, arg_dict,), seed=rng,
#                                                   init = init, updating='immediate', popsize=popsize,
#                                                   recombination=arg_dict['recombination'], maxiter=arg_dict['diff_evol_maxiter'],
#                                                   polish=arg_dict['polish'], disp=arg_dict['disp'])

class TelluricFit:

    def __init__(self, obj_model, tell_model):
        """
        Class to perform the object + telluric model fit to an observed spectrum.

        Parameters
        ----------
        obj_model : :class`~pypeit.telluric.object.AdjustedSpectrumModel`
            The class to use when modeling the object spectrum.
        tell_model : :class:`~pypeit.telluric.model.TelluricModel`
            The class to use when modeling the telluric spectrum.
        """
        self.obj_model = obj_model
        self.tell_model = tell_model

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

        embed()
        exit()

        # Use the guess parameters to generate an initial telluric model
        tell_wave, tell_spec = self.tell_model.sample(tell_par)
        tell_spec_inv = spectrum.Spectrum(tell_wave, tell_spec).inverse()
        # Divide the observed spectrum by the initial telluric model
        corr_spec = obs_spec.multiply(tell_spec_inv)

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

    def fit(
        self, obs_spec, guess_par, bounds, airmass=None,
        ballsize=5e-4, diff_evol_maxiter=1000,
        seed=None, init=None, updating='immediate', popsize=30, recombination=0.7, maxiter=1,
        polish=True, disp=False, 
    ):
        pass


def tellfit_chi2(theta, flux, thismask, arg_dict):
    """
    Loss function which is optimized by differential evolution to perform the object + telluric model fitting for
    telluric corrections. This is a general abstracted routine that provides the loss function for any object model
    that the user provides.

    Args:
        theta (`numpy.ndarray`_):
           
            Parameter vector for the object + telluric model.

            This is actually two concatenated parameter vectors, one for
            the object and one for the telluric, i.e.:
            
            (in PCA mode)
                theta_obj = theta[:-(tell_npca+3)]
                theta_tell = theta[-(tell_npca+3):]
                
            (in grid mode)
                theta_obj = theta[:-7]
                theta_tell = theta[-7:]
                
            The telluric model theta_tell includes a either user-specified
            number of PCA coefficients (in PCA mode) or ambient pressure,
            temperature, humidity, and airmass (in grid mode) followed by
            spectral resolution, shift, and stretch.
            
            That is, in PCA mode,
            
                pca_coeffs = theta_tell[:tell_npca]
            
            while in grid mode,
            
                pressure    = theta_tell[0]
                temperature = theta_tell[1]
                humidity    = theta_tell[2]
                airmass     = theta_tell[3]
                
            with the last three indices of the array corresponding to
            
                resolution = theta_tell[-3]
                shift      = theta_tell[-2]
                stretch    = theta_tell[-1]

            The object model theta_obj can have an arbitrary size and is
            provided as an argument to obj_model_func

        flux (`numpy.ndarray`_):
           The flux of the object being fit
        thismask (`numpy.ndarray`_, boolean):
           A mask indicating which values are to be fit. This is a good pixel mask, i.e. True=Good
        arg_dict (dict):
           A dictionary containing the parameters needed to evaluate the telluric model and the object model. See
           documentation of tellfit for a detailed description.
    Returns:
        float:
           The value of the loss function at the location in parameter space theta. This is loss function is the thing
           that is minimized to perform the fit.

    """
    
    obj_model_func = arg_dict['obj_model_func']
    flux_ivar = arg_dict['ivar']
    teltype = arg_dict['tell_dict']['teltype']

    # TODO: make this work without shift and stretch?
    # Number of telluric model parameters, plus shift, stretch, and resolution
    if teltype == 'pca':
        nfit = arg_dict['tell_npca']+3
    elif teltype == 'grid':
        nfit = 4+3

    theta_obj = theta[:-nfit]
    theta_tell = theta[-nfit:]

    tell_model = eval_telluric(theta_tell, arg_dict['tell_dict'],
                                 ind_lower=arg_dict['ind_lower'], ind_upper=arg_dict['ind_upper'])
    obj_model, model_gpm = obj_model_func(theta_obj, arg_dict['obj_dict'])

    totalmask = thismask & model_gpm
    if not np.any(totalmask):
        return np.inf       # If everyting is masked retrun infinity
    else:
        chi_vec = totalmask * (flux - tell_model*obj_model) * np.sqrt(flux_ivar)
        robust_scale = 2.0
        huber_vec = scipy.special.huber(robust_scale, chi_vec)
        loss_function = np.sum(huber_vec * totalmask)
        return loss_function
    


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

