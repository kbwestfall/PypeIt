from IPython import embed

from astropy import table
import numpy as np
from scipy import interpolate

from pypeit import msgs
from pypeit import dataPaths
from pypeit import __version__
from pypeit import io
from pypeit.core.wavecal import wvutils


class QSOPCAModel:
    """

    """
    def __init__(self, filename, wave=None, redshift=None, npca=None):

        self.file = dataPaths.tel_model.get_file_path(filename)
        tbl = table.Table.read(self.file)

        self.wave = np.squeeze(tbl['WAVE_PCA'][0])
        if redshift is None:
            self.z_fid = 0.0
        else:
            self.z_fid = redshift
            self.wave *= (1 + self.z_fid)

        self.components = np.squeeze(tbl['PCA_COMP'])
        self.coeffs = np.squeeze(tbl['PCA_COEFFS'])
        self.npca = npca
        if self.npca is None:
            self.npca = self.components.shape[0]
        if self.npca > self.components.shape[0]:
            msgs.warn(
                f'Number of requested PCA components ({self.npca}) for QSO model is larger than '
                f'the number available ({self.components.shape[0]}).  Using all PCA components.')
            self.npca = self.components.shape[0]

        if wave is not None:
            interp = interpolate.interp1d(
                self.wave, self.components, bounds_error=False, fill_value=0.0, axis=1
            )
            self.components = interp(wave)
            self.wave = wave

        self.dloglam = np.median(np.diff(np.log10(self.wave)))

    def sample(self, theta):
        r"""
        Sample the QSO model.

        Parameters
        ----------
        theta : `numpy.ndarray`_
            Parameter vector.  The parameters are (0) redshift, (1)
            normalization, and the remaining :math:`N_{\rm PCA}` parameters are
            the PCA coefficients.

        Returns
        `numpy.ndarray`_
            The model QSO spectrum
        """
        # TODO:
        #   - Allow for subpixel shifts
        #   - Mask regions that roll to the other end
        #   - Construct the linear combination of components first, then shift.
        dshift = int(np.round(np.log10((1.0 + theta[0])/(1.0 + self.z_fid))/self.dloglam))
        _comp = np.roll(self.components[:self.npca,:], dshift, axis=1)
        return theta[1] * np.exp(np.dot(np.append(1.0,theta[2:]), _comp))
    

##############
# QSO Model #
##############
def init_qso_model(obj_params, iord, wave, flux, ivar, mask, tellmodel):
    """
    Routine used to initialize the quasar spectrum telluric fits

    Parameters
    ----------
    obj_params : dict

        Dictionary containing parameters necessary for initializing the quasar
        model

    iord : int

        Order in question. This is not used for initializing the qso model, but
        is kept here for compatibility with the standard function argument list

    wave : array shape (nspec,)
        Wavelength array for the object in question

    flux : array shape (nspec,)
        Flux array for the object in question

    ivar : array shape (nspec,)
        Inverse variance array for the oejct in question

    mask : array shape (nspec,)
        Good pixel mask for the object in question

    tellmodel : array shape (nspec,)

        This is a telluric model computed on the wave wavelength grid.
        Initialization usually requires some initial best guess for the telluric
        absorption, which is computed from the mean of the telluric model grid
        using the resolution of the spectrograph.

    Returns
    -------
    obj_dict : dict

        Dictionary containing the meta-information and variables that are used
        for the object model evaluations.  For example for the quasar model
        which based on a PCA decomposition, this dictionary holds the number of
        PCA basis vectors and those basis vectors.

    bounds_obj : tuple

        Tuple of bounds for each parameter that will be fit for the object model



    """

    qso_pca_dict = qso_init_pca(obj_params['pca_file'], wave, obj_params['z_qso'], obj_params['npca'])
    qso_pca_mean = np.exp(qso_pca_dict['components'][0, :])
    tell_mask = tellmodel > obj_params['tell_norm_thresh']
    # Create a reference model and bogus noise
    flux_ref = qso_pca_mean * tellmodel
    ivar_ref = utils.inverse((qso_pca_mean/100.0) ** 2)
    flam_norm_inv = coadd.robust_median_ratio(flux, ivar, flux_ref, ivar_ref, mask=mask, mask_ref=tell_mask)
    flam_norm = 1.0/flam_norm_inv

    # Set the bounds for the PCA and truncate to the right dimension
    coeffs = qso_pca_dict['coeffs'][:,1:obj_params['npca']]
    # Compute the min and max arrays of the coefficients which are not the norm, i.e. grab the coeffs that aren't the first one
    coeff_min = np.amin(coeffs, axis=0)  # only
    coeff_max = np.amax(coeffs, axis=0)
    # QSO redshift: can vary within delta_zqso
    bounds_z = [(obj_params['z_qso'] - obj_params['delta_zqso'], obj_params['z_qso'] + obj_params['delta_zqso'])]
    bounds_flam = [(flam_norm*obj_params['lbound_norm'], flam_norm*obj_params['ubound_norm'])] # Norm: bounds determined from estimate above
    bounds_pca = [(i, j) for i, j in zip(coeff_min, coeff_max)]        # Coefficients:  determined from PCA model
    bounds_obj = bounds_z + bounds_flam + bounds_pca
    # Create the obj_dict
    obj_dict = dict(npca=obj_params['npca'], pca_dict=qso_pca_dict)

    return obj_dict, bounds_obj

# QSO evaluation function. Model for QSO is a PCA spectrum
def eval_qso_model(theta, obj_dict):
    """
    Routine to evaluate a sensitivity function model
    for a QSO spectrum.

    Parameters
    ----------
    theta : `numpy.ndarray`_
        Array containing the PCA coefficients
        shape=(ntheta,)

    obj_dict : dict
       Dictionary containing additional arguments needed to evaluate the PCA model

    Returns
    -------
    qso_pca_model : array with same shape as the PCA vectors (stored in the obj_dict['pca_dict'])
       PCA vectors were already interpolated onto the telluric model grid by init_qso_model

    gpm : `numpy.ndarray`_ : array with same shape as the qso_pca_model
       Good pixel mask indicating where the model is valid

    """

    qso_pca_model = qso_pca_eval(theta, obj_dict['pca_dict'])
    # TODO Is the prior evaluation slowing things down??
    # TODO Disablingthe prior for now as I think it slows things down for no big gain
    #ln_pca_pri = qso_pca.pca_lnprior(theta_PCA, arg_dict['pca_dict'])
    #ln_pca_pri = 0.0
    #flux_model, tell_model, spec_model, modelmask
    return qso_pca_model, (qso_pca_model > 0.0)


##############
# Star Model #
##############
def init_star_model(obj_params, iord, wave, flux, ivar, mask, tellmodel):
    """

    Routine used to initialize the star spectrum model for telluric fits. The
    star model is the true spectrum of the standard star times a polynomial to
    accomodate situations where the star-model is not perfect.

    Parameters
    ----------
    obj_params : dict
        Dictionary containing parameters necessary for initializing the quasar model

    iord : int
        Order in question. This is used here because each echelle order can  have a different polynomial order

    wave : array shape (nspec,)
        Wavelength array for the object in question

    flux : array shape (nspec,)
        Flux array for the object in question

    ivar : array shape (nspec,)
        Inverse variance array for the oejct in question

    mask : array shape (nspec,)
        Good pixel mask for the object in question

    tellmodel : array shape (nspec,)
        This is a telluric model computed on the wave wavelength grid. Initialization usually requires some initial
        best guess for the telluric absorption, which is computed from the midpoint of the telluric model grid parameter
        space using the resolution of the spectrograph and the airmass of the observations.

    Returns
    -------
    obj_dict : dict
        Dictionary containing the meta-information and variables that are used for the object model evaluations.

    bounds_obj : tuple
        Tuple of bounds for each parameter that will be fit for the object model, which are here the polynomial
        coefficients.


    """

    # Model parameter guess for starting the optimizations
    flam_true = scipy.interpolate.interp1d(obj_params['std_dict']['wave'].value,
                                           obj_params['std_dict']['flux'].value, kind='linear',
                                           bounds_error=False, fill_value=np.nan)(wave)
    flam_model = flam_true*tellmodel
    flam_model_ivar = (100.0*utils.inverse(flam_model))**2 # This is just a bogus noise to give  S/N of 100
    flam_model_mask = np.isfinite(flam_model)
    # As solve_poly_ratio is designed to multiply a scale factor into the flux, and not the flux_ref, we
    # set the flux_ref to be the data here, i.e. flux
    scale, fit_tuple, flux_scale, ivar_scale, outmask = coadd.solve_poly_ratio(
        wave, flam_model, flam_model_ivar, flux, ivar, obj_params['polyorder_vec'][iord],
        mask=flam_model_mask, mask_ref=mask, func=obj_params['func'], model=obj_params['model'])

    coeff, wave_min, wave_max = fit_tuple
    if(wave_min != wave.min()) or (wave_max != wave.max()):
        msgs.error('Problem with the wave_min or wave_max')
    # Polynomial coefficient bounds
    bounds_obj = [(np.fmin(np.abs(this_coeff)*obj_params['delta_coeff_bounds'][0], obj_params['minmax_coeff_bounds'][0]),
                   np.fmax(np.abs(this_coeff)*obj_params['delta_coeff_bounds'][1], obj_params['minmax_coeff_bounds'][1]))
                   for this_coeff in coeff]
    # Create the obj_dict
    obj_dict = dict(wave=wave, wave_min=wave_min, wave_max=wave_max, flam_true=flam_true, func=obj_params['func'],
                    model=obj_params['model'], polyorder=obj_params['polyorder_vec'][iord])

    if obj_params['debug']:
        plt.plot(wave, flux, drawstyle='steps-mid', alpha=0.7, zorder=5, label='star spectrum')
        plt.plot(wave, flux_scale, drawstyle='steps-mid', alpha=0.7, zorder=4, label='poly_model*star_model*telluric')
        plt.plot(wave, flam_model, label='star_model*telluric')
        plt.plot(wave, flam_true, label='star_model')
        plt.ylim(-0.1 * flam_model.min(), 1.3 * flam_model.max())
        plt.legend()
        plt.title('Sensitivity Function Guess for iord={:d}'.format(iord+1))  # +1 to account 0-index starting
        plt.show()


    return obj_dict, bounds_obj

# Star evaluation function.
def eval_star_model(theta, obj_dict):
    """
    Routine to evaluate a star spectrum model as a true model spectrum times a polynomial.

    Parameters
    ----------
    theta : `numpy.ndarray`_
        Array containing the polynomial coefficients.
        shape (ntheta,)

    obj_dict : dict
       Dictionary containing additional arguments needed to evaluate the star model.

    Returns
    -------
    star_model : `numpy.ndarray`_
        PCA vectors were already interpolated onto the telluric model grid by init_qso_model.
        array with same shape obj_dict['wave']

    gpm : `numpy.ndarray`_
        Good pixel mask indicating where the model is valid.
        array with same shape as the star_model

    """

    wave_star = obj_dict['wave']
    wave_min = obj_dict['wave_min']
    wave_max = obj_dict['wave_max']
    flam_true = obj_dict['flam_true']
    func = obj_dict['func']
    model = obj_dict['model']
    ymult = coadd.poly_model_eval(theta, func, model, wave_star, wave_min, wave_max)
    star_model = ymult*flam_true

    return star_model, (star_model > 0.0)


####################
# Polynomial Model #
####################
def init_poly_model(obj_params, iord, wave, flux, ivar, mask, tellmodel):
    """
    Routine used to initialize a polynomial object model for telluric fits.

    Parameters
    ----------
    obj_params : dict
        Dictionary containing parameters necessary for initializing the quasar model

    iord : int
        Order in question. This is used here because each echelle order can have a different polynomial order

    wave : array shape (nspec,)
        Wavelength array for the object in question

    flux : array shape (nspec,)
        Flux array for the object in question

    ivar : array shape (nspec,)
        Inverse variance array for the oejct in question

    mask : array shape (nspec,)
        Good pixel mask for the object in question

    tellmodel : array shape (nspec,)
        This is a telluric model computed on the wave wavelength grid. Initialization usually requires some initial
        best guess for the telluric absorption, which is computed from the midpoint of the telluric model grid parameter
        space using the resolution of the spectrograph and the airmass of the observations.

    Returns
    -------
    obj_dict : dict
        Dictionary containing the meta-information and variables that are used for the object model evaluations.

    bounds_obj : tuple
        Tuple of bounds for each parameter that will be fit for the object model, which are here the polynomial
        coefficients.


    """

    tellmodel_ivar = (100.0*utils.inverse(tellmodel))**2 # This is just a bogus noise to give  S/N of 100
    tellmodel_mask = np.isfinite(tellmodel) & mask

    if obj_params['mask_lyman_a']:
        mask = mask & (wave>1216.15*(1+obj_params['z_obj']))

    # As solve_poly_ratio is designed to multiply a scale factor into the flux, and not the flux_ref, we
    # set the flux_ref to be the data here, i.e. flux
    scale, fit_tuple, flux_scale, ivar_scale, outmask = coadd.solve_poly_ratio(
        wave, tellmodel, tellmodel_ivar, flux, ivar, obj_params['polyorder_vec'][iord],
        mask=tellmodel_mask, mask_ref=mask, func=obj_params['func'], model=obj_params['model'], scale_max=1e5)
    # TODO JFH Sticky = False seems to recover better from bad initial fits. Maybe we should change this since poly ratio
    # uses a different optimizer.

    coeff, wave_min, wave_max = fit_tuple
    if(wave_min != wave.min()) or (wave_max != wave.max()):
        msgs.error('Problem with the wave_min or wave_max')
    # Polynomial model
    polymodel = coadd.poly_model_eval(coeff, obj_params['func'], obj_params['model'], wave, wave_min, wave_max)

    # Polynomial coefficient bounds
    bounds_obj = [(np.fmin(np.abs(this_coeff)*obj_params['delta_coeff_bounds'][0], obj_params['minmax_coeff_bounds'][0]),
                   np.fmax(np.abs(this_coeff)*obj_params['delta_coeff_bounds'][1], obj_params['minmax_coeff_bounds'][1]))
                   for this_coeff in coeff]
    # Create the obj_dict
    obj_dict = dict(wave=wave, wave_min=wave_min, wave_max=wave_max, polymodel=polymodel, func=obj_params['func'],
                    model=obj_params['model'], polyorder=obj_params['polyorder_vec'][iord])

    if obj_params['debug']:
        plt.plot(wave, flux, drawstyle='steps-mid', alpha=0.7, zorder=5, label='observed spectrum')
        plt.plot(wave, flux_scale, drawstyle='steps-mid', alpha=0.7, zorder=4, label='poly_model*telluric')
        plt.plot(wave, tellmodel, label='telluric')
        plt.plot(wave, polymodel, label='poly_model')
        plt.xlim(wave[mask].min(), wave[mask].max())
        plt.ylim(-0.3 * flux[mask].min(), 1.3 * flux[mask].max())
        plt.legend()
        plt.title('Sensitivity Function Guess for iord={:d}'.format(iord + 1))   # +1 to account 0-index starting
        plt.show()

    return obj_dict, bounds_obj

# Polynomial evaluation function.
def eval_poly_model(theta, obj_dict):
    """
    Routine to evaluate a star spectrum model as a true 
    model spectrum times a polynomial.

    Parameters
    ----------
    theta : `numpy.ndarray`_
        Array containing the polynomial coefficients.
        shape=(ntheta,)

    obj_dict : dict
       Dictionary containing additional arguments needed to evaluate the star model.

    Returns
    -------
    star_model : `numpy.ndarray`_
        array with same shape obj_dict['polymodel']

    gpm : `numpy.ndarray`_
        Good pixel mask indicating where the model is valid.
        array with same shape as the star_model.

    """
    polymodel = coadd.poly_model_eval(theta, obj_dict['func'], obj_dict['model'],
                                      obj_dict['wave'], obj_dict['wave_min'], obj_dict['wave_max'])

    return polymodel, (polymodel > 0.0)

