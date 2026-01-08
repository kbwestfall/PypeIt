from IPython import embed

from astropy import table
import numpy as np

from pypeit import dataPaths
from pypeit import log
from pypeit import PypeItError
from pypeit import utils
from pypeit.core import coadd
from pypeit.core import standard
from pypeit.core import spectrum


class AdjustedSpectrumModel:
    """
    A model consisting of an underlying spectrum multiplied by a polynomial.

    This is the base model where the spectrum must be provided directly.  Other
    subclasses below build the spectrum as part of the instantiation of the
    model.

    The underlying spectrum can optionally depend on a set of parameters.  If it
    does, the subclasses must provide the following functions:

        - :func:`~pypeit.telluric.object.AdjustedSpectrumModel.spectrum_par_guess`:
          Generate guess parameters for the underlying object spectrum.  Note
          that this does *not* take any arguments; i.e., the guess parameters
          are always the same!

        - :func:`~pypeit.telluric.object.AdjustedSpectrumModel.spectrum_par_bounds`:
          Generate lower and upper parameter boundaries for the underlying
          object spectrum.  Note that this does *not* take any arguments; i.e.,
          the parameter bounds are always the same!

        - :func:`~pypeit.telluric.object.AdjustedSpectrumModel.spectrum_sample`:
          Sample the underlying object spectrum given a set of parameters.

    The order of the multiplicative polynomial must be defined at instantiation.
    If ``order is None``, no polynomial is included in the model.  Note that
    ``order == 0`` includes a constant normalization of the model spectrum,
    which will *not* be included if ``order is None``.

    .. warning::

        The "model" can have 0 parameters if the underlying spectrum requires no
        parameters and the order is set to None.

    Parameters
    ----------
    spec : :class:`~pypeit.core.spectrum.Spectrum`
        Underlying spectrum for the model.
    wave : `numpy.ndarray`, optional
        If provided, resample the provided spectrum to this wavelength grid.
        Unobserved regions in the spectrum will be masked.  If None, the
        wavelength grid used is the same as provided by the input spectrum.
    func : str, optional
        Type of multiplicative polynomial to use.
    model : str, optional
        Operator to use on the polynomial *before* multiplying it with the
        spectrum.  Must be 'poly', 'exp', or 'square'.
    order : int, optional
        Order of the polynomial to include.  Can be None; see description above.

    Attributes
    ----------
    spec : :class:`~pypeit.core.spectrum.Spectrum`
        Underlying spectrum for the model.
    func : str
        Type of multiplicative polynomial to use.
    model : str
        Operator to use on the polynomial *before* multiplying it with the
        spectrum.  Must be 'poly', 'exp', or 'square'.
    order : int
        Order of the polynomial to include.  No polynomial is included if order
        is None.
    wave_min : float
        Minimum wavelength of the spectrum.
    wave_max : float
        Maximum wavelength of the spectrum.
    spec_npar : int
        Number of parameters required to generate the underlying spectrum.
    """
    # TODO: Do I need to allow order to be a vector (listing the orders to include)?
    def __init__(self, spec, wave=None, func='legendre', model='exp', order=None):
        # TODO: Add parameters that can be passed to resample?
        self.spec = spec if wave is None else spec.resample(wave)
        self.func = func
        self.model = model
        self.order = order
        self.wave_min = np.min(self.spec.wave)
        self.wave_max = np.max(self.spec.wave)
        self.spec_npar = 0

    @property
    def wave(self):
        """The wavelength of the spectrum."""
        return self.spec.wave
    
    @property
    def npar(self):
        """
        The total number of parameters in the model.
        """
        return self.spec_npar if self.order is None else self.spec_npar + self.order + 1

    def spectrum_par_guess(self):
        """
        Provide guess parameters for the underlying spectrum.

        Returns
        -------
        `numpy.ndarray` or None
            Parameters for the underlying spectrum.  If None, the underlying
            spectrum has no parameters.
        """
        return None
    
    def par_guess(self, obs_spec):
        """
        Provide initial guess parameters for a fit to the provided spectrum.

        The guess parameters include those for the spectrum model and the
        polynomial.  If :attr:`order` is 0, the model includes a constant
        normalization of the spectrum and the guess value is determined by
        :func:`~pypeit.core.coadd.robust_median_ratio`, where the reference
        spectrum is based on the guess spectrum model.  If :attr:`order` is
        larger than 0, the guess parameters are instead determined by
        :func:`~pypeit.core.coadd.solve_poly_ratio`.

        Parameters
        ----------
        obs_spec : :class:`~pypeit.core.spectrum.Spectrum`
            Observed spectrum to fit

        Returns
        -------
        `numpy.ndarray` or None
            Guess parameters for the model.  The first :attr:`spec_npar` entries
            are the parameters for the underlying spectrum, and the remaining
            ``order+1`` parameters are for the polynomial.
        """
        # The wavelengths must match
        if not np.allclose(self.spec.wave, obs_spec.wave):
            raise PypeItError('Model spectrum must have the same wavelength vector as the data.')

        # Get the guess parameters for the underlying spectrum 
        gsp = self.spectrum_par_guess()

        # No polynomial is included so we're done.  NOTE: This can return None
        # if there are no spectrum guess parameters.
        if self.order is None:
            return gsp

        # Get the model spectrum and adopt a S/N = 100 for determining the
        # parameters
        spec_flux, spec_gpm = self.spectrum_sample(gsp)
        spec_ivar = utils.inverse((spec_flux/100.0)**2)

        if self.order == 0:
            # Just guess a normalization factor
            norm = 1.0/coadd.robust_median_ratio(
                obs_spec.flux, obs_spec.ivar, spec_flux, spec_ivar, mask=obs_spec.gpm,
                mask_ref=spec_gpm
            )
            return np.array([norm]) if gsp is None else np.append(gsp, [norm])

        # Guess the polynomial coefficients
        # NOTE: This uses the observed spectrum as the "reference" spectrum,
        # despite the docstring for the function suggesting that the higher S/N
        # spectrum should be used.
        # TODO: Add the `debug` option?
        _, fit_tuple, _, _, _ = coadd.solve_poly_ratio(
            obs_spec.wave, spec_flux, spec_ivar, obs_spec.flux, obs_spec.ivar, self.order,
            mask=spec_gpm, mask_ref=obs_spec.gpm, func=self.func, model=self.model,
            scale_max=1e5, debug=True
        )

        return np.asarray(fit_tuple[0]) if gsp is None else np.append(gsp, fit_tuple[0])
    
    def spectrum_par_bounds(self):
        """
        Provide the bounds for spectrum-specific parameters.

        Returns
        -------
        list or None
            List of two-tuples with the lower and upper boundaries for the
            spectrum-specific parameters.
        """
        return None

    def par_bounds(self, guess_par, rel_coeff_bounds, abs_coeff_bounds):
        """
        Provide the bounds for all object model parameters.

        The bounds are calculated in an absolute sense (using
        ``abs_coeff_bounds``) and relative to the guess parameters (using
        ``rel_coeff_bounds``); the bound providing the larger range is used.
        For example, if the guess parameter is 2.0 and the relative lower bound
        is 0.5, the lower bound will be set to 1.0, unless the absolute bound is
        smaller.

        Parameters
        ----------
        guess_par : `numpy.ndarray`_
            Guess parameters for the model.
        rel_coeff_bounds : tuple
            The lower and upper boundary of each coefficient relative to the
            value of the guess value.  For example, (0.5, 2,0) means that every
            coefficient must be within a factor of 2 of the guess value.   These
            limits are used for *all* coefficients.  See description above.
        abs_coeff_bounds : tuple
            The absolute lower and upper boundaries for the coefficients; i.e.,
            this is *not* relative to the guess value.  These limits are used
            for *all* coefficients.  See description above.

        Returns
        -------
        list
            List of two-tuples with the lower and upper boundary for each model
            parameter.
        """
        # Get the bounds for the spectrum parameters
        bnd = self.spectrum_par_bounds()
        # Check that the number of bounds make sense
        if bnd is not None and len(bnd) != self.spec_npar:
            raise PypeItError(
                f'Incorrect number of spectrum parameter bounds found:  Got {len(bnd)}, expected '
                f'{self.spec_npar}.'
            )

        # Check that the number of guess parameters is correct
        if len(guess_par) != self.npar:
            raise PypeItError(
                f'Incorrect number of guess parameters provided: Got {len(guess_par)}, expected '
                f'{self.npar}.'
            )

        if bnd is None:
            # There are no spectrum parameters so instantiate the list
            bnd = []

        # Include the coefficient bounds
        for i in range(self.order + 1):
            bnd += [(
                min(np.absolute(guess_par[i+self.spec_npar])*rel_coeff_bounds[0],
                    abs_coeff_bounds[0]),
                max(np.absolute(guess_par[i+self.spec_npar])*rel_coeff_bounds[1],
                    abs_coeff_bounds[1]),
            )]

        return bnd

    def spectrum_sample(self, theta):
        """
        Return the model spectrum before any modifications by the polynomial.

        .. note::

            - This function should *not* affect the overall normalization of the
              spectrum.  That *must* be handled by the polynomial.

            - In this base class, the spectrum has no parameters and ``theta``
              is ignored.

        Parameters
        ----------
        theta : `numpy.ndarray`
            Parameters required to generate the model spectrum.  This *should
            not* include any of the polynomial coefficients. 

        Returns
        -------
        flux : `numpy.ndarray`
            Flux of the model spectrum.
        gpm : `numpy.ndarray`
            Good pixel mask of the model spectrum.
        """
        if theta is not None:
            raise PypeItError(
                f'Parameter vector must be None for {self.__class__.__name__} since the '
                f'underlying spectrum has no parameters.'
            )
        return self.spec.flux.copy(), self.spec.gpm.copy()

    def sample(self, theta):
        """
        Sample the full model spectrum, including the underlying spectrum and
        the multiplicative polynomial.

        Parameters
        ----------
        theta : `numpy.ndarray`
            The full parameter vector required to generate the model spectrum.
            The first :attr:`spec_npar` parameters are used to generate the
            underlying spectrum, and the remainder are treated as coefficients
            of the polynomial.  This can be None as long as the
            :func:`spectrum_sample` method of the class can handle it.

        Returns
        -------
        flux : `numpy.ndarray`
            Flux of the model spectrum.
        gpm : `numpy.ndarray`
            Good pixel mask of the model spectrum.
        """
        if theta is None and self.npar > 0:
            raise PypeItError(
                f'Parameter vector cannot be None for {self.__class__.__name__} since the model '
                f'requires {self.npar} parameters.'
            )
        if theta is not None and theta.size != self.npar:
            raise PypeItError(
                f'Incorrect number of parameters for {self.__class__.__name__}:  Got '
                f'{theta.size}, expected {self.npar}.'
            )
        # Get the base-level spectrum
        if theta is None or self.spec_npar == 0:
            model_flux, model_gpm = self.spectrum_sample(None)
        elif self.spec_npar > 0:
            model_flux, model_gpm = self.spectrum_sample(theta[:self.spec_npar])
        else:
            raise PypeItError('Unable to compute underlying spectrum.')

        # Add the polynomial
        # TODO: Force the polynomial to always be positive?
        if theta is not None and theta.size > self.spec_npar:
            model_flux *= coadd.poly_model_eval(
                theta[self.spec_npar:], self.func, self.model, self.wave, self.wave_min,
                self.wave_max
            )

        # Return the adjusted spectrum and mask
        return model_flux, (model_flux > 0.0) & model_gpm


class QSOPCAModel(AdjustedSpectrumModel):
    r"""
    A QSO spectrum model based on a PCA decomposition.

    The model parameters for the base level QSO spectrum are the redshift and
    the :math:`N_{\rm PCA}-1` coefficients; the coefficient for the first PCA
    component is always set to 1.  For an overall normalization of the model,
    set ``order`` to 0 or larger.

    The attributes provided below are specific to this class; see the
    description of the base class for additional attributes.

    Parameters
    ----------
    filename : str
        A local file or a QSO PCA model file provided by PypeIt.
    z : float
        The fiducial redshift of the QSO model.  The model parameters include
        the redshift, as well.  This should be a redshift used to approximately
        match the observed wavelength range of the spectrum being modeled.
    dz : float, optional
        The :math:`\pm` range relative to the provided redshift (`z`) that is
        used to set the bounds during the modeling process.
    npca : int, optional
        The number of PCA components to use in constructing the model.  A
        warning will be issued if this is larger than the number of components
        available in the data provided by ``filename``.
    kwargs : dict, optional
        Passed directly to the instantiation of the base class.  I.e., these are
        the parameters used to define the multiplicative polynomial.

    Attributes
    ----------
    coeffs : `numpy.ndarray`_
        Table of coefficients determined for the spectra used to build the PCA
        decomposition.  Shape is the number of spectra by the number of PCA
        components.
    z : float
        The fiducial redshift of the QSO model.
    dz : float
        The :math:`\pm` range of the redshift allowed during the model fit.
    npca : int
        Number of PCA components
    spec_gpm : `numpy.ndarray`_
        The good pixel mask for the spectrum model.
    dloglam : float
        The median change in log(wavelength) of the model spectrum.
    """
    def __init__(self, filename, z, dz=0.1, npca=None, **kwargs):

        file = dataPaths.tel_model.get_file_path(filename)
        tbl = table.Table.read(file)

        wave = np.squeeze(tbl['WAVE_PCA'])
        if z is not None:
            wave *= (1 + z)

        components = np.squeeze(tbl['PCA_COMP'])

        _npca = npca
        if _npca is None:
            _npca = components.shape[0]
        if _npca > components.shape[0]:
            log.warning(
                f'Number of requested PCA components ({_npca}) for QSO model is larger than '
                f'the number available ({components.shape[0]}).  Using all PCA components.'
            )
            _npca = components.shape[0]

        # Instantiate the base class
        # TODO: Pass the file name to the metadata of the spectrum?
        # NOTE: The transpose is used because the Spectrum object expects the
        # spectra to be organized along the first axis.
        super().__init__(spectrum.Spectrum(wave, components[:_npca,:].T), **kwargs)

        # Set the number of parameters
        self.npca = _npca
        self.spec_npar = self.npca # redshift + npca-1

        # Save the coefficients in the table to use for setting the parameters
        # boundaries.
        self.coeffs = np.squeeze(tbl['PCA_COEFFS'])[:,:self.npca]
        # TODO: Get the bounds right away?

        # Set the redshift and the redshift range
        self.z = z
        self.dz = dz

        # Calculate quantities that only need to be calculated once to construct
        # the spectrum    
        # NOTE: This gpm will always be the same, independent of the PCA
        # coefficients
        self.spec_gpm = np.any(self.spec.gpm, axis=1)
        self.dloglam = np.median(np.diff(np.log10(self.spec.wave)))

    def spectrum_par_guess(self):
        """
        Provide guess parameters for the underlying spectrum.

        Returns
        -------
        `numpy.ndarray`
            Parameters for the underlying spectrum.  If None, the underlying
            spectrum has no parameters.
        """
        return np.append([self.z], np.zeros(self.npca-1, dtype=float))
    
    def spectrum_par_bounds(self):
        """
        Set the boundaries for the model parameters.

        Returns
        -------
        list
            A list of two-tuples where each tuple sets the upper and lower
            boundary on each parameter.
        """
        # Redshift bounds
        bounds = [(self.z-self.dz, self.z+self.dz)]
        # Coefficient bounds
        bounds += [tuple(bnds) for bnds in zip(
            np.min(self.coeffs[:,1:], axis=0),
            np.max(self.coeffs[:,1:], axis=0)
        )]
        return bounds
    
    def spectrum_sample(self, theta):
        r"""
        Sample the QSO model spectrum.

        Parameters
        ----------
        theta : `numpy.ndarray`_
            Parameter vector.  The parameters are the redshift and the
            :math:`N_{\rm PCA}-1` coefficients; the coefficient for the first
            PCA component is always set to 1.

        Returns
        -------
        `numpy.ndarray`_
            The model QSO spectrum
        """
        if theta is None:
            raise PypeItError(f'Parameter vector cannot be None for {self.__class__.__name__}')
        if theta.size != self.spec_npar:
            raise PypeItError(
                f'Incorrect number of parameters for {self.__class__.__name__}:  Got '
                f'{theta.size}, expected {self.spec_npar}.'
            )

        # TODO:
        #   - Allow for subpixel shifts
        #   - Mask regions unobserved and wrapped spectral regions
        # NOTE: this dot product raises a ValueError when the number of
        # parameters is incorrect
        _flux = np.dot(self.spec.flux, np.append(1.0,theta[1:]))
        dshift = int(np.round(np.log10((1.0 + theta[0])/(1.0 + self.z))/self.dloglam))
        gpm = np.roll(self.spec_gpm, dshift)
        gpm[np.s_[:dshift] if dshift >= 0 else np.s_[dshift:]] = False
        return np.exp(np.roll(_flux, dshift)), gpm


class StellarSpectrumModel(AdjustedSpectrumModel):
    """
    A stellar spectrum model.

    This is a simple wrapper that builds the standard star spectrum and
    instantiates the :class:`AdjustedSpectrumModel` base class.

    For the parameters, see :func:`pypeit.core.standard.get_standard_spectrum`.
    The remaining ``kwargs`` are passed directly to the instantiation of the
    base class.
    """
    def __init__(self, spectral_type=None, V_mag=None, ra=None, dec=None, tol=20.,
                 archives='default', **kwargs):
        spec = standard.get_standard_spectrum(
            spectral_type=spectral_type, V_mag=V_mag, ra=ra, dec=dec, tol=tol, archives=archives
        )
        super().__init__(spec, **kwargs)


# TODO: It's wasteful to generate a unity spectrum to then multiply it by a
# polynomial.  Consider a better solution, like allowing the spec attribute of
# the AdjustedSpectrumModel to be None.
class PolynomialModel(AdjustedSpectrumModel):
    """
    A model consisting of only a polynomial.

    It uses :class:`~pypeit.core.standard.PseudoStandard` as the underlying
    spectrum.  See :class:`AdjustedSpectrumModel` for the parameters; the
    keyword arguments are passed directly to the base class.
    """
    def __init__(self, **kwargs):
        # If wave is provided, use it to generate the PseudoStandard instead of
        # its default wavelengths.
        wave = kwargs.pop('wave') if 'wave' in kwargs else None
        spec = standard.PseudoStandard(wave=wave)
        super().__init__(spec, **kwargs)
