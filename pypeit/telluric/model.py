"""
Module providing classes for telluric models.

.. include common links, assuming primary doc root is up one directory
.. include:: ../include/links.rst
"""

from IPython import embed
import numpy as np
from scipy import signal

from pypeit import dataPaths
from pypeit import log
from pypeit import PypeItError
from pypeit import __version__
from pypeit import io
from pypeit.core.wavecal import wvutils


class TelluricModel:
    r"""
    Base class for all telluric models.

    The subclasses must provide the following functions:

        - :func:`~pypeit.telluric.model.TelluricModel.load`: Load the telluric
          data and perform the remaining instantiation steps.
        - :func:`~pypeit.telluric.model.TelluricModel.base_par_guess`: Generate
          guess parameters for the underlying telluric spectrum.
        - :func:`~pypeit.telluric.model.TelluricModel.base_par_bounds`: Generate
          lower and upper parameter boundaries for the underlying telluric
          spectrum.
        - :func:`~pypeit.telluric.model.TelluricModel.base_sample`: Sample the
          underlying telluric spectrum given a set of parameters.

    Parameters
    ----------
    filename : :obj:`str`
        File with the telluric model data
    wave_min : :obj:`float`, optional
        Minimum wavelength at which the model is desired
    wave_max : :obj:`float`, optional
        Maximum wavelength at which the model is desired.
    pad_frac : :obj:`float`, optional
        Percentage padding to be added to the model boundaries if ``wave_min`` or
        ``wave_max`` are input; ignored otherwise. The resulting grid will
        extend from ``(1.0 - pad_frac)*wave_min`` to ``(1.0 +
        pad_frac)*wave_max``.
    model_res : :obj:`float`, optional
        Spectral resolution (:math:`R = \lambda / \Delta\lambda`) of the model
        spectra.  The resolution is expected to be constant as a function of
        wavelength.
    load : :obj:`bool`, optional
        Flag to load the data upon instantiation.
    """
    def __init__(
        self, filename, wave_min=None, wave_max=None, pad_frac=0.1, model_res=None, load=True
    ):
        to_pkg = 'move' if ".dev" in __version__ else None
        self.file = dataPaths.telgrid.get_file_path(filename, to_pkg=to_pkg)
        self.wave_min = wave_min
        self.wave_max = wave_max
        self.pad_frac = pad_frac
        self.model_res = model_res

        # Defined by the subclass load functions
        self.wave_grid = None
        self.tell_grid = None
        self.dloglam = None
        self.tell_pad_pix = None

        # The number of parameters needed to generate the underlying telluric
        # model (everything except the resolution, shift, and stretch)
        self.base_npar = None
        # The total number of parameters for the model.  This is always
        # base_npar + 3, but we haven't defined base_npar yet.  It should always
        # be set by the load() function.
        self.npar = None

        # Load the data
        if load:
            self.load()

    def load(self):
        """
        Load the telluric data from the reference file.
        """
        raise PypeItError(f'{self.__class__.__name__} has not defined a load function!')
    
    def _finalize_wave_grid(self, wave_grid_full, model_grid_full):
        r"""
        Provided the raw telluric data, limit the wavelength range to the
        provided minimum (:attr:`wave_min`), maximum (:attr:`wave_max`), and
        padding (:attr:`pad_frac`).

        Parameters
        ----------
        wave_grid_full : `numpy.ndarray`_
            Full wavelength array.  Expected to be 1D.
        model_grid_full : `numpy.ndarray`_
            Full model grid.  The number of pixels per spectrum should match the
            wavelength grid, and they are expected to be organized along the
            last axis of the array.

        Returns
        -------
        wave_grid : `numpy.ndarray`_
            The final wavelength grid.
        model_grid : `numpy.ndarray`_
            The final model grid.
        dloglam : :obj:`float`
            The step in log(wavelength) for each pixel; see
            :func:`~pypeit.core.wavecal.wvutils.get_sampling`.
        tell_pad_px : :obj:`int`
            Number of pixels to pad the models needed to improve accuracy of the
            convolution used to match the model to the resolution of the
            observed spectrum.  The default is to pad 10 times the sigma of the
            resolution element, assuming there are 3 pixels per resolution
            element.
        """
        # Complete the instantiation
        nspec_full = wave_grid_full.size

        if nspec_full != model_grid_full.shape[-1]:
            raise PypeItError(
                'Model telluric grid and wavelength grid have different numbers of spectral '
                'pixels.'
            )

        # Get the new pixel ranges
        ind_lower = np.argmin(np.abs(wave_grid_full - (1.0 - self.pad_frac)*self.wave_min)) \
                        if self.wave_min is not None else 0
        ind_upper = np.argmin(np.abs(wave_grid_full - (1.0 + self.pad_frac)*self.wave_max)) \
                        if self.wave_max is not None else nspec_full

        # Get the grid subsections
        wave_grid = wave_grid_full[ind_lower:ind_upper]
        model_grid = model_grid_full[...,ind_lower:ind_upper]

        # Determine the sampling, padding, and return
        # TODO: Why do we need pix per sigma here?  We're always returning 10
        # sigma, and we're always assuming there are three pixels per resolution
        # element.  Isn't this always going to be 13 pixels?
        _, dloglam, _, pix_per_sigma = wvutils.get_sampling(wave_grid)
        if self.dloglam == 0.0:
            raise PypeItError(
                'The telluric model grid cannot have a log wavelength spacing that is 0!'
            )
        return wave_grid, model_grid, dloglam, int(np.ceil(10.0 * pix_per_sigma))

    # TODO: This should account for the current resolution of the model...
    def _convolve(self, tspec, res):
        """
        Convolve the telluric model to a desired resolution.

        Parameters
        ----------
        tspec : `numpy.ndarray`_
            1D transmission spectrum for the telluric model at its native
            resolution.
        res : :obj:`float`
            Desired resolution expressed as lambda/dlambda. Note that here
            dlambda is linear, whereas dloglam is the delta of the log10.

        Returns
        -------
        `numpy.ndarray`_
            Convolved transmission spectrum, with a shape that matches the input
            model.
        """
        # Check the input resolution
        if res <= 0.0:
            raise PypeItError('Resolution must be positive.')

        # Factor converting sigma to FWHM (~2.35)
        sig2fwhm = np.sqrt(8. * np.log(2.))

        # Compute the sigma of the Gaussian kernel in pixels
        dres = 1./res if self.model_res is None else np.sqrt(1.0/res**2 - 1.0/self.model_res**2)
        sigma = dres / sig2fwhm / self.dloglam / np.log(10.)

        # Require the sigma to be at least 1 pixel (previous version used 0.5 pix)
        # TODO: We can make this significantly smaller if we use the analytic
        # FFT of a Gaussian to perform the convolution.  I'm inclined to add a
        # dependency on ppxf.
        if sigma < 1.0:
            log.warning(
                'Gaussian sigma to change resolution of telluric model is less than 1 pixel.  '
                'Skipping resolution matching.  We recommend using/creating higher resolution '
                'telluric models!'
            )
            return tspec
        
        # x = loglam/sigma on the wavelength grid from -4 to 4, symmetric, centered about zero.
        # g = Gaussian evaluated over x
        samp = np.arange(0,4,1.0 / sigma)
        g = np.exp(-0.5*np.append(-samp[:0:-1], samp)**2)
        # Convolve with a normalized Gaussian kernel
        # TODO: Aim to make this faster
        return signal.convolve(tspec, g/g.sum(), mode='same')

    # TODO: Shift is not independent of scale, and we should be using resampling
    # not interpolation.
    def _shift_and_stretch(self, loglam, tspec, shift, stretch):
        """
        Shift and scale the transmission spectrum of the telluric model via
        interpolation.

        Parameters
        ---------- 
        loglam : `numpy.ndarray`_
            Base-10 log of the wavelength coordinate of each pixel.
        tspec : `numpy.ndarray`_
            Transmission spectrum for the telluric model.
        shift : :obj:`float`
            Shift to apply in pixels (can be sub-pixel).
        stretch : :obj:`float`
            Stretch to apply.

        Returns
        -------
        `numpy.ndarray`_
            Shifted telluric model. Shape = same size as input tell_model.
        """
        loglam_shift = loglam[0] + shift * self.dloglam \
            + np.arange(len(loglam)) * self.dloglam * stretch
        return np.interp(loglam_shift, loglam, tspec)

    def base_par_guess(self):
        """
        Generate a first-guess for the model parameters specific to the
        construction of the base-level spectral model.

        Returns
        -------
        `numpy.ndarray` 
            Guess model parameters.
        """
        raise PypeItError(f'{self.__class__.__name__} has not defined a base_par_guess function!')
    
    def par_guess(self, obs_spec):
        """
        Generate a first-guess for the model parameters.

        Parameters
        ----------
        obs_spec : :class:`~pypeit.core.spectrum.Spectrum`
            Observed spectrum to be fit.  The wavelength vector is used to
            estimate the resolution; see
            :func:`~pypeit.core.wavecal.wvutils.get_sampling`.

        Returns
        -------
        `numpy.ndarray`_
            Guess model parameters including the resolution, shift, and stretch
            parameters.  The shift guess is always 0 pixels, and the stretch
            guess is always 1.0 (i.e., no stretch).
        """
        # TODO: Need to check that obs_spec.wave is the right thing to pass
        # here...
        resolution_guess = wvutils.get_sampling(obs_spec.wave)[2]
        return np.append(self.base_par_guess(), [resolution_guess, 0.0, 1.0])

    def base_par_bounds(self):
        """
        Generate the parameter bounds for the base-level spectral model.

        Returns
        -------
        list
            List of tuples with the lower and upper bounds of the parameters for
            the base-level model.
        """
        raise PypeItError(f'{self.__class__.__name__} has not defined a base_par_bounds function!')
    
    def par_bounds(
        self, guess_par, resolution_frac_bounds=(0.3, 1.5), pix_shift_bounds=(-5.0,5.0),
        pix_stretch_bounds=(0.98,1.02)
    ):
        """
        Set the boundaries for the model parameters.

        Parameters
        ----------
        guess_par : `numpy.ndarray`_
            Guess model parameters including the resolution, shift, and stretch
            parameters.  The shift guess is always 0 pixels, and the stretch
            guess is always 1.0 (i.e., no stretch).
        resolution_frac_bounds : :obj:`tuple`, optional
            Lower and upper bounds for the spectral resolution expressed as a
            fraction of the guessed resolution.
        pix_shift_bounds : :obj:`tuple`, optional
            Lower and upper bounds for the pixel shift.
        pix_stretch_bounds : :obj:`tuple`, optional
            Lower and upper bounds for the pixel stretch.

        Returns
        -------
        list
            A list of tuples that provide the lower and upper bounds for each
            model parameter.
        """
        # guess_par[-3] is the guess resolution.
        return self.base_par_bounds() + [
            tuple(guess_par[-3] * np.asarray(resolution_frac_bounds)),
            pix_shift_bounds,
            pix_stretch_bounds,
        ]

    def base_sample(self):
        """
        Sample the telluric model at its native resolution and wavelength grid.
        """
        raise PypeItError(f'{self.__class__.__name__} has not defined a base_sample function!')

    def sample(self, theta, start=0, end=None):
        r"""
        Evaluate the telluric model.

        This routine performs the following steps:

            #. sample the telluric transmission spectrum (see,
               e.g., :func:~pypeit.telluric.model.PCATelluricModel.base_sample`)
               at its native resolution and wavelength grid,

            #. convolve the atmosphere model to the provided spectral resolution

            #. shift and stretch the telluric model.

        The parameters are ordered such that the first :attr:`base_npar` are
        used to sample the telluric model spectrum (e.g., the number of PCA
        components to use), and the remaining parameters are the spectral
        resolution, spectral shift, and pixel stretch.  I.e., the true number of
        parameters is :math:`N_{\rm base} + 3`.

        Sampling the model can be dramatically sped up by only selecting a
        relevant wavelength range, using the ``start`` and ``end`` parameters.
        This is because we only need to convolve the portion that is needed for
        the current model fit.

        Parameters
        ----------
        theta : `numpy.ndarray`_
            Vector with the full set of parameters.  The number of parameters is
            :math:`N_{\rm base} + 3` where :math:`N_{\rm base}` is
            :attr:`base_npar`, which is the number of parameters needed to
            sample the "base" model (see, e.g.,
            :func:`~pypeit.telluric.model.PCATelluricModel.base_sample`).  The
            three remaining parameters are (**in this order**) the spectral
            resolution, spectral shift, and pixel stretch.
        start : :obj:`int`, optional
            The index (inclusive) of the first spectral pixel to include in the
            model. 
        end : :obj:`int`, optional
            The index (exclusive) of the last spectral pixel to include in the
            model.

        Returns
        -------
        wave : `numpy.ndarray`_
            Wavelength at which the transmission spectrum has been evaluated
        tspec : `numpy.ndarray`_
            Transmission spectrum.
        """
        if len(theta) != self.npar:
            raise PypeItError(
                f'Incorrect number of parameters provided.  Expected {self.npar}, got '
                f'{len(theta)}.'
            )
        _start = start
        _end = self.wave_grid.size if end is None else end

        # Deal with padding for the convolutions
        start_pad = np.fmax(_start - self.tell_pad_pix, 0)
        end_pad = np.fmin(_end + self.tell_pad_pix, self.wave_grid.size)
#        ## FW: There is an extreme case with ind_upper == ind_upper_pad, the previous -0 won't work
#        ind_lower_final = ind_lower_pad if ind_lower_pad == ind_lower else ind_lower - ind_lower_pad
#        ind_upper_final = ind_upper_pad if ind_upper_pad == ind_upper else ind_upper - ind_upper_pad

        # Get the raw transmission spectrum
        tspec = self.base_sample(theta[:self.base_npar], start=start_pad, end=end_pad)
        # Convolve it to the provided resolution

        # TODO: Match resolution
        tspec = self._convolve(tspec[start_pad:end_pad], theta[-3])
                                   
        # Stretch and shift telluric wavelength grid
        # TODO: Resample
        wave = self.wave_grid[start_pad:end_pad]
        return wave, self._shift_and_stretch(np.log10(wave), tspec, theta[-2], theta[-1])


class PCATelluricModel(TelluricModel):
    """
    Telluric model based on the PCA of a large grid of atmospheric models.

    The attributes provided below are specific to this class; see the
    description of the base class for additional attributes.

    .. note::
        
        When selecting the number of PCA components to use in the model, the
        number may be larger than the number PCA components available; if so,
        the class will automatically just use all available components.  Be
        aware of warnings that are issued if you don't see different results as
        you increase the number of components.

        The number of coefficients required to construct the base-level telluric
        model (excluding the resolution, shift, and stretch parameters) is
        :attr:`npca` - 1; the coefficient for the first PCA component is
        *always* set to 1.0.

    Parameters
    ----------
    filename : :obj:`str`
        File with the telluric model data
    npca : :obj:`int`, optional
        Number of PCA components to use in the model.  If None, all available
        components are used.  Must be 2 or more because, otherwise, the
        base-level telluric model will have *no* free parameters; see the note
        above.
    kwargs : :obj:`dict`, optional
        Passed directly to :class:`~pypeit.telluric.model.TelluricModel` base
        class.

    Attributes
    ----------
    npca : :obj:`int`
        Number of PCA components available in the telluric model.
    coeff_bounds : :list
        List of two-tuples with the lower and upper bounds for each of the PCA
        coefficients; length is :attr:`npca` - 1.
    """
    def __init__(self, filename, npca=None, **kwargs):
        # NOTE: These must come *before* instantiating the base class because
        # the instantiation method calls the load() function.
        if npca is not None and npca < 2:
            raise PypeItError('Number of PCA components must be 2 or more!')
        self.npca = npca
        self.coeff_bounds = None
        super().__init__(filename, **kwargs)

    def load(self):
        """
        Reads in the telluric PCA components from a file.
        """
        # Open the file
        log.info(f'Attempting to load telluric file: {self.file.name}')
        hdu = io.fits_open(self.file)

        # Make sure the file type is correct
        _npca = hdu[0].header.get('NCOMP')
        # check that the telgrid file is the correct one for this method
        if _npca is None:
            raise PypeItError(
                'Could NOT read the number of PCA components of the telluric model.  This error '
                'can occur if you have set teltype=pca and have instead used a grid-based '
                'telluric file.  Make sure you are using a TellPCA_* file or set teltype=grid.'
             )
        # Include the 0th component in the total count of the available PCA
        # components
        _npca += 1

        # Set the number of PCA components
        if self.npca is None:
            # Default to using all components
            self.npca = _npca
        if self.npca > _npca:
            log.warning(
                f'Requested {self.npca} PCA components, which is more than the maximum '
                f'available.  Using all {_npca} PCA components.'
            )
            self.npca = _npca

        # Get the relevant data
        wave_grid_full = hdu[1].data
        pca_comp_full = hdu[0].data[:self.npca]         # Only keep the first npca components

        # Coefficient bounds are provided by the data file.  Keep those relevant
        # to the model parameters and convert the object into a list of
        # two-tuples.
        self.coeff_bounds = [tuple(bnd) for bnd in hdu[2].data.T[1:self.npca]]

        # The file also includs a set of model coefficient values.  These can be
        # used as a prior for future refactors, but they are currently not
        # saved.
#        self.model_coeffs = hdu[3].data

        # Close the fits file
        hdu.close()

        # Try to get the resolution
        if self.model_res is None:
            # NOTE: This expects the filename to be of the form:
            #   *_{lambda start}_{lambda end}_R{resolution}.fits
            try:
                self.model_res = int(self.file.name.split('_')[-1][1:])
            except Exception as e:
                log.warning(
                    'Could not determine the spectral resolution of the telluric model from the '
                    'filename and it was not provided directly to the telluric model code.  '
                    'Continuing by assuming that the telluric model spectral resolution is '
                    'effectively infinite compared to your observed data.  '
                    f'File name is {self.file.name};  Exception raised: {e}'
                )

        # Number of parameters
        self.base_npar = self.npca - 1
        self.npar = self.base_npar + 3

        # Finalize
        self.wave_grid, self.tell_grid, self.dloglam, self.tell_pad_pix \
            = self._finalize_wave_grid(wave_grid_full, pca_comp_full)

    def base_par_guess(self):
        """
        Generate a first-guess for the model parameters.

        Returns
        -------
        `numpy.ndarray`
            Guess model parameters.
        """
        return np.zeros(self.base_npar, dtype=float)

    def base_par_bounds(self):
        """
        Generate the parameter bounds for the base-level spectral model.

        Returns
        -------
        list
            List of tuples with the lower and upper bounds of the parameters for
            the base-level model.
        """
        # TODO: Return a copy?
        return self.coeff_bounds

    def base_sample(self, theta, start=0, end=None):
        """
        Sample the telluric model.

        Parameters
        ----------
        theta : `numpy.ndarray`_
            PCA coefficients for the (:attr:`npca` - 1) components.  Length must
            be :attr:`base_npar`.
        start : :obj:`int`, optional
            Starting pixel at which to return the model.
        end : :obj:`int`, optional
            Ending pixel (exclusive) at which to return the model.  If None, the
            end point is the end of the spectrum.

        Returns
        -------
        `numpy.ndarray`_
            Telluric model based on the provided weights.  Pixels outside the
            range of ``start:end`` are set to 0.
        """
        if len(theta) != self.base_npar:
            raise PypeItError(
                f'Incorrect number of coefficients.  Expected {self.base_npar}, got {len(theta)}'
            )
        # Evaluate PCA model after truncating the wavelength range
        tellmodel_hires = np.zeros_like(self.wave_grid)
        tellmodel_hires[start:end] = np.dot(
            np.append(1,theta), self.tell_grid[:,start:end]
        )
                                 
        # PCA model is inverse sinh of the optical depth, convert to transmission here
        tellmodel_hires[start:end] = np.sinh(tellmodel_hires[start:end])

        # It should generally be very rare, but trim negative optical depths here just in case.
        tellmodel_hires[tellmodel_hires < 0] = 0
        tellmodel_hires[start:end] = np.exp(-tellmodel_hires[start:end])
        return tellmodel_hires


class AtmGridTelluricModel(TelluricModel):
    """
    Telluric model based on a large grid of atmospheric models.

    The attributes provided below are specific to this class; see the
    description of the base class for additional attributes.

    Parameters
    ----------
    filename : :obj:`str`
        File with the telluric model data
    kwargs : :obj:`dict`, optional
        Passed directly to :class:`~pypeit.telluric.model.TelluricModel` base
        class.

    Attributes
    ----------
    pressure_grid : `numpy.ndarray`_
        Grid of pressures sampled by the telluric grid.
    temp_grid : `numpy.ndarray`_
        Grid of temperatures sampled by the telluric grid.
    h2o_grid : `numpy.ndarray`_
        Grid of humidity levels sampled by the telluric grid.
    airmass_grid : `numpy.ndarray`_
        Grid of airmass values sampled by the telluric grid.
    """
    def __init__(self, filename, **kwargs):
        # NOTE: These must come *before* instantiating the base class because
        # the instantiation method calls the load() function.
        self.pressure_grid = None
        self.temp_grid = None
        self.h2o_grid = None
        self.airmass_grid = None
        super().__init__(filename, **kwargs)

    def load(self):
        """
        Reads the telluric models for a full atmospheric grid from a file.
        """
        # Read the file
        log.info(f'Attempting to load telluric file: {self.file.name}')
        hdu = io.fits_open(self.file)

        # Check that the telgrid file is the correct one for this method
        if hdu[0].header.get('PRES0') is None:
            raise PypeItError(
                'Could NOT read the atmospheric information from the telluric model.  This error '
                'can occur if you have set teltype=grid and have instead used a pca-based '
                'telluric file.  Make sure you are using a TelFit_* file (to continue using a '
                'grid-based model) or set teltype=pca (to use the PCA-based model).'
             )

        # Get the relevant data
        wave_grid_full = 10.0*hdu[1].data
        model_grid_full = hdu[0].data

        # Construct the parameter grid
        self.pressure_grid = hdu[0].header['PRES0'] \
            + hdu[0].header['DPRES'] * np.arange(hdu[0].header['NPRES'])
        self.temp_grid = hdu[0].header['TEMP0'] \
            + hdu[0].header['DTEMP'] * np.arange(hdu[0].header['NTEMP'])
        self.h2o_grid = hdu[0].header['HUM0'] \
            + hdu[0].header['DHUM'] * np.arange(hdu[0].header['NHUM'])
        if hdu[0].header['NAM'] > 1:
            self.airmass_grid = hdu[0].header['AM0'] \
                + hdu[0].header['DAM'] * np.arange(hdu[0].header['NAM'])
        else:
            self.airmass_grid = np.array([hdu[0].header['AM0']])

        # Close the fits file
        hdu.close()

        # Try to get the resolution
        if self.model_res is None:
            # NOTE: This expects the filename to be of the form:
            #   *_{lambda start}_{lambda end}_R{resolution}.fits
            try:
                self.model_res = int(self.file.name.split('_')[-1][1:])
            except Exception as e:
                log.warning(
                    'Could not determine the spectral resolution of the telluric model from the '
                    'filename and it was not provided directly to the telluric model code.  '
                    'Continuing by assuming that the telluric model spectral resolution is '
                    'effectively infinite compared to your observed data.  '
                    f'File name is {self.file.name};  Exception raised: {e}'
                )

        # Set the number of parameters
        self.base_npar = 4
        self.npar = self.base_npar + 3

        # Finalize
        self.wave_grid, self.tell_grid, self.dloglam, self.tell_pad_pix \
            = self._finalize_wave_grid(wave_grid_full, model_grid_full)
        
    def base_par_guess(self):
        """
        Generate a first-guess for the model parameters.

        Returns
        -------
        `numpy.ndarray`
            Guess model parameters.
        """
        return np.array([
            np.median(self.pressure_grid),
            np.median(self.temp_grid),
            np.median(self.h2o_grid),
            np.median(self.airmass_grid),
        ])

    def base_par_bounds(self):
        """
        Generate the parameter bounds for the base-level spectral model.

        Returns
        -------
        list
            List of tuples with the lower and upper bounds of the parameters for
            the base-level model.
        """
        return [
            (np.min(self.pressure_grid), np.max(self.pressure_grid)),
            (np.min(self.temp_grid), np.max(self.temp_grid)),
            (np.min(self.h2o_grid), np.max(self.h2o_grid)),
            (np.min(self.airmass_grid), np.max(self.airmass_grid)),
        ]

    def base_sample(self, theta, start=0, end=None):
        """
        Interpolate the telluric model grid to the specified location in
        parameter space.

        The interpolation is only performed over the 4D parameter space specified
        by pressure, temperature, humidity, and airmass. This routine performs
        nearest-gridpoint interpolation to evaluate the telluric model at an
        arbitrary location in this 4-d space. The telluric grid is assumed to be
        uniformly sampled in this parameter space.

        Parameters
        ----------
        theta : `numpy.ndarray`_
            A 4-element vector with the telluric model parameters **in the
            following order**: pressure, temperature, humidity, and airmass.
        start : :obj:`int`, optional
            Starting pixel at which to return the model.
        end : :obj:`int`, optional
            Ending pixel (exclusive) at which to return the model.  If None, the
            end point is the end of the spectrum.

        Returns
        -------
        `numpy.ndarray`_
            Telluric model evaluated at the provided 4D position in parameter
            space.  Pixels outside the range of ``start:end`` are set to 0.
        """
        if len(theta) != self.base_npar:
            raise PypeItError(
                f'Incorrect number of grid parameters. Expected {self.base_npar}, got {len(theta)}'
            )
        # TODO: Use np.digitize?
        indx = [
            int(np.round((p - g[0])/(g[1]-g[0]))) if len(g) > 1 else 0 
            for p, g in zip(
                theta, [self.pressure_grid, self.temp_grid, self.h2o_grid, self.airmass_grid]
            )
        ]
        tellmodel_hires = np.zeros_like(self.wave_grid)
        tellmodel_hires[start:end] = self.tell_grid[*indx][start:end]
        return tellmodel_hires
