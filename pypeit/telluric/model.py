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
from pypeit.core import pixels


class TelluricModel:
    r"""
    Base class for all telluric models.

    The subclasses must provide the following functions:

        - :func:`~pypeit.telluric.model.TelluricModel.load`: Load the telluric
          data and perform the remaining instantiation steps.

        - :func:`~pypeit.telluric.model.TelluricModel.base_par_guess`: Generate
          guess parameters for the underlying telluric spectrum based on an
          observed spectrum.

        - :func:`~pypeit.telluric.model.TelluricModel.base_par_bounds`: Generate
          lower and upper parameter boundaries for the underlying telluric
          spectrum based on an initial set of guess parameters.

        - :func:`~pypeit.telluric.model.TelluricModel.base_sample`: Sample the
          underlying telluric spectrum given a set of parameters.

    There are two ways to restrict the wavelength range of the model:
        
        #. At instantiation, use the ``wave_min`` and ``wave_max`` arguments to
           limit the wavelength range of the underlying model data that is held
           in memory.  These parameters are passed directly to the :func:`load`
           function.

        #. After instantiation, use the :func:`restrict_wave_range` function to
           set the wavelength range for the evaluated model.  The underlying
           model data is still held in memory, but the base model evaluation is
           only over the specified wavelength range.

    The first approach limits the amount of data held in memory, while the
    second can be used to speed up the convolution of the model by, e.g.,
    limiting it to the wavelength range of an observed spectrum being fit.

    Parameters
    ----------
    filename : :obj:`str`
        File with the telluric model data
    model_res : :obj:`float`, optional
        Spectral resolution (:math:`R = \lambda / \Delta\lambda`) of the model
        spectra.  The resolution is expected to be constant as a function of
        wavelength.  If None, the resolution is assumed to be effectively
        infinite.
    load : :obj:`bool`, optional
        Flag to load the data upon instantiation.
    wave_min : :obj:`float`, optional
        Minimum wavelength accessible to the model.  If None, a lower limit is
        not set.
    wave_max : :obj:`float`, optional
        Maximum wavelength accessible to the model.  If None, an upper limit is
        not set.

    Attributes
    ----------
    file : :class:`pathlib.Path`
        Path to the telluric model data file.
    model_res : :obj:`float`
        Spectral resolution (:math:`R = \lambda / \Delta\lambda`) of the model
        spectra.
    wave : `numpy.ndarray`_
        Wavelength grid of the telluric model.
    tell_grid : `numpy.ndarray`_
        Data used to construct the telluric model.
    dloglam : :obj:`float`
        Delta log10(lambda) between pixels in the telluric model wavelength vector.
    npad : :obj:`int`
        Number of pixels to pad the telluric model when performing convolutions
        to avoid edge effects.
    base_npar : :obj:`int`
        Number of parameters needed to generate the underlying telluric model
        (everything except the resolution, shift, and stretch).
    npar : :obj:`int`
        Total number of parameters for the model.  This is always the number of
        base parameters plus 3 (for the resolution, shift, and stretch).
    s_wave : :obj:`int`
        Starting pixel in the wavelength grid to use when evaluating the model.
    e_wave : :obj:`int`
        Ending pixel (exclusive) in the wavelength grid to use when evaluating
        the model.
    """
    def __init__(self, filename, model_res=None, load=True, wave_min=None, wave_max=None):
        to_pkg = 'move' if ".dev" in __version__ else None
        self.file = dataPaths.telgrid.get_file_path(filename, to_pkg=to_pkg)
        self.model_res = model_res

        # Defined by the subclass load functions
        self.wave = None
        self.tell_grid = None
        self.dloglam = None
        self.npad = None

        # The number of parameters needed to generate the underlying telluric
        # model (everything except the resolution, shift, and stretch)
        self.base_npar = None
        # The total number of parameters for the model.  This is always
        # base_npar + 3, but we haven't defined base_npar yet.  It should always
        # be set by the load() function.
        self.npar = None

        # Can be set based on a wavelength range set by restricted_wave()
        self.s_wave = None
        self.e_wave = None

        # Load the data
        if load:
            self.load(wave_min=wave_min, wave_max=wave_max)

    def load(self, wave_min=None, wave_max=None):
        """
        Load the telluric data from the reference file.
        **Must be implemented by each subclass.**

        Parameters
        ----------
        wave_min : :obj:`float`, optional
            Minimum wavelength accessible to the model.  If None, a lower limit
            is not set.
        wave_max : :obj:`float`, optional
            Maximum wavelength accessible to the model.  If None, an upper limit
            is not set.
        """
        raise PypeItError(f'{self.__class__.__name__} has not defined a load function!')
    
    def restrict_wave_range(self, wave_min=None, wave_max=None, pad_frac=0.1):
        """
        Restrict the wavelength range of the evaluated model.

        This function sets the starting (:attr:`s_wave`) and ending
        (:attr:`e_wave`) pixel values in the wavelength grid (:attr:`wave`)
        to be used when evaluating the telluric model.  If both ``wave_min`` and
        ``wave_max`` are None, the full wavelength is used.  This function can
        also be used to reset the object to use the full wavelength range, by
        running it without any arguments: ``self.restrict_wave_range()``.

        Parameters
        ----------
        wave_min : :obj:`float`, optional
            Minimum wavelength at which to evaluate the model.  If None, a lower
            limit is not set.
        wave_max : :obj:`float`, optional
            Maximum wavelength at which to evaluate the model.  If None, an
            upper limit is not set.
        pad_frac : :obj:`float`, optional
            Percentage padding to be added to the model boundaries if
            ``wave_min`` or ``wave_max`` are provided; ignored otherwise. The
            resulting grid will extend from ``(1.0 - pad_frac)*wave_min`` to
            ``(1.0 + pad_frac)*wave_max``.
        """
        if self.wave is None:
            raise PypeItError('Telluric model wavelength vector has not been defined yet!')

        if wave_min is None:
            _wave_min = None
        else:
            _wave_min = wave_min if pad_frac is None else (1.0 - pad_frac) * wave_min

        if wave_max is None:
            _wave_max = None
        else:
            _wave_max = wave_max if pad_frac is None else (1.0 + pad_frac) * wave_max

        self.s_wave, self.e_wave = pixels.convert_to_pixel_range(
            self.wave, x_min=_wave_min, x_max=_wave_max
        )

    @property
    def restricted_wave_range(self):
        """
        Return True if the wavelength range of the model has been restricted.
        """
        return self.s_wave is not None or self.e_wave is not None
    
    def _base_sample_spectral_axis(self):
        """
        Return the pixel range over which to sample the model and the associated
        wavelength vector and good-pixel mask.  This will apply the restricted
        wavelength range (if defined) and add convolution padding.

        Returns
        -------
        start_pad : :obj:`int`
            Starting pixel at which to return the model, including padding.
        end_pad : :obj:`int`
            Ending pixel (exclusive) at which to return the model, including
            padding.
        wave : `numpy.ndarray`_
            Wavelength vector over which to evaluate the model.
        gpm : `numpy.ndarray`_
            Good-pixel mask indicating valid pixels within the wavelength vector.

        Raises
        ------
        PypeItError
            Raised if the wavelength grid and/or padding have not been defined
            yet.
        """
        if self.wave is None or self.npad is None:
            raise PypeItError(
                'Telluric model wavelength vector and/or padding have not been defined yet!'
            )

        # Restrict the wavelength range
        _start = 0 if self.s_wave is None else self.s_wave
        _end = self.wave.size if self.e_wave is None else self.e_wave
        # Include padding to deal with inaccurate convolutions from edge effects
        start_pad = np.fmax(_start - self.npad, 0)
        end_pad = np.fmin(_end + self.npad, self.wave.size)
        # The truncating the wavelength range
        _wave = self.wave[start_pad:end_pad]
        # The good-pixel mask, which always masks at least npad on
        # either end of the model spectrum
        gpm = np.ones_like(_wave, dtype=bool)
        gpm[:max(_start - start_pad, self.npad)] = False
        gpm[-max(_end - end_pad, self.npad):] = False
        # Return the results
        return start_pad, end_pad, _wave, gpm

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
        # TODO: This should be a wavelength-dependent kernel
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
        # TODO: Optimize this convolution to be faster
        return signal.convolve(tspec, g/g.sum(), mode='same')

    # TODO: Shift is not independent of scale, and we should be using resampling
    # instead of interpolation.
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
        **Must be implemented by each subclass.**

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
        **Must be implemented by each subclass.**

        Returns
        -------
        list
            List of two-tuples with the lower and upper bounds of the parameters
            for the base-level model.
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
            A list of two-tuples that provide the lower and upper bounds for
            each model parameter.
        """
        # guess_par[-3] is the guess resolution.
        return self.base_par_bounds() + [
            tuple(guess_par[-3] * np.asarray(resolution_frac_bounds)),
            pix_shift_bounds,
            pix_stretch_bounds,
        ]

    def base_sample(self, theta):
        """
        Sample the telluric model at its native resolution and wavelength vector.
        **Must be implemented by each subclass.**

        This function should account for any restricted wavelength range and
        padding.

        Parameters
        ----------
        theta : `numpy.ndarray`_
            Vector with the parameters needed to sample the telluric model.
            Length must be :attr:`base_npar`.

        Returns
        -------
        wave : `numpy.ndarray`_
            Wavelength at which the transmission spectrum has been evaluated
        tspec : `numpy.ndarray`_
            Transmission spectrum.
        gpm : `numpy.ndarray`_
            Good pixel mask for the transmission spectrum.
        """
        raise PypeItError(f'{self.__class__.__name__} has not defined a base_sample function!')

    def sample(self, theta):
        r"""
        Evaluate the telluric model.

        This routine performs the following steps:

            #. sample the telluric transmission spectrum (see,
               e.g., :func:~pypeit.telluric.model.PCATelluricModel.base_sample`)
               at its native resolution and over the (restricted) wavelength
               grid,

            #. convolve the model to the provided spectral resolution

            #. shift and stretch the model.

        The parameters are ordered such that the first :attr:`base_npar` are
        used to sample the telluric model spectrum (e.g., the number of PCA
        components to use), and the remaining parameters are the spectral
        resolution, spectral shift, and pixel stretch.  I.e., the true number of
        parameters is :math:`N_{\rm base} + 3`.

        Sampling the model can be dramatically sped up by restricting the
        wavelength range of the model; see
        :func:`~pypeit.telluric.model.TelluricModel.restrict_wave_range`.

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

        Returns
        -------
        wave : `numpy.ndarray`_
            Wavelength at which the transmission spectrum has been evaluated
        tspec : `numpy.ndarray`_
            Transmission spectrum.
        gpm : `numpy.ndarray`_
            Good pixel mask for the transmission spectrum.
        """
        if len(theta) != self.npar:
            raise PypeItError(
                f'Incorrect number of parameters provided.  Expected {self.npar}, got '
                f'{len(theta)}.'
            )

        # Get the raw transmission spectrum; base_sample should restrict the
        # wavelength range and add any necessary padding!
        wave, tspec, gpm = self.base_sample(theta[:self.base_npar])

        # Convolve it to the provided resolution
        # TODO: Match resolution
        tspec = self._convolve(tspec, theta[-3])
                                   
        # Stretch and shift telluric wavelength grid
        # TODO: Resample
        return wave, self._shift_and_stretch(np.log10(wave), tspec, theta[-2], theta[-1]), gpm


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

    def load(self, wave_min=None, wave_max=None):
        """
        Reads in the telluric PCA components from a file.

        Parameters
        ----------
        wave_min : :obj:`float`, optional
            Minimum wavelength accessible to the model.  If None, a lower limit
            is not set.
        wave_max : :obj:`float`, optional
            Maximum wavelength accessible to the model.  If None, an upper limit
            is not set.
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

        # Apply the restricted wavelength range
        s_wave, e_wave = pixels.convert_to_pixel_range(hdu[1].data, x_min=wave_min, x_max=wave_max)

        # Get the relevant data
        self.wave = hdu[1].data[s_wave:e_wave]
        self.tell_grid = hdu[0].data[:self.npca,s_wave:e_wave]
        _, self.dloglam, _, pix_per_sigma = wvutils.get_sampling(self.wave)
        self.npad = int(np.ceil(10.0 * pix_per_sigma))

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
            # TODO: This will go wrong if the file is in the cache!
            try:
                self.model_res = float(self.file.stem.split('_')[-1][1:])
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
            List of two-tuples with the lower and upper bounds of the parameters
            for the base-level model.
        """
        # TODO: Return a copy?
        return self.coeff_bounds

    def base_sample(self, theta):
        """
        Sample the telluric model.

        Parameters
        ----------
        theta : `numpy.ndarray`_
            PCA coefficients for the (:attr:`npca` - 1) components.  Length must
            be :attr:`base_npar`.

        Returns
        -------
        wave : `numpy.ndarray`_
            Wavelength at which the transmission spectrum has been evaluated
        tspec : `numpy.ndarray`_
            Transmission spectrum.
        gpm : `numpy.ndarray`_
            Good pixel mask for the transmission spectrum.

        Raises
        ------
        PypeItError
            Raised if the number of parameters provided by ``theta`` does not
            match :attr:`base_npar`.
        """
        if len(theta) != self.base_npar:
            raise PypeItError(
                f'Incorrect number of coefficients.  Expected {self.base_npar}, got {len(theta)}'
            )

        # Get the pixel range over which to evaluate the model, and the
        # associated wavelength vector and good-pixel mask
        start_pad, end_pad, wave, gpm = self._base_sample_spectral_axis()
        # Combine the PCA components
        tspec = np.dot(np.append(1,theta), self.tell_grid[:,start_pad:end_pad])
        # PCA model is inverse sinh of the optical depth, convert to transmission here
        tspec = np.sinh(tspec)
        # It should generally be very rare, but trim negative optical depths here just in case.
        tspec[tspec < 0] = 0
        tspec = np.exp(-tspec)

        # Return the model
        return wave, tspec, gpm

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

    def load(self, wave_min=None, wave_max=None):
        """
        Reads the telluric models for a full atmospheric grid from a file.

        Parameters
        ----------
        wave_min : :obj:`float`, optional
            Minimum wavelength accessible to the model.  If None, a lower limit
            is not set.
        wave_max : :obj:`float`, optional
            Maximum wavelength accessible to the model.  If None, an upper limit
            is not set.
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

        # Apply the restricted wavelength range
        s_wave, e_wave = pixels.convert_to_pixel_range(
            10.0 * hdu[1].data, x_min=wave_min, x_max=wave_max
        )

        # Get the relevant data
        self.wave = 10.0 * hdu[1].data[s_wave:e_wave]
        self.tell_grid = hdu[0].data[...,s_wave:e_wave]
        _, self.dloglam, _, pix_per_sigma = wvutils.get_sampling(self.wave)
        self.npad = int(np.ceil(10.0 * pix_per_sigma))

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
                self.model_res = float(self.file.stem.split('_')[-1][1:])
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
            List of two-tuples with the lower and upper bounds of the parameters
            for the base-level model.
        """
        return [
            (np.min(self.pressure_grid), np.max(self.pressure_grid)),
            (np.min(self.temp_grid), np.max(self.temp_grid)),
            (np.min(self.h2o_grid), np.max(self.h2o_grid)),
            (np.min(self.airmass_grid), np.max(self.airmass_grid)),
        ]

    def base_sample(self, theta):
        """
        Interpolate the telluric model grid to the specified location in
        parameter space.

        The interpolation is only performed over the 4D parameter space specified
        by pressure, temperature, humidity, and airmass.  This routine performs
        nearest-gridpoint interpolation to evaluate the telluric model at an
        arbitrary location in this 4-d space. The telluric grid is assumed to be
        uniformly sampled in this parameter space.

        Parameters
        ----------
        theta : `numpy.ndarray`_
            A 4-element vector with the telluric model parameters **in the
            following order**: pressure, temperature, humidity, and airmass.
            Length must be :attr:`base_npar`.

        Returns
        -------
        wave : `numpy.ndarray`_
            Wavelength at which the transmission spectrum has been evaluated
        tspec : `numpy.ndarray`_
            Transmission spectrum.
        gpm : `numpy.ndarray`_
            Good pixel mask for the transmission spectrum.

        Raises
        ------
        PypeItError
            Raised if the number of parameters provided by ``theta`` does not
            match :attr:`base_npar`.
        """
        if len(theta) != self.base_npar:
            raise PypeItError(
                f'Incorrect number of grid parameters. Expected {self.base_npar}, got {len(theta)}'
            )

        # Get the pixel range over which to evaluate the model, and the
        # associated wavelength vector and good-pixel mask
        start_pad, end_pad, wave, gpm = self._base_sample_spectral_axis()
        # TODO: Use np.digitize?
        indx = [
            int(np.round((p - g[0])/(g[1]-g[0]))) if len(g) > 1 else 0 
            for p, g in zip(
                theta, [self.pressure_grid, self.temp_grid, self.h2o_grid, self.airmass_grid]
            )
        ]
        return wave, self.tell_grid[*indx][start_pad:end_pad], gpm
