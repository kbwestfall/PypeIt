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
    """
    Base class

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
    """
    def __init__(self, filename, wave_min=None, wave_max=None, pad_frac=0.1):
        to_pkg = 'move' if ".dev" in __version__ else None
        self.file = dataPaths.telgrid.get_file_path(filename, to_pkg=to_pkg)
        self.wave_min = wave_min
        self.wave_max = wave_max
        self.pad_frac = pad_frac

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
        return wave_grid, model_grid, dloglam, int(np.ceil(10.0 * pix_per_sigma))

    def sample_raw(self):
        """
        Sample the telluric model at its native resolution and wavelength grid.
        """
        raise PypeItError(f'{self.__class__.__name__} has not defined a sample_raw function!')

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
        # Check the input values
        if res <= 0.0:
            raise PypeItError('Resolution must be positive.')
        if self.dloglam == 0.0:
            raise PypeItError('The telluric model grid cannot have a log wavelength spacing that is 0!')
        # number of dloglam pixels per 1 sigma dispersion
        pix_per_sigma = 1.0/res/(self.dloglam*np.log(10.0))/(2.0 * np.sqrt(2.0 * np.log(2)))
        # number of sigma per 1 pix
        sig2pix = 1.0/pix_per_sigma
        if sig2pix > 2.0:
            log.warning(
                'The telluric model grid is not sampled finely enough to properly convolve to '
                'the desired resolution.  Skipping resolution convolution for now. Create a '
                'higher resolution telluric model grid.'
            )
            return tspec

        # x = loglam/sigma on the wavelength grid from -4 to 4, symmetric, centered about zero.
        x = np.hstack([-np.flip(np.arange(sig2pix,4,sig2pix)), np.arange(0,4,sig2pix)])
        # g = Gaussian evaluated over x
        g = np.exp(-0.5*x**2)
        # Convolve with a normalized Gaussian kernel
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

    def sample(self, theta, start=0, end=None):
        r"""
        Evaluate the telluric model.

        This routine performs the following steps:

            #. sample the telluric transmission spectrum (see,
               e.g., :func:~pypeit.telluric.model.PCATelluricModel.sample_raw`)
               at its native resolution and wavelength grid,

            #. convolve the atmosphere model to the provided spectral resolution

            #. shift and stretch the telluric model.

        The parameters are ordered such that the first :attr:`n_raw_par` are
        used to sample the raw telluric spectrum (e.g., the number of PCA
        components to use), and the remaining parameters are the spectral
        resolution, spectral shift, and pixel stretch.  I.e., the true number of
        parameters is :math:`N_{\rm raw} + 3`.

        Sampling the model can be dramatically sped up by only selecting a
        relevant wavelength range, using the ``start`` and ``end`` parameters.
        This is because we only need to convolve the portion that is needed for
        the current model fit.

        Parameters
        ----------
        theta : `numpy.ndarray`_
            Vector with the full set of parameters.  The number of parameters is
            :math :math:`N_{\rm raw} + 3` where :math:`N_{\rm raw}` is
            :attr:`n_raw_par`, which is the number of parameters needed to
            sample the "raw" model (see, e.g.,
            :func:`~pypeit.telluric.model.PCATelluricModel.sample_raw`).  The
            three remaining parameters are (**in this order**) the spectral
            resolution, spectral shift, and pixel stretch.
        start : :obj:`int`, optional
            The index (inclusive) of the first pixel to include in the model. 
        end : :obj:`int`, optional
            The index (exclusive) of the last pixel to include in the model.

        Returns
        -------
        wave : `numpy.ndarray`_
            Wavelength at which the transmission spectrum has been evaluated
        tspec : `numpy.ndarray`_
            Transmission spectrum.
        """
        if len(theta) != self.npar + 3:
            raise PypeItError(
                f'Incorrect number of parameters provided.  Expected {self.npar + 3}, got '
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
        tspec = self.sample_raw(theta[:self.npar], start=start_pad, end=end_pad)
        # Convolve it to the provided resolution

        # TODO: Match resolution
        tspec = self._convolve(tspec[start_pad:end_pad], theta[-3])
                                   
        # Stretch and shift telluric wavelength grid
        # TODO: Resample
        wave = self.wave_grid[start_pad:end_pad]
        return wave, self._shift_and_stretch(np.log10(wave), tspec, theta[-2], theta[-1])


class PCATelluricModel(TelluricModel):
    """
    Telluric model based on the PCA of a large grid of atmospheric models
    """
    def __init__(self, filename, wave_min=None, wave_max=None, pad_frac=0.1, npca=None):
        self.npar = npca
        super().__init__(filename, wave_min=wave_min, wave_max=wave_max, pad_frac=pad_frac)

    def load(self):
        """
        Reads in the telluric PCA components from a file.
        """
        # Open the file
        log.info(f'Attempting to load telluric file: {self.file.name}')
        hdu = io.fits_open(self.file)

        # Make sure the file type is correct
        self.ncomp = hdu[0].header.get('NCOMP')
        # check that the telgrid file is the correct one for this method
        if self.ncomp is None:
            raise PypeItError(
                'Could NOT read the number of PCA components of the telluric model.  This error '
                'can occur if you have set teltype=pca and have instead used a grid-based '
                'telluric file.  Make sure you are using a TellPCA_* file or set teltype=grid.'
             )

        # Get the relevant data
        wave_grid_full = hdu[1].data
        pca_comp_full = hdu[0].data
        self.bounds = hdu[2].data
        self.model_coefs = hdu[3].data

        # Close the fits file
        hdu.close()

        # Finalize
        self.wave_grid, self.tell_grid, self.dloglam, self.tell_pad_pix \
            = self._finalize_wave_grid(wave_grid_full, pca_comp_full)
        
        if self.npar is None:
            self.npar = self.ncomp
        if self.npar > self.ncomp:
            log.warning(
                f'Requested {self.npar} PCA components, which is more than the maximum '
                f'available.  Using all {self.ncomp} PCA components.'
            )
            self.npar = self.ncomp
        
    def sample_raw(self, theta, start=0, end=None):
        """
        Sample the telluric model.

        Parameters
        ----------
        theta : `numpy.ndarray`_
            Weights for the first :attr:`npar` components.
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
        if len(theta) != self.npar:
            raise PypeItError(
                f'Incorrect number of coefficients.  Expected {self.npar}, got {len(theta)}'
            )
        # Evaluate PCA model after truncating the wavelength range
        tellmodel_hires = np.zeros_like(self.wave_grid)
        tellmodel_hires[start:end] = np.dot(
            np.append(1,theta), self.tell_grid[:self.npar+1][:,start:end])
                                 
        # PCA model is inverse sinh of the optical depth, convert to transmission here
        tellmodel_hires[start:end] = np.sinh(tellmodel_hires[start:end])

        # It should generally be very rare, but trim negative optical depths here just in case.
        tellmodel_hires[tellmodel_hires < 0] = 0
        tellmodel_hires[start:end] = np.exp(-tellmodel_hires[start:end])
        return tellmodel_hires


class AtmGridTelluricModel(TelluricModel):
    """
    Telluric model based on the PCA of a large grid of atmospheric models
    """
    def __init__(self, filename, wave_min=None, wave_max=None, pad_frac=0.1):
        self.npar = 4
        super().__init__(filename, wave_min=wave_min, wave_max=wave_max, pad_frac=pad_frac)

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
                'telluric file.  Make sure you are using a TelFit_* file or set teltype=pca.'
             )

        # Get the relevant data
        wave_grid_full = 10.0*hdu[1].data
        model_grid_full = hdu[0].data

        # Construct the parameter grid
        self.pressure_grid = hdu[0].header['PRES0'] \
            + hdu[0].header['DPRES'] * np.arange(0,hdu[0].header['NPRES'])
        self.temp_grid = hdu[0].header['TEMP0'] \
            + hdu[0].header['DTEMP'] * np.arange(0,hdu[0].header['NTEMP'])
        self.h2o_grid = hdu[0].header['HUM0'] \
            + hdu[0].header['DHUM'] * np.arange(0,hdu[0].header['NHUM'])
        if hdu[0].header['NAM'] > 1:
            self.airmass_grid = hdu[0].header['AM0'] \
                + hdu[0].header['DAM'] * np.arange(0,hdu[0].header['NAM'])
        else:
            self.airmass_grid = np.array([hdu[0].header['AM0']])

        # Close the fits file
        hdu.close()

        # Finalize
        self.wave_grid, self.tell_grid, self.dloglam, self.tell_pad_pix \
            = self._finalize_wave_grid(wave_grid_full, model_grid_full)
        
    def sample_raw(self, theta, start=0, end=None):
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
        if len(theta) != 4:
            raise PypeItError('Input parameter vector must have 4 and only 4 values.')
        indx = [
            int(np.round((p - g[0])/(g[1]-g[0]))) if len(g) > 1 else 0 for p, g in zip(theta, [
                self.pressure_grid, self.temp_grid, self.h2o_grid, self.airmass_grid
                ])
        ]
        tellmodel_hires = np.zeros_like(self.wave_grid)
        tellmodel_hires[start:end] = self.tell_grid[*indx][start:end]
        return tellmodel_hires
