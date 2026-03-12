"""
Temporary light-weight spectrum object.

Ideally this would be replaced by specutils.Spectrum

"""

from copy import deepcopy

from IPython import embed
from matplotlib import pyplot
from matplotlib import ticker
import numpy as np
from scipy import interpolate

from pypeit import bspline
from pypeit import log
from pypeit import PypeItError
from pypeit import sampling
from pypeit import utils
from pypeit.core import fitting
from pypeit.core import wavemask


class Spectrum:
    r"""
    A light-weight container class for a spectrum.

    The flux array can be either 1D or 2D.  If 2D, its shape is always assumed
    to be :math:`(N_{\rm pix},N_{\rm spec})`, where :math:`N_{\rm pix}` is the
    number of pixels per spectrum and :math:`N_{\rm spec}` is the number of
    spectra.  The wavelength array must always be provided as 1D; i.e., for a 2D
    flux array, the wavelength vector must be correct for each flux vector along
    the 1st axis of the 2D array.
    
    Parameters
    ----------
    wave : array-like
        Vacuum wavelengths in angstrom.  Must be 1D, and its length must match
        the first axis of ``flux``.
    flux : array-like
        Flux data at each wavelength.  Can be 2D or 1D; see class description.
    ivar : array-like, optional
        Inverse variance in the flux.  Shape must match ``flux``.  If None,
        assumes uncertainties are unknown.
    gpm : array-like, optional
        Boolean good-pixel mask.  Shape must match ``flux``.  If None, assumes
        all pixels are valid.
    meta : dict, optional
        Collection of relevant metadata.
    assoc : dict, optional
        A set of arrays associated with the flux data.  E.g., these can be
        measurements of the sky flux, the spectral resolution, or a spectral
        model.  The key of the provided dictionary should be a unique identifier
        (name) for each array, and the dictionary item provides its data.  The
        shape of each associated array *must* match the shape of the ``flux``
        array.
    """
    def __init__(self, wave, flux, ivar=None, gpm=None, meta=None, assoc=None):

        # Throughout I use the copy method to ensure the original arrays are
        # copied
        self.flux = np.asarray(flux, dtype=float).copy()

        self.wave = np.asarray(wave, dtype=float).copy()
        if self.wave.ndim != 1:
            raise PypeItError('wavelength array must always be 1D in the spectrum object')
        if self.wave.size != self.flux.shape[0]:
            raise PypeItError('wavelength vector must match length of flux array')

        if ivar is None:
            self.ivar = None
        else:
            self.ivar = np.asarray(ivar, dtype=float).copy()
            if self.ivar.shape != self.flux.shape:
                raise PypeItError('Wavelength and inverse variance arrays do not have the same shape.')

        if gpm is None:
            self.gpm = np.ones(self.flux.shape, dtype=bool)
        else:
            self.gpm = np.asarray(gpm, dtype=bool).copy()
            if self.gpm.shape != self.flux.shape:
                raise PypeItError('Wavelength and good-pixel arrays do not have the same size.')

        self.meta = None if meta is None else deepcopy(meta)

        self.assoc = self._ingest_associated(assoc)

    def _ingest_associated(self, assoc):
        """
        Helper function to ingest associated data arrays
        """
        if assoc is None:
            return None
        if not isinstance(assoc, dict):
            raise PypeItError('Associated data arrays must be provided via a dictionary.')
        
        self.assoc = {}
        for key, arr in assoc.items():
            _arr = np.asarray(arr).copy()
            if _arr.shape != self.flux.shape:
                raise PypeItError(
                    f'Associated data array {key} does not match the shape of the flux array.'
                )
            self.assoc[key] = _arr

    @property
    def size(self):
        """
        The size of the flux array
        """
        return self.flux.size
    
    @property
    def shape(self):
        """
        The shape of the flux array
        """
        return self.flux.shape
    
    @property
    def ndim(self):
        """
        The dimensionality of the flux array
        """
        return self.flux.ndim
    
    def copy(self):
        """
        Return a deepcopy of the object.
        """
        _ivar = None if self.ivar is None else self.ivar.copy()
        _meta = None if self.meta is None else deepcopy(self.meta)
        _assoc = (
            None if self.assoc is None
            else {key : arr.copy() for key, arr in self.assoc.items()}
        )
        return self.__class__(
            self.wave.copy(), self.flux.copy(), ivar=_ivar, gpm=self.gpm.copy(), meta=_meta,
            assoc=_assoc
        )

    def multiply(self, a):
        """
        Multiply the spectrum by a scalar, vector, or another spectrum.

        *This modifies the spectral data in place.*  If uncertainties are
        available, they are propagated.  Any divisions by 0 result in an inverse
        variance of 0 and the good pixel mask is set to False.

        Parameters
        ----------
        a : scalar, array-like, :class:`pypeit.core.spectrum.Spectrum`
            Multiplicative factor.  If an array, its shape must match
            :attr:`flux`.  If a spectrum, the wavelength arrays of the two
            spectrum *must be identical*.
        """
        if self.assoc is not None:
            log.warning('Removing associated arrays due to use of multiply().')
            self.assoc = None

        # Multiply by a scalar
        if isinstance(a, (int, np.integer, float, np.floating)):
            if float(a) == 0.:
                log.warning('Multiplicative factor is 0!')
            self.flux *= a
            if self.ivar is not None:
                if np.absolute(a) > 0:
                    self.ivar /= a**2
                else:
                    self.ivar *= 0.
            return

        # Workspace
        sqr_err_ratio = None
        a_gpm = None

        if isinstance(a, Spectrum):
            # Pull the necessary data out of the spectrum

            # Check the wavelength vectors
            # TODO: Loosen this; i.e., use isclose instead of array_equal?
            if not np.array_equal(a.wave, self.wave):
                raise PypeItError('To multiply two spectra, their wavelength vectors must be identical.')
            a_flux = a.flux
            a_gpm = a.gpm
            if a.ivar is not None:
                # Square of the ratio between the error and flux in a
                sqr_err_ratio = utils.inverse(a.flux**2 * a.ivar)
        else:
            # Convert the array-like object to a numpy array
            a_flux = np.asarray(a)

        # Check the input
        if a_flux.ndim > self.ndim:
            raise PypeItError(
                'Multiplication does not allow the dimensionality of the spectrum to change.  '
                f'The dimensionality of this spectrum is {self.ndim} and the multiplier is '
                f'{a_flux.ndim}.'
            )
        # Numpy broadcasting rules mean that the arithmetic operations performed
        # below should work, as long as the last a.ndim dimensions of a and this
        # spectrum match.
        if a_flux.shape != self.shape[:a_flux.ndim]:
            raise PypeItError(
                'Numpy will not be able to successfully broadcast arithmetic operations between '
                f'this spectrum, shape={self.shape}, and the multiplier, shape={a_flux.shape}.'
            )

        # Reshape the arrays, if necessary
        if a_flux.ndim != self.flux.ndim:
            a_flux = np.expand_dims(
                a_flux, tuple(np.arange(len(self.shape))[a_flux.ndim:].tolist())
            )
            if sqr_err_ratio is not None:
                sqr_err_ratio = np.expand_dims(
                    sqr_err_ratio, tuple(np.arange(len(self.shape))[sqr_err_ratio.ndim:].tolist())
                )
            if a_gpm is not None:
                a_gpm = np.expand_dims(
                    a_gpm, tuple(np.arange(len(self.shape))[a_gpm.ndim:].tolist())
                )

        # Add the error.  NOTE: This *must* be done before the multiplication by
        # a_flux below because that changes self.flux.
        if self.ivar is not None:
            # Square of the ratio between the error and flux in this spectrum
            if sqr_err_ratio is None:
                sqr_err_ratio = utils.inverse(self.flux**2 * self.ivar)
            else:
                sqr_err_ratio += utils.inverse(self.flux**2 * self.ivar)

        # Complete the multiplication
        self.flux *= a_flux
        if sqr_err_ratio is not None:
            # Propagate the error
            sqr_err = self.flux**2 * sqr_err_ratio
            self.ivar = utils.inverse(sqr_err)
            self.gpm[np.logical_not(self.ivar > 0)] = False
        if a_gpm is not None:
            # Propagate the good-pixel mask
            self.gpm &= a_gpm

    def inverse(self):
        """
        Replace the spectrum with its multiplicative inverse.

        *This modifies the spectrum in place.*  If uncertainties are available,
        they are propagated.  Any divisions by 0 result in an inverse variance
        of 0 and the good pixel mask is set to False.
        """
        if self.assoc is not None:
            log.warning('Removing associated arrays due to use of inverse().')
            self.assoc = None
        if self.ivar is not None:
            self.ivar *= self.flux**4
            self.gpm[np.logical_not(self.ivar > 0)] = False
        self.gpm[np.logical_not(self.flux > 0)] = False
        self.flux = utils.inverse(self.flux)

    def to_magnitude(self, zeropoint=0.):
        r"""
        Convert the spectrum to magnitudes.

        For fluxes, :math:`f`, this returns

        .. math::

            m = -2.5 \log_{\rm 10} (f) + Z,

        where :math:`Z` is the provided zeropoint.

        *This modifies the spectrum in place.*  If uncertainties are available,
        they are propagated.  Any pixels with non-positive fluxes are masked.

        Parameters
        ----------
        zeropoint : float, optional
            The magnitude conversion zeropoint (see above)
        """
        if self.assoc is not None:
            log.warning('Removing associated arrays due to use of to_magnitude().')
            self.assoc = None

        if self.ivar is not None:
            self.ivar *= (self.flux * np.log(10) / 2.5)**2
        self.gpm[np.logical_not(self.flux > 0)] = False
        self.flux[np.logical_not(self.gpm)] = 0.
        self.flux[self.gpm] = -2.5 * np.log10(self.flux[self.gpm]) + zeropoint

    def resample(self, new_wave, pixel_fraction_threshold=0.8, conserve=False):
        r"""
        Resample the spectrum to a new wavelength array.

        If available, errors and masking are both propagated through the
        calculation.  This function is basically a wrapper for
        :class:`~pypeit.sampling.Resample`.

        Parameters
        ----------
        new_wave : array-like
            New wavelength vector for the spectrum
        pixel_fraction_threshold : float, optional
            The resampling calculates the fraction of each output pixel that has
            unmasked contributions from the original spectrum.  Fractions below
            this threshold will be masked in the output spectrum.
        conserve : bool, optional
            Conserve the flux in the resampled spectrum.  If the units of the
            spectrum are flux integrated over the pixel, this should typically
            be True; if the units are flux density (e.g., :math:`{\rm
            ergs/s/cm}^2{\rm /angstrom}`), this should typically be False.

        Returns
        -------
        :class:`~pypeit.core.spectrum.Spectrum`
            A new spectrum object with the resample data.
        """
        if self.assoc is not None:
            log.warning('Removing associated arrays due to use of resample().')
            self.assoc = None

        # Setup
        bpm = None if np.all(self.gpm) else np.logical_not(self.gpm).T
        if self.ivar is None:
            err = None
        else:
            err = np.zeros(self.ivar.shape, dtype=float)
            err[self.gpm] = np.sqrt(utils.inverse(self.ivar[self.gpm]))
            err = err.T

        # Resample accepts both 1D and 2D arrays, but it expects the 2D arrays
        # to have the spectra organized along the 2nd axis; i.e.,
        # (N_spec,N_pix) instead of (N_pix,N_spec).
        r = sampling.Resample(
            self.flux.T, e=err, mask=bpm, x=self.wave, newx=new_wave, conserve=conserve
        )
        ivar = None if err is None else utils.inverse(r.oute.T)**2
        return Spectrum(
            r.outx, r.outy.T, ivar=ivar, gpm=r.outf.T > pixel_fraction_threshold, meta=self.meta,
        )


def fit_spectrum_bspline(
    spec, bkspace=None, resolution=2700., nresln=20., region_mask=None, maxiter=35, upper=3.0,
    lower=3.0
):
    r"""
    Fit a bspline model to the continuum of a spectrum.

    Parameters
    ----------
    spec : :class:`~pypeit.core.spectrum.Spectrum`
        The spectrum to fit.  Note that the good-pixel mask of the spectrum is
        used to ignore pixels during the fit; see also ``region_mask``
    bkspace : :obj:`float`, optional
        The spacing in angstroms between breakpoints in the bspline; see
        :func:`fit_spectrum_bspline_breakpoints`.  If provided, ``resolution``
        and ``nresln`` are ignored.  If None, ``resolution`` and ``nresln`` must
        be provided and these are used to set the breakpoints.
    resolution : :obj:`int`, :obj:`float`, optional
        The resolution of the spectrum (:math:`R=\lambda/\Delta\lambda`).  If
        ``bkspace`` is not provided, the combination of ``resolution`` and
        ``nresln`` are used to set the breakpoint spacing; they are ignored
        otherwise.
    nresln : :obj:`int`, :obj:`float`, optional
        The number of resolution elements between adjacent breakpoints.  If
        ``bkspace`` is not provided, the combination of ``resolution`` and
        ``nresln`` are used to set the breakpoint spacing; they are ignored
        otherwise.
    region_mask : `numpy.ndarray`_, optional
        A :math:`(N_{\rm mask},2)` array with starting and ending wavelengths
        for a set of spectral regions to mask during the fit.  See
        :func:`~pypeit.core.wavemask.build_wavelength_gpm`.  If None, the
        masking only incorporates the good-pixel mask of the spectrum.
    maxiter : :obj:`int`, optional
        Maximum number of fit and rejection iterations to perform.
        See :func:`~pypeit.bspline.bspline.iterfit`.
    upper : :obj:`int`, :obj:`float`, optional
        Number of sigma used for rejecting positive residuals during bspline fitting.
    lower : :obj:`int`, :obj:`float`, optional
        Number of sigma used for rejecting negative residuals during bspline fitting.

    Returns
    -------
    fit_gpm : `numpy.ndarray`_
        Boolean array (good-pixel mask) selecting pixels that were initially
        included in the bspline fit.  Note this can be different from
        ``fit_rej_gpm``, which excludes measurements that are rejected during
        the iterative fitting procedure.  Shape matches ``spec``.
    fit_gpm_rej : `numpy.ndarray`_
        Same as ``fit_gpm``, except that measurements rejected by the iterative
        fitting procedures have been flagged as bad.  Shape matches ``spec``.
    bspl : :class:`~pypeit.bspline.bspline.bspline`
        Best-fitting bspline model.  To sample the model at the observed
        wavelengths, use ``bspl.value(spec.wave)`` (see
        :func:`~pypeit.bspline.bspline.bspline.value`).
    """
    # TODO: I think changes need to be made to the lines below to enable the
    # function to work on an multi-vector spectrum.

    # Construct the good-pixel mask to use while fitting
    if region_mask is None:
        fit_gpm = spec.gpm.copy()
    else:
        fit_gpm = spec.gpm & wavemask.build_wavelength_gpm(spec.wave, region_mask)

    # Set the bspline breakpoints
    init_breakpoints = fit_spectrum_bspline_breakpoints(
        spec.wave, gpm=spec.gpm, fit_gpm=fit_gpm, bkspace=bkspace, resolution=resolution,
        nresln=nresln
    )

    # Perform the fit
    # TODO:
    #   - remove hardcoding of maxrej
    #   - pass all djs_reject parameters?
    kwargs_reject = {'maxrej': 5}
    bspl, fit_gpm_rej = fitting.iterfit(
        spec.wave, spec.flux, invvar=spec.ivar, inmask=fit_gpm, upper=upper, lower=lower,
        fullbkpt=init_breakpoints, maxiter=maxiter, kwargs_reject=kwargs_reject
    )

    # Return the results
    return fit_gpm, fit_gpm_rej, bspl


def fit_spectrum_bspline_breakpoints(
    wave, gpm=None, fit_gpm=None, bkspace=None, resolution=None, nresln=None
):
    """
    Create the vector of breakpoints for fitting a spectrum.

    Parameters
    ----------
    wave : `numpy.ndarray`_
        Vector of observed wavelengths in angstroms.
    gpm : `numpy.ndarray`_, optional
        Good-pixel mask selecting wavelength regions where the measurements are
        good.  Shape must match ``wave``.  If None, all pixels are assumed to be
        good.
    fit_gpm : `numpy.ndarray`_, optional
        Good-pixel mask selecting wavelength regions to include in the fit.
        Shape must match ``wave``.  If None, this is assumed to be identical to
        ``gpm``.
    bkspace : :obj:`float`, optional
        The spacing in angstroms between breakpoints in the bspline`.  If
        provided, ``resolution`` and ``nresln`` are ignored.  If None,
        ``resolution`` and ``nresln`` must be provided and these are used to set
        the breakpoints.
    resolution : :obj:`int`, :obj:`float`, optional
        The resolution of the spectrum (:math:`R=\lambda/\Delta\lambda`).  If
        ``bkspace`` is not provided, the combination of ``resolution`` and
        ``nresln`` are used to set the breakpoint spacing; they are ignored
        otherwise.
    nresln : :obj:`int`, :obj:`float`, optional
        The number of resolution elements between adjacent breakpoints.  If
        ``bkspace`` is not provided, the combination of ``resolution`` and
        ``nresln`` are used to set the breakpoint spacing; they are ignored
        otherwise.

    Returns
    -------
    :class:`numpy.ndarray`
        A vector with the breakpoint locations in angstroms; i.e., this is
        ``fullbkpt`` in :func:`~pypeit.bspline.bspline.iterfit`.
    """
    # Only use the valid wavelengths
    _wave = wave if gpm is None else wave[gpm]

    if bkspace is None:
        if resolution is None or nresln is None:
            raise PypeItError(
                'If not providing breakpoint spacing when fitting spectra, you must provide the '
                'resolution and the number of resolution elements between breakpoints (nresln).'
            )
        dw = np.diff(sampling.centers_to_borders(wave))
        dw_pix = np.median(dw)
        dw_res = np.median(_wave/resolution)
        if nresln * dw_res < dw_pix:
            _nresln = 2 * dw_pix / dw_res
            log.warning(
                'Nominal breakpoint spacing is less than one pixel.  Adjusting the number of '
                f'resolution elements from {nresln:.1f} to {_nresln:.1f}.'
            )
        else:
            _nresln = nresln
        _bkspace = dw_res * _nresln
        log.info(f'Median wavelength step per pixel: {dw_pix:.2f} Å')
        log.info(f'Median wavelength step per resolution element: {dw_res:.2f} Å')
    else:
        _bkspace = bkspace
    log.info(f'Breakpoint spacing: {_bkspace:.2f} Å')

    # Control the set of breakpoints used
    init_bspline = bspline.bspline(_wave, bkspace=_bkspace)
    if fit_gpm is None:
        return init_bspline.breakpoints

    _fit_gpm = fit_gpm if gpm is None else fit_gpm[gpm]
    # remove masked regions from breakpoints
    msk_bkpt = interpolate.interp1d(
        _wave, _fit_gpm.astype(float), kind='nearest', fill_value='extrapolate'
    )
    return init_bspline.breakpoints[msk_bkpt(init_bspline.breakpoints) > 0.999]


def fit_spectrum_bspline_qa(spec, fit_gpm, fit_gpm_rej, bspl, ofile=None):
    """
    Quality assessment plot for the spectrum bspline modeling.

    Parameters
    ----------
    spec : :class:`~pypeit.core.spectrum.Spectrum`
        Observed spectrum.
    fit_gpm : `numpy.ndarray`_
        Boolean array (good-pixel mask) selecting pixels that were initially
        included in the bspline fit.  Shape matches ``spec``.
    fit_gpm_rej : `numpy.ndarray`_
        Same as ``fit_gpm``, except that measurements rejected by the iterative
        fitting procedures have been flagged as bad.  Shape matches ``spec``.
    bspl : :class:`~pypeit.bspline.bspline.bspline`
        Best-fitting bspline model.
    ofile : :obj:`str`, `Path`_, optional
        If provided, the plot is written to a file.  If None, the plot is shown
        in a matplotlib window.
    """

    bspl_model, bspl_model_gpm = bspl.value(spec.wave)
    bspl_model = np.ma.MaskedArray(bspl_model, mask=np.logical_not(bspl_model_gpm))
    bspl_model_bkpt = bspl.value(bspl.breakpoints)[0]
    fit_bpm = np.logical_not(fit_gpm)
    # The data rejected during the fit
    fit_rejected = fit_gpm & np.logical_not(fit_gpm_rej)

    wflux = np.amax(spec.flux) - np.amin(spec.flux)
    cflux = (np.amax(spec.flux) + np.amin(spec.flux))/2
    flux_lim = [cflux - 1.1 * wflux / 2, cflux + 1.1 * wflux / 2]
    wave_lim = [np.amin(spec.wave), np.amax(spec.wave)]

    dflux = spec.flux - bspl_model
    mean_dflux = np.mean(dflux[fit_gpm])
    sdev_dflux = np.std(dflux[fit_gpm])
    dflux_lim = [mean_dflux - 5 * sdev_dflux, mean_dflux + 5 * sdev_dflux]

    # Set figure
    w,h = pyplot.figaspect(1)
    fig = pyplot.figure(figsize=(3*w,1.5*h))

    ax = fig.add_axes([0.08, 0.3, 0.90, 0.68])
    ax.minorticks_on()
    ax.tick_params(which='major', length=8, direction='in', top=True, right=True)
    ax.tick_params(which='minor', length=4, direction='in', top=True, right=True)
    ax.grid(True, which='major', color='0.9', zorder=0, linestyle='-')
    ax.set_xlim(wave_lim)
    ax.set_ylim(flux_lim)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.text(-0.05, 0.5, 'Zeropoint (AB mag)', ha='center', va='center', rotation='vertical',
            transform=ax.transAxes)

    ax.plot(spec.wave, spec.flux,
            drawstyle='steps-mid', color='black', label='Zeropoint Data', zorder=2)
    ax.plot(spec.wave, bspl_model,
            color='cornflowerblue', label='Bspline fit', linewidth=1.0, zorder=3)
    ax.scatter(spec.wave[fit_bpm], spec.flux[fit_bpm],
                marker='+', color='red', s=5, label='masked on input', zorder=5)
    ax.scatter(spec.wave[fit_rejected], spec.flux[fit_rejected],
                marker='x', color='pink', s=5, label='rejected by fit', zorder=4)
    ax.scatter(bspl.breakpoints, bspl_model_bkpt,
                marker= '.', color='cyan', s=8, label='breakpoints', zorder=10)
    ax.plot(spec.wave, 1.0 / np.sqrt(spec.ivar), color='orange', label='1-sigma error')

    pyplot.legend()

    ax = fig.add_axes([0.08, 0.1, 0.90, 0.2])
    ax.minorticks_on()
    ax.tick_params(which='major', length=8, direction='in', top=True, right=True)
    ax.tick_params(which='minor', length=4, direction='in', top=True, right=True)
    ax.grid(True, which='major', color='0.9', zorder=0, linestyle='-')
    ax.set_xlim(wave_lim)
    ax.set_ylim(dflux_lim)
    ax.text(-0.05, 0.5, 'Residuals (AB mag)', ha='center', va='center', rotation='vertical',
            transform=ax.transAxes)
    ax.text(0.5, -0.25, 'Wavelength (Angstroms)', ha='center', va='center',
            transform=ax.transAxes)

    ax.plot(spec.wave, dflux, drawstyle='steps-mid', color='black', zorder=2)
    ax.scatter(spec.wave[fit_bpm], dflux[fit_bpm],
                marker='+', color='red', s=5, zorder=5)
    ax.scatter(spec.wave[fit_rejected], dflux[fit_rejected],
                marker='x', color='pink', s=5, zorder=4)
    ax.scatter(bspl.breakpoints, np.zeros(bspl.breakpoints.size),
                marker= '.', color='cyan', s=8, zorder=10)
    ax.plot(spec.wave, 1.0 / np.sqrt(spec.ivar), color='orange')

    if ofile is None:
        pyplot.show()
    else:
        fig.canvas.print_figure(ofile, bbox_inches='tight')
    fig.clear()
    pyplot.close(fig)

