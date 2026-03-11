""" Module for fluxing routines

.. include common links, assuming primary doc root is up one directory
.. include:: ../include/links.rst

"""

from IPython import embed

import numpy as np

from scipy import interpolate

from matplotlib import pyplot as plt
from matplotlib import ticker

from astropy import units
from astropy import constants
from astropy import table

from pypeit import log
from pypeit import PypeItError
from pypeit import utils
from pypeit import bspline
from pypeit import io
from pypeit import sampling
from pypeit.core.wavecal import wvutils
from pypeit.core import atmextinction
from pypeit.core import fitting
from pypeit.core import spectrum
from pypeit.core import wavemask
from pypeit import dataPaths


PYPEIT_FLUX_SCALE = 1e-17
r"""
Global variable defining the flux scale used by PypeIt.  Units are
erg/s/cm:math:`^2`/Angstrom.
"""


def zp_unit_const():
    """
    This constant defines the units for the spectroscopic zeropoint. See
    :ref:`fluxcalib`.
    """
    return -2.5*np.log10(((units.angstrom**2/constants.c) * 
                          (PYPEIT_FLUX_SCALE*units.erg/units.s/units.cm**2/units.angstrom)
                         ).to('Jy')/(3631 * units.Jy)).value


ZP_UNIT_CONST = zp_unit_const()
"""
Global variable with the spectroscopic zeropoint.

Value is: ZP_UNIT_CONST = 40.092117379602044.
"""


def sensfunc(obs_spec, std_spec, **kwargs):
    """
    Calculate the sensitivity functions for one or more observed spectra given
    the spectrum of the flux standard.

    Parameters
    ----------
    obs_spec : array-like, :class:`~pypeit.core.spectrum.Spectrum`
        One or more observed spectra provided in
        :class:`~pypeit.core.spectrum.Spectrum` objects.
    std_spec : :class:`~pypeit.core.spectrum.Spectrum`
        Flux standard spectrum.
    **kwargs
        Passed directly to
        :func:`~pypeit.core.flux_calib_refactor.standard_zeropoint`.

    Returns
    -------

    """
    # There is only one observed spectrum, so this is a simple wrapper for
    # standard_zeropoint
    if isinstance(obs_spec, spectrum.Spectrum):
        return standard_zeropoint(obs_spec, std_spec.resample(obs_spec.wave), **kwargs)

    raise PypeItError('Entering untested part of the function!!')
#    embed(header='in sensfunc()')
#    exit()

    _obs_spec = np.asarray(obs_spec)
    if not all([isinstance(s, spectrum.Spectrum) for s in _obs_spec]):
        raise PypeItError('Multiple spectra must be provided as a list of pypeit Spectrum objects.')

    results = []
    for _spec in _obs_spec:
        r = sampling.Resample(std_spec.flux, x=std_spec.wave, newx=_obs_spec.wave, conserve=False)
        _std_spec = spectrum.Spectrum(r.outx, r.outy, gpm=r.outf > 0.8)
        results += [list(standard_zeropoint(_obs_spec, _std_spec, **kwargs))]
    return tuple([r.tolist() for r in np.asarray(results).T])


def get_sensfunc_factor(wave, wave_zp, zeropoint, exptime, tellmodel=None, delta_wave=None, extinct_correct=False,
                         airmass=None, longitude=None, latitude=None, extinctfilepar=None, extrap_sens=False):
    """
    Get the final sensitivity function factor that will be multiplied into a spectrum in units of counts to flux calibrate it.
    This code interpolates the sensitivity function and can also multiply in extinction and telluric corrections.

    FLAM, FLAM_SIG, and FLAM_IVAR are generated

    Args:
        wave (float `numpy.ndarray`_): shape = (nspec,)
            Wavelength vector for the spectrum to be flux calibrated
        wave_zp (float `numpy.ndarray`_):
            Zeropoint wavelength vector shape = (nsens,)
        zeropoint (float `numpy.ndarray`_): shape = (nsens,)
            Zeropoint, i.e. sensitivity function
        exptime (float):
            Exposure time in seconds
        tellmodel (float `numpy.ndarray`_, optional):
            Apply telluric correction if it is passed it (shape = (nspec,)).
            Note this is only used to generate the std fluxed QA plot. It should be None otherwise.
            To telluric correct the data, use the telluric correct method.
        delta_wave (float, `numpy.ndarray`_, optional):
            The wavelength sampling of the spectrum to be flux calibrated.
        extinct_correct (bool, optional)
            If True perform an extinction correction. Default = False
        airmass (float, optional):
            Airmass used if extinct_correct=True. This is required if extinct_correct=True
        longitude (float, optional):
            longitude in degree for observatory
            Required for extinction correction
        latitude:
            latitude in degree for observatory
            Required for extinction correction
        extinctfilepar (str):
            [sensfunc][UVIS][extinct_file] parameter
            Used for extinction correction
        extrap_sens (bool, optional):
            Extrapolate the sensitivity function (instead of crashing out)

    Returns
    -------
    sensfunc_factor: `numpy.ndarray`_
        This quantity is defined to be sensfunc_interp/exptime/delta_wave. shape = (nspec,)

    """
    # Initialise some variables
    zeropoint_obs = np.zeros_like(wave)
    wave_mask = wave > 1.0  # filter out masked regions or bad wavelengths
    if delta_wave is not None:
        # Check that the delta_wave is the same size as the wave vector
        if isinstance(delta_wave, float):
            _delta_wave = delta_wave
        elif isinstance(delta_wave, np.ndarray):
            if wave.size != delta_wave.size:
                raise PypeItError('The wavelength vector and delta_wave vector must be the same size')
            _delta_wave = delta_wave
        else:
            log.warning('Invalid type for delta_wave - using a default value')
            _delta_wave = wvutils.get_delta_wave(wave, wave_mask)
    else:
        # If delta_wave is not passed in, then we will use the native wavelength sampling of the spectrum
        _delta_wave = wvutils.get_delta_wave(wave, wave_mask)

#    print(f'get_sensfunc_factor: {np.amin(wave_zp):.1f}, {np.amax(wave_zp):.1f}, '
#          f'{np.amin(wave[wave_mask]):.1f}, {np.amax(wave[wave_mask]):.1f}')

    try:
        zeropoint_obs[wave_mask] \
                = interpolate.interp1d(wave_zp, zeropoint, bounds_error=True)(wave[wave_mask])
    except ValueError:
        if extrap_sens:
            zeropoint_obs[wave_mask] \
                    = interpolate.interp1d(wave_zp, zeropoint, bounds_error=False)(wave[wave_mask])
            log.warning("Your data extends beyond the bounds of your sensfunc. You should be "
                      "adjusting the par['sensfunc']['extrap_blu'] and/or "
                      "par['sensfunc']['extrap_red'] to extrapolate further and recreate your "
                      "sensfunc. But we are extrapolating per your direction. Good luck!")
        else:
            raise PypeItError(
                'Your data extends beyond the bounds of your sensfunc.  Adjust the '
                'par["sensfunc"]["extrap_blu"] and/or par["sensfunc"]["extrap_red"] to '
                'extrapolate further and recreate your sensfunc.'
            )

    # This is the S_lam factor required to convert N_lam = counts/sec/Ang to
    # F_lam = 1e-17 erg/s/cm^2/Ang, i.e.  F_lam = S_lam*N_lam
    sensfunc_obs = Nlam_to_Flam(wave, zeropoint_obs)

    # Telluric corrections used here only to generate the std fluxed QA plot
    # Did the user request a telluric correction?
    if tellmodel is not None:
        # This assumes there is a separate telluric key in this dict.
        #log.warning("Telluric corrections via this method are deprecated")
        log.info('Applying telluric correction')
        sensfunc_obs = sensfunc_obs * (tellmodel > 1e-10) / (tellmodel + (tellmodel < 1e-10))

    if extinct_correct:
        # Apply Extinction if optical bands
        log.info("Applying extinction correction")
        log.warning("Extinction correction applied only if the spectra covers <10000Ang.")
        # Get the atmospheric extinction
        if extinctfilepar == 'closest':
            atmext = atmextinction.AtmosphericExtinction.from_coordinates(longitude, latitude)
        else:
            atmext = atmextinction.AtmosphericExtinction.from_file(extinctfilepar)
        senstot = sensfunc_obs * atmext.correction_factor(wave, airmass=airmass)
    else:
        senstot = sensfunc_obs.copy()


    # senstot is the conversion from N_lam to F_lam, and the division by exptime and delta_wave are to convert
    # the spectrum in counts/pixel into units of N_lam = counts/sec/angstrom
    return senstot/exptime/_delta_wave


# These are physical limits on the allowed values of the zeropoint in magnitudes
def eval_zeropoint(theta, func, wave, wave_min, wave_max, log10_blaze_func_per_ang=None):
    """
    Evaluate the zeropoint model.

    Parameters
    ----------
    theta : `numpy.ndarray`_
        Parameter vector for the zeropoint model
    func : callable
        Function for the zeropoint model from the set of available functions in
        :func:`~pypeit.core.fitting.evaluate_fit`.
    wave : `numpy.ndarray`_, shape = (nspec,)
        Wavelength vector for zeropoint. 
    wave_min : float
        Minimum wavelength for the zeropoint fit to be passed as an argument to
        :func:`~pypeit.core.fitting.evaluate_fit`
    wave_max : float
        Maximum wavelength for the zeropoint fit to be passed as an argument to
        :func:`~pypeit.core.fitting.evaluate_fit`
    log10_blaze_func_per_ang : `numpy.ndarray`_, optional, shape = (nspec,)
        Log10 blaze function per angstrom. This option is used if the zeropoint
        model is relative to the non-parametric blaze function determined from
        flats. The blaze function is defined on the wavelength grid wave. 

    Returns
    -------
    zeropoint : `numpy.ndarray`_, shape = (nspec,)
        Zeropoint evaluated on the wavelength grid wave.
    """
    poly_model = fitting.evaluate_fit(theta, func, wave, minx=wave_min, maxx=wave_max)
    zeropoint = poly_model - 5.0 * np.log10(wave) + ZP_UNIT_CONST
    if log10_blaze_func_per_ang is not None:
        zeropoint += 2.5*log10_blaze_func_per_ang

    return zeropoint


def Nlam_to_Flam(wave, zeropoint, zp_min=5.0, zp_max=30.0):
    r"""
    The factor that when multiplied into N_lam 
    converts to F_lam, i.e. S_lam where S_lam \equiv F_lam/N_lam

    Parameters
    ----------
    wave: `numpy.ndarray`_
       Wavelength vector for zeropoint
    zeropoint: `numpy.ndarray`_
       zeropoint
    zp_min: float, optional
       Minimum allowed value of the ZP. For smaller values the S_lam factor is set to zero
    zp_max: float, optional
       Maximum allowed value of the ZP. For larger values the S_lam factor is set to zero

    Returns
    -------
    factor: `numpy.ndarray`_
         S_lam factor

    """
    gpm = (wave > 1.0) & (zeropoint > zp_min) & (zeropoint < zp_max)
    factor = np.zeros_like(wave)
    factor[gpm] = np.power(10.0, -0.4*(zeropoint[gpm] - ZP_UNIT_CONST))/np.square(wave[gpm])
    return factor


def Flam_to_Nlam(wave, zeropoint, zp_min=5.0, zp_max=30.0):
    r"""
    The factor that when multiplied into F_lam converts to N_lam, 
    i.e. 1/S_lam where S_lam \equiv F_lam/N_lam


    Parameters
    ----------
    wave: `numpy.ndarray`_
       Wavelength array, float, shape (nspec,)
    zeropoint: `numpy.ndarray`_
       zeropoint array, float, shape (nspec,)

    Returns
    -------
    factor: `numpy.ndarray`_
        Factor that when multiplied into F_lam converts to N_lam, i.e. 1/S_lam

    """
    gpm = (wave > 1.0) & (zeropoint > zp_min) & (zeropoint < zp_max)
    factor = np.zeros_like(wave)
    factor[gpm] = np.power(10.0, 0.4*(zeropoint[gpm] - ZP_UNIT_CONST))*np.square(wave[gpm])
    return factor


def zeropoint_to_throughput(wave, zeropoint, eff_aperture):
    """
    Routine to compute the spectrograph throughput from the zeropoint and effective aperture.

    Parameters
    ----------
    wave: `numpy.ndarray`_
         Wavelength array shape (nspec,) or (nspec, norders)
    zeropoint: `numpy.ndarray`_
         Zeropoint array shape (nspec,) or (nspec, norders)
    eff_aperture: float
         Effective aperture of the telescope in m^2. See spectrograph object

    Returns
    -------
    throughput: `numpy.ndarray`_
        Throughput of the spectroscopic setup. 
        Same shape as wave and zeropoint

    """

    eff_aperture_m2 = eff_aperture*units.m**2
    S_lam_units = PYPEIT_FLUX_SCALE*units.erg/units.cm**2
    # Set the throughput to be -1 in places where it is not defined.
    throughput = np.full_like(zeropoint, -1.0)
    zeropoint_gpm = (zeropoint > 5.0) & (zeropoint < 30.0) & (wave > 1.0)
    inv_S_lam = Flam_to_Nlam(wave[zeropoint_gpm], zeropoint[zeropoint_gpm])/S_lam_units
    inv_wave = utils.inverse(wave[zeropoint_gpm])/units.angstrom
    thru = ((constants.h*constants.c)*inv_wave/eff_aperture_m2*inv_S_lam).decompose()
    throughput[zeropoint_gpm] = thru
    return throughput


def zeropoint_qa_plot(wave, zeropoint_data, zeropoint_data_gpm, zeropoint_fit, zeropoint_fit_gpm, title='Zeropoint QA', axis=None, show=False):
    """
    QA plot for zeropoint

    Parameters
    ----------
    wave : `numpy.ndarray`_
        Wavelength array
    zeropoint_data : `numpy.ndarray`_
        Zeropoint data array
    zeropoint_data_gpm : boolean `numpy.ndarray`_
        Good pixel mask array for zeropoint_data
    zeropoint_fit : `numpy.ndarray`_
        Zeropoint fitting array
    zeropoint_fit_gpm : boolean `numpy.ndarray`_
        Good pixel mask array for zeropoint_fit
    title : str, optional
        Title for the QA plot
    axis : `matplotlib.axes.Axes`_, optional
        axis used for ploting.  If None, a new plot is created
    show : bool, optional
        Whether to show the QA plot
    """

    wv_gpm = wave > 1.0
    if axis is None:
        plt.close()
        fig = plt.figure(figsize=(12,8))
        axis = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    rejmask = zeropoint_data_gpm[wv_gpm] & np.logical_not(zeropoint_fit_gpm[wv_gpm])
    axis.plot(wave[wv_gpm], zeropoint_data[wv_gpm], label='Zeropoint estimated', drawstyle='steps-mid', color='k', alpha=0.7, zorder=5, linewidth=1.0)
    axis.plot(wave[wv_gpm], zeropoint_fit[wv_gpm], label='Zeropoint fit', color='red', linewidth=2.0, zorder=7, alpha=0.7)
    axis.plot(wave[wv_gpm][rejmask], zeropoint_data[wv_gpm][rejmask], 's', zorder=2, mfc='None', mec='blue', mew=0.7, label='rejected pixels from fit')
    axis.plot(wave[wv_gpm][np.logical_not(zeropoint_data_gpm[wv_gpm])], zeropoint_data[wv_gpm][np.logical_not(zeropoint_data_gpm[wv_gpm])], 'v',
             zorder=1, mfc='None', mec='orange', mew=0.7, label='originally masked')
    med_filt_mask = zeropoint_data_gpm[wv_gpm] & np.isfinite(zeropoint_data[wv_gpm])
    zp_med_filter = utils.fast_running_median(zeropoint_data[wv_gpm][med_filt_mask], 11)
    axis.set_ylim(0.95 * zp_med_filter.min(), 1.05 * zp_med_filter.max())
    axis.legend()
    axis.set_xlabel('Wavelength')
    axis.set_ylabel('Zeropoint (AB mag)')
    axis.set_title(title, fontsize=12)
    if show:
        plt.show()


def standard_zeropoint(obs_spec, std_spec, exptime=1., atm_extinction=None, airmass=1.,
                       telluric_model=None, bkspace=None, resolution=2700., nresln=20.,
                       region_mask=None, maxiter=35, upper=3.0, lower=3.0):
    r"""
    Generate a sensitivity function based on observed flux and standard spectrum.

    Parameters
    ----------
    obs_spec : :class:`~pypeit.core.spectrum.Spectrum`
        Observed spectrum.  The input wavelength and flux units are expected to
        be angstroms and counts, respectively.  The spectrum is expected to be a
        single vector.  Note that the good-pixel mask is used to ignore pixels
        during the fit; see also ``region_mask``.
    std_spec : :class:`~pypeit.core.spectrum.Spectrum`
        Standard, flux calibrated spectrum.  Flux must be in :math:`10^{-17}
        {\rm erg/s/cm}^2/\AA`.  Must be sampled at the same wavelengths as the
        observed spectrum.
    exptime : :obj:`float`, optional
        Exposure time in seconds.
    atm_extinction : :class:`~pypeit.core.atmextinction.AtmosphericExtinction`, optional
        Atmospheric extinction profile.  If None, the observed spectrum is
        assumed to already been corrected for atmospheric extinction.
    airmass : :obj:`float`, optional
        The airmass of the observation used to calculate the atmospheric
        extinction correction factor; see
        :func:`~pypeit.core.atmextinction.AtmosphericExtinction.correction_factor`.
    telluric_model : :class:`numpy.ndarray`, optional
        A model of the telluric spectrum sampled at the same wavelengths as the
        observed spectrum.  This used to remove the telluric signatures in the
        observed spectrum.  Note that if both ``atm_extinction`` and
        ``telluric_model`` are provided, they are *both* used in the zeropoint
        calculation.
    bkspace : :obj:`float`, optional
        The spacing in angstroms between breakpoints in the bspline used to fit
        the sensitivity function zeropoints; see :func:`zeropoint_breakpoints`.
        If None, ``resolution`` and ``nresln`` must be provided.  If provided,
        ``resolution`` and ``nresln`` are ignored.
    resolution : :obj:`int`, :obj:`float`, optional
        The spectral resolution of the *observed* data.  The combination of
        ``resolution`` and ``nresln`` are used to set the breakpoint spacing.
        If ``bkspace`` is provided, both ``resolution`` and ``nresln`` are
        ignored.
    nresln : :obj:`int`, :obj:`float`, optional
        The number of resolution elements between adjacent breakpoints.  The
        combination of ``resolution`` and ``nresln`` are used to set the
        breakpoint spacing in angstroms.  If ``bkspace`` is provided, both
        ``resolution`` and ``nresln`` are ignored.
    region_mask : `numpy.ndarray`_, optional
        A :math:`(N_{\rm mask},2)` array with starting and ending wavelengths
        for a set of spectral regions to mask during the zeropoint fitting.  See
        :func:`~pypeit.core.wavemask.build_wavelength_gpm`.
    maxiter : :obj:`int`, optional
        Maximum number of fit and rejection iterations for the bspline fitting.
        See :func:`~pypeit.bspline.bspline.iterfit`.
    upper : :obj:`int`, :obj:`float`, optional
        Number of sigma used for rejecting positive residuals during bspline fitting.
    lower : :obj:`int`, :obj:`float`, optional
        Number of sigma used for rejecting negative residuals during bspline fitting.

    Returns
    -------
    zp_spec : :class:`~pypeit.core.spectrum.Spectrum`
        Measured spectrum of zeropoints.
    fit_gpm : `numpy.ndarray`_
        Boolean array (good-pixel mask) selecting pixels that were initially
        included in the bspline fit.  Note this can be different from
        ``fit_rej_gpm``, which excludes measurements that are rejected during
        the iterative fitting procedure.  Shape matches ``zp_spec``.
    fit_gpm_rej : `numpy.ndarray`_
        Same as ``fit_gpm``, except that measurements rejected by the iterative
        fitting procedures have been flagged as bad.  Shape matches ``zp_spec``.
    zp_bspl : :class:`~pypeit.bspline.bspline.bspline`
        Best-fitting bspline model for the zeropoints.  To sample the model at
        the observed wavelengths, use ``bspl.value(obs_spec.wave)`` (see
        :func:`~pypeit.bspline.bspline.bspline.value`).
    """
    zp_spec = calculate_zeropoint(
        obs_spec, std_spec, exptime=exptime, atm_extinction=atm_extinction, airmass=airmass,
        telluric_model=telluric_model
    )
    fit_gpm, fit_gpm_rej, zp_bspl = fit_zeropoint(
        zp_spec, bkspace=bkspace, resolution=resolution, nresln=nresln, region_mask=region_mask,
        maxiter=maxiter, upper=upper, lower=lower
    )
    return zp_spec, fit_gpm, fit_gpm_rej, zp_bspl


def calculate_zeropoint(obs_spec, std_spec, exptime=1., atm_extinction=None, airmass=1.,
                        telluric_model=None, relative_throughput=None):
    r"""
    Calculate the flux zeropoints based on observed flux and standard spectrum.

    Parameters
    ----------
    obs_spec : :class:`~pypeit.core.spectrum.Spectrum`
        Observed spectrum.  The input wavelength and flux units are expected to
        be angstroms and counts/electrons per pixel, respectively.  The spectrum
        is expected to be a single vector.
    std_spec : :class:`~pypeit.core.spectrum.Spectrum`
        Standard, flux calibrated spectrum.  Flux must be in :math:`10^{-17}
        {\rm erg/s/cm}^2/\AA` and it must be sampled at the same wavelengths as
        the observed spectrum.
    exptime : :obj:`float`, optional
        Exposure time in seconds.
    atm_extinction : :class:`~pypeit.core.atmextinction.AtmosphericExtinction`, optional
        Atmospheric extinction profile.  If None, the observed spectrum is
        assumed to already been corrected for atmospheric extinction.
    airmass : :obj:`float`, optional
        The airmass of the observation used to calculate the atmospheric
        extinction correction factor; see
        :func:`~pypeit.core.atmextinction.AtmosphericExtinction.correction_factor`.
    telluric_model : :class:`numpy.ndarray`, optional
        A model of the telluric spectrum sampled at the same wavelengths as the
        observed spectrum.  This used to remove the telluric signatures in the
        observed spectrum.  Note that if both ``atm_extinction`` and
        ``telluric_model`` are provided, they are *both* used in the zeropoint
        calculation.
    relative_throughput : :class:`numpy.ndarray`, optional
        The normalized throughput of the spectrum.

    Returns
    -------
    :class:`~pypeit.core.spectrum.Spectrum`
        Measured spectrum of zeropoints.
    """
    # Check the input
    if not isinstance(obs_spec, spectrum.Spectrum):
        raise PypeItError('Must provide observed spectrum as a Spectrum object.')
    if obs_spec.ndim != 1:
        raise PypeItError('Must provide a single observed spectrum.')
    if not isinstance(std_spec, spectrum.Spectrum):
        raise PypeItError('Must provide standard spectrum as a Spectrum object.')
    if not np.allclose(obs_spec.wave, std_spec.wave):
        raise PypeItError(
            'Standard spectrum is expected to be sampled at the same wavelengths as the observed '
            'spectrum.'
        )
    if telluric_model is not None and telluric_model.shape != obs_spec.shape:
        raise PypeItError(
            'Telluric model must be sampled at the same wavelengths as the observed spectrum.'
        )
    if relative_throughput is not None and relative_throughput.shape != obs_spec.shape:
        raise PypeItError(
            'Relative throughput must be sampled at the same wavelengths as the observed spectrum.'
        )
    
    # The calculations below use the relevant spectrum.Spectrum methods to
    # propagate errors.

    # Convert observed spectrum to counts/s/angstrom
    dw = np.diff(sampling.centers_to_borders(obs_spec.wave))
    zp_spec = obs_spec.copy()
    zp_spec.multiply(1./exptime/dw) # This performs the error propagation

    # Correct for the atmospheric extinction
    if atm_extinction is not None:
        zp_spec.multiply(atm_extinction.correction_factor(zp_spec.wave, airmass=airmass))
    if telluric_model is not None:
        zp_spec.multiply(1./telluric_model)

    # Correct for relative throughput variations
    if relative_throughput is not None:
        zp_spec.multiply(1./relative_throughput)

    # Compute the zeropoint at each wavelength.
    zp_spec.inverse()
    zp_spec.multiply(std_spec)
    zp_spec.multiply(zp_spec.wave**2)
    zp_spec.to_magnitude(zeropoint=ZP_UNIT_CONST)

    return zp_spec


# TODO: It would be straight-forward to generalize this to fit a generic
# spectrum.  I.e., there's nothing in this function that's specific to flux
# calibration.
def fit_zeropoint(zp_spec, bkspace=None, resolution=2700., nresln=20., region_mask=None,
                  maxiter=35, upper=3.0, lower=3.0):
    r"""
    Fit a bspline model to the measured flux zeropoints.

    Parameters
    ----------
    zp_spec : :class:`~pypeit.core.spectrum.Spectrum`
        Measured spectrum of zeropoints.  Note that the good-pixel mask of the
        spectrum is used to ignore pixels during the fit; see also
        ``region_mask``
    bkspace : :obj:`float`, optional
        The spacing in angstroms between breakpoints in the bspline used to fit
        the sensitivity function zeropoints; see :func:`zeropoint_breakpoints`.
        If None, ``resolution`` and ``nresln`` must be provided.  If provided,
        ``resolution`` and ``nresln`` are ignored.
    resolution : :obj:`int`, :obj:`float`, optional
        The spectral resolution of the *observed* data.  The combination of
        ``resolution`` and ``nresln`` are used to set the breakpoint spacing.
        If ``bkspace`` is provided, both ``resolution`` and ``nresln`` are
        ignored.
    nresln : :obj:`int`, :obj:`float`, optional
        The number of resolution elements between adjacent breakpoints.  The
        combination of ``resolution`` and ``nresln`` are used to set the
        breakpoint spacing in angstroms.  If ``bkspace`` is provided, both
        ``resolution`` and ``nresln`` are ignored.
    region_mask : `numpy.ndarray`_, optional
        A :math:`(N_{\rm mask},2)` array with starting and ending wavelengths
        for a set of spectral regions to mask during the zeropoint fitting.  See
        :func:`~pypeit.core.wavemask.build_wavelength_gpm`.
    maxiter : :obj:`int`, optional
        Maximum number of fit and rejection iterations for the bspline fitting.
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
        the iterative fitting procedure.  Shape matches ``zp_spec``.
    fit_gpm_rej : `numpy.ndarray`_
        Same as ``fit_gpm``, except that measurements rejected by the iterative
        fitting procedures have been flagged as bad.  Shape matches ``zp_spec``.
    zp_bspl : :class:`~pypeit.bspline.bspline.bspline`
        Best-fitting bspline model for the zeropoints.  To sample the model at
        the observed wavelengths, use ``bspl.value(obs_spec.wave)`` (see
        :func:`~pypeit.bspline.bspline.bspline.value`).
    """
    # TODO: I think changes need to be made to the lines below to enable the
    # function to work on an multi-vector spectrum.

    # Construct the good-pixel mask to use while fitting
    if region_mask is None:
        fit_gpm = zp_spec.gpm.copy()
    else:
        fit_gpm = zp_spec.gpm & wavemask.build_wavelength_gpm(zp_spec.wave, region_mask)

    # Set the bspline breakpoints
    init_breakpoints = zeropoint_breakpoints(
        zp_spec.wave, gpm=zp_spec.gpm, fit_gpm=fit_gpm, bkspace=bkspace, resolution=resolution,
        nresln=nresln
    )
    log.info(f'Number of breakpoints: {init_breakpoints.size}')

    # Perform the fit
    kwargs_reject = {'maxrej': 5}
    zp_bspl, fit_gpm_rej = fitting.iterfit(
        zp_spec.wave, zp_spec.flux, invvar=zp_spec.ivar, inmask=fit_gpm, upper=upper, lower=lower,
        fullbkpt=init_breakpoints, maxiter=maxiter,
        kwargs_reject=kwargs_reject
    )

    # Return the results
    return fit_gpm, fit_gpm_rej, zp_bspl


def zeropoint_breakpoints(wave, gpm=None, fit_gpm=None, bkspace=None, resolution=None, nresln=None):
    """
    Create the vector of breakpoints for fitting zeropoints.

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
        The spacing in angstroms between breakpoints in the bspline used to fit
        the sensitivity function zeropoints.  If None, ``resolution`` and
        ``nresln`` must be provided.  If provided, ``resolution`` and ``nresln``
        are ignored.
    resolution : :obj:`int`, :obj:`float`, optional
        The spectral resolution of the *observed* data.  The combination of
        ``resolution`` and ``nresln`` are used to set the breakpoint spacing.
        If ``bkspace`` is provided, both ``resolution`` and ``nresln`` are
        ignored.
    nresln : :obj:`int`, :obj:`float`, optional
        The number of resolution elements between adjacent breakpoints.  The
        combination of ``resolution`` and ``nresln`` are used to set the
        breakpoint spacing in angstroms.  If ``bkspace`` is provided, both
        ``resolution`` and ``nresln`` are ignored.

    Returns
    -------
    `numpy.ndarray`_
        A vector with the breakpoint locations in angstroms; i.e., this is
        ``fullbkpt`` in :func:`~pypeit.bspline.bspline.iterfit`.
    """
    # Only use the valid wavelengths
    _wave = wave if gpm is None else wave[gpm]

    if bkspace is None:
        if resolution is None or nresln is None:
            raise PypeItError('If not providing breakpoint spacing, must provide resolution and the '
                       'number of resolution elements between breakpoints (nresln).')
        dw = np.diff(sampling.centers_to_borders(wave))
        std_pix = np.median(dw)
        std_res = np.median(_wave/resolution)
        if nresln * std_res < std_pix:
            _nresln = 2 * std_pix / std_res
            log.warning('Nominal breakpoint spacing is less than one pixel.  Adjusting the number '
                      f'of resolution elements from {nresln:.1f} to {_nresln:.1f}.')
        else:
            _nresln = nresln
        _bkspace = std_res * _nresln
        log.info(f'Median wavelength step per pixel: {std_pix:.2f} Å')
        log.info(f'Median wavelength step per resolution element: {std_res:.2f} Å')
    else:
        _bkspace = bkspace
    log.info(f'Breakpoint spacing: {_bkspace:.2f} Å')

    # Control the set of breakpoints used
    init_bspline = bspline.bspline(_wave, bkspace=_bkspace)
    if fit_gpm is None:
        return init_bspline.breakpoints

    _fit_gpm = fit_gpm if gpm is None else fit_gpm[gpm]
    # remove masked regions from breakpoints
    msk_bkpt = interpolate.interp1d(_wave, _fit_gpm.astype(float), kind='nearest',
                                    fill_value='extrapolate')
    return init_bspline.breakpoints[msk_bkpt(init_bspline.breakpoints) > 0.999]


def standard_zeropoint_qa(zp_spec, fit_gpm, fit_gpm_rej, zp_bspl, ofile=None):
    """
    Quality assessment plot for the zeropoint modeling.

    Parameters
    ----------
    zp_spec : :class:`~pypeit.core.spectrum.Spectrum`
        Measured spectrum of zeropoints.
    fit_gpm : `numpy.ndarray`_
        Boolean array (good-pixel mask) selecting pixels that were initially
        included in the bspline fit.  Shape matches ``zp_spec``.
    fit_gpm_rej : `numpy.ndarray`_
        Same as ``fit_gpm``, except that measurements rejected by the iterative
        fitting procedures have been flagged as bad.  Shape matches ``zp_spec``.
    zp_bspl : :class:`~pypeit.bspline.bspline.bspline`
        Best-fitting bspline model.
    ofile : :obj:`str`, `Path`_, optional
        If provided, the plot is written to a file.  If None, the plot is shown
        in a matplotlib window.
    """

    zp_model, zp_model_gpm = zp_bspl.value(zp_spec.wave)
    zp_model = np.ma.MaskedArray(zp_model, mask=np.logical_not(zp_model_gpm))
    zp_model_bkpt = zp_bspl.value(zp_bspl.breakpoints)[0]
    fit_bpm = np.logical_not(fit_gpm)
    # The data rejected during the fit
    fit_rejected = fit_gpm & np.logical_not(fit_gpm_rej)

    wflux = np.amax(zp_spec.flux) - np.amin(zp_spec.flux)
    cflux = (np.amax(zp_spec.flux) + np.amin(zp_spec.flux))/2
    flux_lim = [cflux - 1.1 * wflux / 2, cflux + 1.1 * wflux / 2]
    wave_lim = [np.amin(zp_spec.wave), np.amax(zp_spec.wave)]

    dflux = zp_spec.flux - zp_model
    mean_dflux = np.mean(dflux[fit_gpm])
    sdev_dflux = np.std(dflux[fit_gpm])
    dflux_lim = [mean_dflux - 5 * sdev_dflux, mean_dflux + 5 * sdev_dflux]

    # Set figure
    w,h = plt.figaspect(1)
    fig = plt.figure(figsize=(3*w,1.5*h))

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

    ax.plot(zp_spec.wave, zp_spec.flux,
            drawstyle='steps-mid', color='black', label='Zeropoint Data', zorder=2)
    ax.plot(zp_spec.wave, zp_model,
            color='cornflowerblue', label='Bspline fit', linewidth=1.0, zorder=3)
    ax.scatter(zp_spec.wave[fit_bpm], zp_spec.flux[fit_bpm],
                marker='+', color='red', s=5, label='masked on input', zorder=5)
    ax.scatter(zp_spec.wave[fit_rejected], zp_spec.flux[fit_rejected],
                marker='x', color='pink', s=5, label='rejected by fit', zorder=4)
    ax.scatter(zp_bspl.breakpoints, zp_model_bkpt,
                marker= '.', color='cyan', s=8, label='breakpoints', zorder=10)
    ax.plot(zp_spec.wave, 1.0 / np.sqrt(zp_spec.ivar), color='orange', label='1-sigma error')

    plt.legend()

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

    ax.plot(zp_spec.wave, dflux, drawstyle='steps-mid', color='black', zorder=2)
    ax.scatter(zp_spec.wave[fit_bpm], dflux[fit_bpm],
                marker='+', color='red', s=5, zorder=5)
    ax.scatter(zp_spec.wave[fit_rejected], dflux[fit_rejected],
                marker='x', color='pink', s=5, zorder=4)
    ax.scatter(zp_bspl.breakpoints, np.zeros(zp_bspl.breakpoints.size),
                marker= '.', color='cyan', s=8, zorder=10)
    ax.plot(zp_spec.wave, 1.0 / np.sqrt(zp_spec.ivar), color='orange')

    if ofile is None:
        plt.show()
    else:
        fig.canvas.print_figure(ofile, bbox_inches='tight')
    fig.clear()
    plt.close(fig)


def load_filter_file(filter):
    """
    Load a system response curve for a given filter.
    All supported filters can be found at `pypeit.data.filters`_

    Parameters
    ----------
    filter: str
        Name of filter

    Returns
    -------
    wave: `numpy.ndarray`_
        wavelength in units of Angstrom
    instr: `numpy.ndarray`_
        filter throughput

    """

    filter_file = dataPaths.filters.get_file_path('filter_list.ascii')
    tbl = table.Table.read(filter_file, format='ascii')

    allowed_options = tbl['filter'].data

    # Check
    if filter not in allowed_options:
        raise PypeItError("PypeIt is not ready for filter = {}".format(filter))

    trans_file = dataPaths.filters.get_file_path('filtercurves.fits')
    trans = io.fits_open(trans_file)
    wave = trans[filter].data['lam']  # Angstroms
    instr = trans[filter].data['Rlam']  # Am keeping in atmospheric terms
    keep = instr > 0.
    # Parse
    wave = wave[keep]
    instr = instr[keep]

    # Return
    return wave, instr

# TODO Replace this stuff wth calls to the astropy speclite package.
def scale_in_filter(wave, flux, gpm, scale_dict):
    """
    Scale spectra to input magnitude in a given filter

    Parameters
    ----------
    wave : `numpy.ndarray`_
        spectral wavelength array
    flux : `numpy.ndarray`_
        flux density array
    gpm : boolean `numpy.ndarray`_
        Good pixel mask array
    scale_dict : :class:`~pypeit.par.pypeitpar.Coadd1DPar`
        Object with filter and magnitude data.

    Returns
    -------
    scale : float
        scale value for the flux, i.e. ``newflux = flux * scale``
    """

    # Mask further?
    if scale_dict['filter_mask'] is not None:
        # Funny formatting
        if isinstance(scale_dict['filter_mask'], str):
            regions = scale_dict['filter_mask'].split(',')
        else:
            regions = scale_dict['filter_mask']
        for region in regions:
            mask = region.split(':')
            gpm[(wave > float(mask[0])) & (wave < float(mask[1]))] = False
    mag_type = scale_dict['mag_type']

    # Parse the spectrum
    wave = wave[gpm]
    flux = flux[gpm]

    # Grab the instrument response function
    log.info("Integrating spectrum in filter: {}".format(scale_dict['filter']))
    fwave, trans = load_filter_file(scale_dict['filter'])
    tfunc = interpolate.interp1d(fwave, trans, bounds_error=False, fill_value=0.)

    # TODO this expression below is incorrect for irregular gridded wavelengths. FIX
    # Convolve
    allt = tfunc(wave)
    wflam = np.sum(flux*allt)/np.sum(allt)* PYPEIT_FLUX_SCALE*units.erg/units.s/units.cm**2/units.AA

    mean_wv = np.sum(fwave*trans)/np.sum(trans) * units.AA

    #
    if mag_type == 'AB':
        # Convert flam to AB magnitude
        fnu = wflam * mean_wv**2 / constants.c
        # Apparent AB
        AB = -2.5 * np.log10(fnu.to('erg/s/cm**2/Hz').value) - 48.6
        # Scale factor
        Dm = AB - scale_dict['filter_mag']
        scale = np.power(10.0,(Dm/2.5))
        log.info("Scaling spectrum by {}".format(scale))
    else:
        raise PypeItError("Bad magnitude type")

    return scale

