"""
Module implementing a class that determines the telluric correction for an
observed spectrum.
"""

from pathlib import Path

from IPython import embed
import numpy as np

from pypeit import datamodel
from pypeit import loader
from pypeit import log
from pypeit import PypeItError
from pypeit import telluric
from pypeit.par import pypeitpar


class TelluricCorrection(datamodel.DataContainer):
    """
    Determine the telluric correction for an observed spectrum.

    Parameters
    ----------
    specfile : str, :class:`Path`
        PypeIt output file that contains 1D spectra to correct.
    par : :class:`~pypeit.par.pypeitpar.TelluricPar`
        The parameters used to determine the telluric correction.
    extract : str, optional
        The extraction used to produce the spectrum.  Must be either None,
        ``'BOX'`` (for a boxcar extraction), or ``'OPT'`` for optimal
        extraction.  If None, the optimal extraction will be returned, if it
        exists, otherwise the boxcar extraction will be returned.  Only relevant
        if the input is a spec1d file.
    fluxed : bool, optional
        If True, return the flux-calibrated spectrum, if it exists.  If the flux
        calibration hasn't been performed or ``fluxed=False``, the spectrum is
        returned in counts.  Only relevant if the input is a spec1d file.
    chk_version : :obj:`bool`, optional
        When reading in existing files written by PypeIt, perform strict
        version checking to ensure a valid file.  If False, the code will
        try to keep going, but this may lead to faults and quiet failures.
        User beware!
    """

    version = '1.0.0'

    datamodel = {}

    internals = [
        'par', 'head', 'spectra', 'nspec', 'wave_min', 'wave_max', 'tel_model', 'src_model',
        'fitter'
    ]

    def __init__(self, specfile, par, extract=None, fluxed=False, chk_version=True):

        # Instantiate as an empty DataContainer
        super().__init__()

        # Check the input parameter object
        if not isinstance(par, pypeitpar.TelluricPar):
            raise PypeItError('Must provide a TelluricPar object to TelluricCorrection.')
        self.par = par

        # Load the spectral data
        self.head, self.spectra = loader.load_spectra(
            specfile, extract=extract, fluxed=fluxed, chk_version=chk_version
        )
        self.nspec = len(self.spectra)

        # Get the wavelength range of all the spectra
        self.wave_min = np.min([np.min(s.wave) for s in self.spectra])
        self.wave_max = np.max([np.max(s.wave) for s in self.spectra])

        # Initialize the telluric model components
        self.tel_model = None
        self._init_telluric_model()
        self.src_model = None
        self._init_source_model()

        # Initialize the fitter
        self.fitter = telluric.fitter.ObservedSourceModelFitter(self.src_model, self.tel_model)

    def _init_telluric_model(self):
        """
        Initialize the desired telluric model.
        """
        match self.par['tel_type']:
            case 'pca':
                self.tel_model = telluric.model.PCATelluricModel(
                    self.par['tel_file'], npca=self.par['tel_npca'],
                    wave_min=self.wave_min/1.1, wave_max=self.wave_max*1.1
                )
            case 'grid':
                self.tel_model = telluric.model.AtmGridTelluricModel(
                    self.par['tel_file'], wave_min=self.wave_min/1.1,
                    wave_max=self.wave_max*1.1
                )
            case _:
                raise PypeItError(f'Telluric type must be pca or grid, not {self.par["tel_type"]}!')

    def _init_source_model(self):
        """
        Initialize the source model.
        """
        # NOTE: The telluric model is currently needed to set the wavelengths
        # for the source model.
        if self.tel_model is None:
            raise PypeItError('The telluric model must be initialized before the source model!')

        match self.par['src_type']:
            case 'qso':
                self.src_model = telluric.source.QSOPCAModel(
                    self.par['qso_pca_file'], self.par['qso_z'], dz=self.par['qso_dz'],
                    npca=self.par['qso_npca'], wave=self.tel_model.wave,
                    func=self.par['poly_func'], model=self.par['poly_model'],
                    order=self.par['poly_order']
                )
            case 'star':
                self.src_model = telluric.source.StellarSpectrumModel(
                    spectral_type=self.par['star_type'], V_mag=self.par['star_mag'],
                    ra=self.par['star_ra'], dec=self.par['star_dec'], 
#                    tol=20., archives='default',
                    wave=self.tel_model.wave, func=self.par['poly_func'],
                    model=self.par['poly_model'], order=self.par['poly_order']
                )
            case 'poly':
                self.src_model = telluric.source.PolynomialModel(
                    wave=self.tel_model.wave, func=self.par['poly_func'],
                    model=self.par['poly_model'], order=self.par['poly_order']
                )
            case _:
                raise PypeItError(
                    f'Object model must be qso, star, or poly, not {self.par["src_type"]}!'
                )
            
    def fit(self, ispec=None, show=False, debug=False):
        """
        Fit the source + telluric model to an observed spectrum.

        Parameters
        ----------
        ispec : int, array-like, optional
            One or more indices of the spectra to fit.  If None, all spectra
            will be fit.
        show : bool, optional
            Show the final result of the fit.
        debug : bool, optional
            Run in debug mode.

        Returns
        -------
        best_fit_par : :class:`numpy.ndarray`
            The best-fitting parameters for the source + telluric model.
        best_fit_gpm : :class:`numpy.ndarray`
            A good pixel mask for the best-fitting model.
        """
        indx = np.arange(self.nspec) if ispec is None else [ispec]

        # TODO:
        #   - Parse only_orders and sn_clip and apply them to the spectra being fit.
        #   - Decide how to parse the wavelength range to fit, fit_wave_range
        #   - Decide how to apply the masking, spec_mask_files

        #   - Decide how to save the best-fit parameters and models (i.e., the datamodel)

        #   - Do something with the show and debug flags in the fit methods
        #   - Fix the ObservedSourceModelFitter tests
        #   - Add a test of only_orders to the dev-suite

        for i in indx:
            # Get the parameter guesses
            gp = self.fitter.par_guess(
                self.spectra[i], resolution_guess=self.par['resolution_guess']
            )
            # ... and bounds
            bp = self.fitter.par_bounds(
                gp, rel_coeff_bounds=self.par['rel_coeff_bounds'], 
                abs_coeff_bounds=self.par['abs_coeff_bounds'],
                resolution_frac_bounds=self.par['resolution_frac_bounds'],
                pix_shift_bounds=self.par['pix_shift_bounds'],
                # TODO: This is not defined yet!
#                pix_stretch_bounds=self.par['pix_stretch_bounds']
            )

            best_fit_par = self.fitter.fit(
                    self.spectra[i], bp, guess_par=gp, ballsize=self.par['ballsize'],
                    de_par=self.par['diff_evol'], debug=debug, #show=show
            )

            embed()
            exit()

            if self.par['max_rej_iter'] > 0:
                # Perform the fit with rejection iterations
                best_fit_par, best_fit_gpm = self.fitter.iter_fit(
                    self.spectra[i], bp, guess_par=gp, ballsize=self.par['ballsize'],
                    max_rej_iter=self.par['max_rej_iter'], de_par=self.par['diff_evol'],
                    rej_par=self.par['reject'], show=show, debug=debug
                )
            else:
                # ... or without
                best_fit_par = self.fitter.fit(
                    self.spectra[i], bp, guess_par=gp, ballsize=self.par['ballsize'],
                    de_par=self.par['diff_evol'], show=show, debug=debug
                )

