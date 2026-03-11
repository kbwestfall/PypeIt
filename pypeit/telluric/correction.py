"""
Module implementing a class that determines the telluric correction for an
observed spectrum.
"""

from astropy import table
from IPython import embed
import numpy as np

from pypeit import datamodel
# TODO: Add some log messages?
from pypeit import log
from pypeit import PypeItError
from pypeit import telluric
from pypeit.core.spectrum import Spectrum
from pypeit.par import pypeitpar


class TelluricCorrection(datamodel.DataContainer):
    """
    Determine the telluric correction for an observed spectrum.

    Parameters
    ----------
    spectra : list
        List of :class:`~pypeit.core.spectrum.Spectrum` objects to fit.
    par : :class:`~pypeit.par.pypeitpar.TelluricPar`
        The parameters used to determine the telluric correction.
    """

    version = '1.0.0'

    datamodel = {
        'parameters': dict(
            otype=str,
            descr='Full set of user-level parameters used to model the telluric correction.'
        ),
        'tel_npar': dict(
            otype=int, descr='Number of telluric model parameters'
        ),
        'npar': dict(
            otype=int,
            descr=(
                'Total number of model parameters, which is the sum of the number of telluric '
                'model parameters and the number source model parameters.'
            ),
        ),
        'model': dict(
            otype=table.Table, descr='Telluric correction modeling results'
        ),
#                 'airmass': dict(otype=float, descr='Airmass of the observation'),
#                 'exptime': dict(otype=float, descr='Exposure time (s)'),
    }
    """DataContainer datamodel."""

    internals = [
        'par', 'spectra', 'nspec', 'npix', 'wave_min', 'wave_max', 'tel_model', 'src_model',
        'fitter'
    ]

    # TODO: Change the interface to provide the spectra directly, instead of
    # providing a file with spectra to read
    def __init__(self, spectra, par):

        # Instantiate as an empty DataContainer
        super().__init__()

        # Check the input parameter object
        if not isinstance(par, pypeitpar.TelluricPar):
            raise PypeItError('Must provide a TelluricPar object to TelluricCorrection.')
        self.par = par

        # Load the spectral data
        self.spectra = np.atleast_1d(spectra).tolist()
        if not all(isinstance(s, Spectrum) for s in self.spectra):
            raise PypeItError('Must provide Spectrum objects for telluric correction/modeling.')
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
        self.npix = self.fitter.wave.size

        # Initialize elements of the datamodel
        self.parameters = str(self.par.to_dict())
        self.tel_npar = self.tel_model.npar
        self.npar = self.fitter.npar
        self.model = self.empty_model_table(self.nspec, self.npix, self.npar)

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

    # TODO: Can we relax the requirement that all spectra have to be the same length?
    @staticmethod
    def empty_model_table(nfit, npix, npar):
        """
        Construct an empty :class:`astropy.table.Table` for the telluric model
        results.

        Parameters
        ----------

        nfit : :obj:`int`
            The number of spectra to fit.
        npix : :obj:`int`
            The number of pixels in each spectrum; all spectra must be the same
            length.
        npar : :obj:`int`
            Total number of model parameters

        Returns
        -------
        :class:`astropy.table.Table`
            An empty table to hold the results of the telluric correction
            modeling.
        """
        return table.Table(data=[
            # TODO: Currently WAVE is the same for all spectra
            table.Column(
                name='WAVE', dtype=float, length=nfit, shape=(npix,),
                description='Wavelength vector'
            ),
            table.Column(
                name='MODEL', dtype=float, length=nfit, shape=(npix,),
                description='Best-fitting telluric+source model spectrum'
            ),
            table.Column(
                name='MODEL_GPM', dtype=bool, length=nfit, shape=(npix,),
                description='Good-pixel mask for the telluric+source model spectrum'
            ),
            table.Column(
                name='MODEL_REJ', dtype=bool, length=nfit, shape=(npix,),
                description='Pixels rejected during rejection iterations'
            ),
            table.Column(
                name='TEL_MODEL', dtype=float, length=nfit, shape=(npix,),
                description='Best-fitting telluric-only model spectrum'
            ),
            table.Column(
                name='TEL_MODEL_GPM', dtype=float, length=nfit, shape=(npix,),
                description='Good-pixel mask for the best-fitting telluric-only model spectrum'
            ),
            table.Column(
                name='THETA', dtype=float, length=nfit, shape=(npar,),
                description='Best-fitting model parameters'
            ),
            table.Column(
                name='THETA_GUESS', dtype=float, length=nfit, shape=(npar,),
                description='Initial guess for model parameters'
            ),
            table.Column(
                name='THETA_LBOUND', dtype=float, length=nfit, shape=(npar,),
                description='Lower bounds used for the model parameters'
            ),
            table.Column(
                name='THETA_UBOUND', dtype=float, length=nfit, shape=(npar,),
                description='Upper bounds used for the model parameters'
            ),
            table.Column(
                name='FOM', dtype=float, length=nfit,
                description='Figure-of-merit evaluated for the best-fit model'
            ),
            # NOTE: The next table.Column entry forces the initial values of the
            # SUCCESS column to be False.  This is actually true *without*
            # having to instantiate the values (i.e., just be setting the length
            # of the column like all the other columns), but I explicitly do
            # this just to ensure the initial values are as expected.
            # TODO: We may want to change "SUCCESS" this to an integer code that
            # can differentiate between (1) a fit that has not yet been
            # attempted, (2) a failed fit, or (3) a successful fit.
            table.Column(
                name='SUCCESS', dtype=bool, data=[False]*nfit,
                description='Flag that fit was successful'
            ),
            table.Column(
                name='WAVE_MIN', dtype=float, length=nfit,
                description='Minimum wavelength included in the fit'
            ),
            table.Column(
                name='WAVE_MAX', dtype=float, length=nfit,
                description='Maximum wavelength included in the fit'
            )
        ])

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

        Raises
        ------
        PypeItError
            Raised if any of the indices provided by ``ispec`` are not valid.
        """
        indx = np.arange(self.nspec) if ispec is None else np.atleast_1d([ispec])

        bad_indx = (indx < 0) | (indx >= self.nspec)
        if any(bad_indx):
            raise PypeItError(
                f'There are {self.nspec} spectra available to fit.  Indices {indx[bad_indx]} are '
                'not valid.'
            )

        for i in indx:
            # Get the parameter guesses
            self.model['THETA_GUESS'][i] = self.fitter.par_guess(
                self.spectra[i], resolution_guess=self.par['resolution_guess']
            )
            # ... and bounds
            bp = self.fitter.par_bounds(
                self.model['THETA_GUESS'][i], rel_coeff_bounds=self.par['rel_coeff_bounds'], 
                abs_coeff_bounds=self.par['abs_coeff_bounds'],
                resolution_frac_bounds=self.par['resolution_frac_bounds'],
                pix_shift_bounds=self.par['pix_shift_bounds'],
                # TODO: This is not defined yet!
#                pix_stretch_bounds=self.par['pix_stretch_bounds']
            )
            self.model['THETA_LBOUND'][i], self.model['THETA_UBOUND'][i] = np.array(bp).T

            if self.par['max_rej_iter'] > 0:
                # Perform the fit with rejection iterations
                (
                    self.model['THETA'][i], self.model['SUCCESS'][i], self.model['MODEL_REJ'][i]
                ) = self.fitter.iter_fit(
                    self.spectra[i], bp, guess_par=self.model['THETA_GUESS'][i],
                    ballsize=self.par['ballsize'], max_rej_iter=self.par['max_rej_iter'],
                    de_par=self.par['diff_evol'], rej_par=self.par['reject'], show=show,
                    debug=debug
                )
            else:
                # ... or without
                self.model['THETA'][i], self.model['SUCCESS'][i] = self.fitter.fit(
                    self.spectra[i], bp, guess_par=self.model['THETA_GUESS'][i],
                    ballsize=self.par['ballsize'], de_par=self.par['diff_evol'], show=show,
                    debug=debug
                )

            # Save the results
            #   - The full, source+telluric model
            (
                self.model['WAVE'][i], self.model['MODEL'][i], self.model['MODEL_GPM'][i]
            ) = self.fitter.sample(self.model['THETA'][i])
            #   - The telluric-only model
            _, self.model['TEL_MODEL'][i], self.model['TEL_MODEL_GPM'][i] = self.tel_model.sample(
                self.model['THETA'][i][-self.tel_model.npar:]
            )
            #   - The figure-of-merit and wavelength limits
            self.model['FOM'][i] = self.fitter.fit_fom(self.model['THETA'][i])
            self.model['WAVE_MIN'][i] = self.wave_min
            self.model['WAVE_MAX'][i] = self.wave_max

    # TODO:
    #   - Parse only_orders and sn_clip and apply them to the spectra being fit.
    #   - Decide how to parse the wavelength range to fit, fit_wave_range
    #   - Decide how to apply the masking, spec_mask_files

    #   - Fix the ObservedSourceModelFitter tests
    #   - Add a test of only_orders to the dev-suite

    #   - Add a vet test to the dev-suite that checks the results of tellfit
    #     when run on FRB180924_opt.fits in the gemini_gmos_gs_ham/R400_700
    #     dataset.

    #   - Add a method that applies the telluric correction to a spectrum

