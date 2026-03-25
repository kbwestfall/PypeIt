"""
Implements the objects used to construct sensitivity functions.

.. include:: ../include/links.rst
"""
from pathlib import Path

from astropy.io import fits
from astropy import table
from IPython import embed
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import interpolate

from pypeit import datamodel
from pypeit import io
from pypeit import log
from pypeit import PypeItError
from pypeit import specobjs
from pypeit import specobj
from pypeit import utils
from pypeit.core import atmextinction
from pypeit.core import coadd
from pypeit.core import flux_calib_refactor
from pypeit.core import telluric
from pypeit.core import spectrum
from pypeit.core import standard
from pypeit.core.wavecal import wvutils
from pypeit.core import meta
from pypeit.core import wavemask
from pypeit.onespec import OneSpec
from pypeit.scripts import loader
from pypeit.spectrographs.util import load_spectrograph


# TODO Add the data model up here as a standard thing using DataContainer.

# TODO Standard output location for sensfunc?

# TODO Add some QA plots, and plots to the screen if show is set.

#        # QA and throughput plot filenames
#        self.qafile = sensfile.replace('.fits', '') + '_QA.pdf'
#        self.thrufile = sensfile.replace('.fits', '') + '_throughput.pdf'
#        self.fstdfile = sensfile.replace('.fits', '') + '_fluxed_std.pdf'

class SensFunc(datamodel.DataContainer):
    r"""
    Base class for generating sensitivity functions from a standard-star
    spectrum.

    This class should not be instantated by itself; instead instantiate
    either :class:`UVISSensFunc` or :class:`IRSensFunc`, depending on the
    wavelength range of your data (UVIS for :math:`\lambda < 7000` angstrom,
    IR for :math:`\lambda > 7000` angstrom.)

    When calculating the sensitivity function using multiple spectra, the
    expectation is that these are multiple spectra of the same object taken
    during a single observation; i.e., they are spectra from multiple orders of
    an echelle or spectra the cross detectors in a multi-slit observation.
    These should *not* be separate observations of the same standard star taken
    at different times.

    The datamodel attributes are:

    .. include:: ../include/class_datamodel_sensfunc.rst

    Parameters
    ----------
    spec1dfiles (:obj:`list`):
        One or more PypeIt spec1d files with the observations of the standard
        star.
    sensfile (:obj:`str`):
        File name for the sensitivity function data.
    par (:class:`~pypeit.par.pypeitpar.SensFuncPar`):
        The parameters required for the sensitivity function computation.
    par_fluxcalib (:class:`~pypeit.par.pypeitpar.FluxCalibratePar`, optional):
        The parameters required for flux calibration. These are only used
        for flux calibration of the standard star spectrum for the QA plot.
        If None, defaults will be used.
    debug (:obj:`bool`, optional):
        Run in debug mode, sending diagnostic information to the screen.
    chk_version (:obj:`bool`, optional):
        Check the version of the data model.
    """
    version = '1.1.0'
    """Datamodel version."""

    # TODO: Add this if we want to set the output float type for the np.ndarray
    # elements of the datamodel...
#    output_float_dtype = np.float32
#    """Regardless of datamodel, output floating-point data have this fixed bit size."""

    datamodel = {
        'PYP_SPEC': dict(otype=str, descr='PypeIt spectrograph name'),
        'pypeline': dict(otype=str, descr='PypeIt pipeline reduction path'),
        'spec1df': dict(otype=str, descr='PypeIt spec1D file(s) used to for sensitivity function'),
        'extr': dict(otype=str, descr='Extraction method used for the standard star (OPT or BOX)'),
        'std_name': dict(otype=str, descr='Type of standard source'),
        'std_cal': dict(otype=str, descr='File name (or shorthand) with the standard flux data'),
        'std_ra': dict(otype=float, descr='RA of the standard source'),
        'std_dec': dict(otype=float, descr='DEC of the standard source'),
        'airmass': dict(otype=float, descr='Airmass of the observation'),
        'exptime': dict(otype=float, descr='Exposure time'),
        'telluric': dict(
            otype=telluric.Telluric,
            descr='Telluric model; see :class:`~pypeit.core.telluric.Telluric`'
        ),
        'sens': dict(otype=table.Table, descr='Table with the sensitivity function'),
        'wave': dict(otype=np.ndarray, atype=float, descr='Wavelength vectors'),
#        'model_flat': dict(
#            otype=np.ndarray, atype=float,
#            descr=(
#                'A smooth model of the flat-field spectrum extracted at the location of the '
#                'standard-star observation'
#            )
#        ),
        'zeropoint': dict(otype=np.ndarray, atype=float, descr='Sensitivity function zeropoints'),
        'gpm': dict(otype=np.ndarray, atype=(bool, np.bool), descr='Good-pixel mask for the zeropoint data'),
        'throughput': dict(
            otype=np.ndarray, atype=float, descr='Spectrograph throughput measurements'
        ),
        # TODO: Subclasses overwrite this with a class attribute.  Not sure what
        # to do here.  Maybe just change the _bundle and _parse methods.
        'algorithm': dict(otype=str, descr='Algorithm used for the sensitivity calculation.')
    }
    """DataContainer datamodel."""

    internals = [
        'spec1d_arr',
        'spectrograph',
        'par',
        'par_fluxcalib',
        'debug',
        'obs_spec',
        'obs_spec_twk',
        'model_flat',
        'relative_throughput',
        'zp_spec',
        'nspec_in',
        'norderdet',
        'wave_splice',
        'zeropoint_splice',
        'throughput_splice',
        'splice_multi_det',
        'std_spec',
        'atmext',
        'region_mask',
    ]

    _algorithm = None
    """Algorithm used for the sensitivity calculation."""

    @staticmethod
    def empty_sensfunc_table(norders, nspec, nspec_in, ncoeff=1):
        """
        Construct an empty `astropy.table.Table`_ for the sensitivity
        function.

        Args:
            norders (:obj:`int`):
                The number of slits/orders on the detector.
            nspec (:obj:`int`):
                The number of spectral pixels for the zeropoint arrays.
            nspec_in (:obj:`int`):
                The number of spectral pixels on the detector for the input
                standard star spectrum.
            ncoeff (:obj:`int`, optional):
                Number of coefficients for smooth model fit to zeropoints

        Returns:
            `astropy.table.Table`_: Instance of the empty sensitivity
            function table.
        """
        return table.Table(data=[
            table.Column(
                name='WAVE', dtype=float, length=norders, shape=(nspec,),
                description='Wavelength vector'
            ),
            table.Column(
                name='ZEROPOINT', dtype=float, length=norders, shape=(nspec,),
                description='Measured sensitivity zero-point data'
            ),
            table.Column(
                name='ZEROPOINT_GPM', dtype=bool, length=norders, shape=(nspec,),
                description='Good-pixel mask for the measured zero points'
            ),
            table.Column(
                name='ZEROPOINT_FIT', dtype=float, length=norders, shape=(nspec,),
                description='Best-fit smooth model to the zero points'
            ),
            table.Column(
                name='ZEROPOINT_FIT_GPM', dtype=bool, length=norders, shape=(nspec,),
                description='Good-pixel mask for the model zero points'
            ),
            table.Column(
                name='MODEL_FLAT', dtype=float, length=norders, shape=(nspec,),
                description=(
                    'Model of a spectrum extracted from the flat-field image at the location of '
                    'the standar-star observation.'
                )
            ),
            table.Column(
                name='REL_THROUGHPUT', dtype=float, length=norders, shape=(nspec,),
                description='Ratio of the observed flat-field spectrum to the best-fitting model'
            ),
            table.Column(
                name='THROUGHPUT', dtype=float, length=norders, shape=(nspec,),
                description=(
                    'Measurement of the telescope+spectrograph system throughput for the '
                    'standard-star observation, measured using the zeropoint model '
                    '(ZEROPOINT_FIT), not the direct measurements.'
                ),
            ),
            table.Column(
                name='WAVE_MIN', dtype=float, length=norders,
                description='Minimum wavelength with direct zeropoint measurements'
            ),
            table.Column(
                name='WAVE_MAX', dtype=float, length=norders,
                description='Maximum wavelength with direct zeropoint measurements'
            ),

            table.Column(name='SENS_COEFF', dtype=float, length=norders, shape=(ncoeff,),
                         description='Coefficients of smooth model fit to zero points'),
            table.Column(name='ECH_ORDERS', dtype=int, length=norders,
                         description='Echelle order for this specrum (echelle data only)'),
            table.Column(name='POLYORDER_VEC', dtype=int, length=norders,
                         description='Polynomial order for each slit/echelle (if applicable)'),
            table.Column(name='SENS_FLUXED_STD_WAVE', dtype=float, length=norders, shape=(nspec_in,),
                         description='The wavelength array for the fluxed standard star spectrum'),
            table.Column(name='SENS_FLUXED_STD_FLAM', dtype=float, length=norders, shape=(nspec_in,),
                         description='The F_lambda for the fluxed standard star spectrum'),
            table.Column(name='SENS_FLUXED_STD_FLAM_IVAR', dtype=float, length=norders, shape=(nspec_in,),
                         description='The inverse variance of F_lambda for the fluxed standard star spectrum'),
            table.Column(name='SENS_FLUXED_STD_MASK', dtype=bool, length=norders, shape=(nspec_in,),
                         description='The good pixel mask for the fluxed standard star spectrum '),
            table.Column(name='SENS_STD_MODEL_FLAM', dtype=float, length=norders, shape=(nspec_in,),
                         description='The F_lambda for the standard model spectrum')])

    # Superclass factory method generates the subclass instance
    @classmethod
    def get_instance(cls, spec1dfiles, par, par_fluxcalib=None, debug=False, chk_version=True):
        """
        Instantiate the relevant subclass based on the algorithm provided in
        ``par``.
        """
        return next(c for c in cls.__subclasses__()
                    if c.__name__ == f"{par['algorithm']}SensFunc")(
                        spec1dfiles, par, par_fluxcalib=par_fluxcalib, debug=debug,
                        chk_version=chk_version)

    def __init__(self, spec1dfiles, par, par_fluxcalib=None, debug=False, chk_version=True):

        # Instantiate as an empty DataContainer
        super().__init__()

        # Input and Output files
        # this is just the name of the spec1d that go into the Sensfunc container
        self.spec1df = ','.join([Path(s).name for s in spec1dfiles])
        # this is the array of spec1d that are used in the calculations
        self.spec1d_arr = np.array(spec1dfiles)
        self.extr = par['extr']
        self.par = par
        # Spectrograph
        header = fits.getheader(self.spec1d_arr[0])
        self.PYP_SPEC = header['PYP_SPEC']
        self.spectrograph = load_spectrograph(self.PYP_SPEC)
        self.pypeline = self.spectrograph.pypeline
        # TODO: This line is necessary until we figure out a way to instantiate
        # spectrograph objects with configuration specific information from
        # spec1d files.
        self.spectrograph.dispname = header['DISPNAME']
        self.par_fluxcalib = self.spectrograph.default_pypeit_par()['fluxcalib'] \
                                if par_fluxcalib is None else par_fluxcalib

        # Set the algorithm in the datamodel
        # TODO: Why do we need this both here and in par?
        self.algorithm = self.__class__._algorithm

        # Other
        self.debug = debug

        # Are we splicing together multiple detectors?
        # TODO: Which one do we want?  This or what is returned by load_standard
        self.splice_multi_det = self.par['multi_spec_det'] is not None

        # TODO: Note that by default this gets the unfluxed spectra and tries to
        # include the flat.  The latter will fault for any onespec spectra.

        # NOTE: load_standard *always* returns a list of spectra, even if the
        # list only has one spectrum.
        self.obs_spec, self.splice_multi_det = loader.load_standard(
            spec1dfiles, extract=self.par['extr'], include_flat=True,
            multi_spec_det=self.par['multi_spec_det'], chk_version=chk_version
        )
        self.norderdet = len(self.obs_spec)

        # TODO: Add wave_range as a parameter?
        self.obs_spec_twk = self.spectrograph.tweak_standard(
            self.obs_spec, trim_std_pixs=self.par['trim_std_pixs']
        )

        # Get metadata that must be the same for all spectra.  These data will
        # be contained in the `meta` dictionary of each Spectrum, unless they
        # are unavailable.  See Spectrograph.parse_spec_header, used by
        # sobjs.to_spectrum.

        # Exptime and airmass are part of the datamodel
        self.exptime = self.obs_spec.get_global_meta('EXPTIME')
        if self.exptime is None:
            log.warning(
                'Exposure time for the standard star observation is not available!  This may '
                'cause the code to fault.'
            )
        self.airmass = self.obs_spec.get_global_meta('AIRMASS')
        if self.airmass is None:
            log.warning(
                'The airmass during the standard star observation is not available!  This may '
                'cause the code to fault.'
            )

        # If the user provided RA and DEC use those instead of what is in meta
        star_ra = (
            self.obs_spec.get_global_meta('RA')
            if self.par['star_ra'] is None else self.par['star_ra']
        )
        star_dec = (
            self.obs_spec.get_global_meta('DEC')
            if self.par['star_dec'] is None else self.par['star_dec']
        )
        if star_ra is None or star_dec is None:
            # TODO: We need to ensure this does not happen, and we should allow
            # users to define the name and "archive" of the observed standard so
            # that it can be pulled directly without having to match the
            # coordinates.
            raise PypeItError('Unable to determine the RA/Dec of the standard star observed.')

        # Convert to decimal deg, as needed
        star_ra, star_dec = meta.convert_radec(star_ra, star_dec)

        # Get the archive standard star spectrum
        self.std_spec = standard.get_standard_spectrum(
            spectral_type=self.par['star_type'], V_mag=self.par['star_mag'], ra=star_ra,
            dec=star_dec
        )

        # Add the components of the datamodel.  These are added to the standard
        # star meta dictionary when the spectra are loaded.  See
        # pypeit.core.standard.ArchivedFluxStandard._init_meta.
        self.std_cal = self.std_spec.meta['source']
        self.std_name = self.std_spec.meta['Name']
        self.std_ra = self.std_spec.meta['ra_deg']
        self.std_dec = self.std_spec.meta['dec_deg']

        # Check if this is the right standard star for the observation, i.e., if
        # there is overlap in the wavelength coverage between the archival and
        # observed standard star spectrum
        frac_overlap = np.array([
            np.sum(
                (s.wave[s.gpm] >= np.min(self.std_spec.wave))
                & (s.wave[s.gpm] <= np.max(self.std_spec.wave))
            ) / np.sum(s.gpm) for s in self.obs_spec
        ])

        if np.all(np.isclose(frac_overlap, 0.)):
            raise PypeItError(
                 'No wavelength overlap between the archival and observed standard star '
                 'spectrum. This is not the right standard star for your observations.'
            )
        elif np.any(np.isclose(frac_overlap, 0.)):
            log.warning(
                'Some of the observed spectra (e.g. one or more echelle orders) to use to '
                'calculate the sensitivity function do not overlap with the standard star '
                'observation.  These spectra will be ignored during the calculation.'
            )
        elif np.any(frac_overlap < 0.8):
            log.warning(
                 'The observed spectra cover the following fractions of the spectra: '
                 f'{np.round(frac_overlap, decimals=1)}.  Beware of extrapolation errors in the '
                 'calibration.'
            )

        # Get the wavelength regions to mask
        # TODO: Add ability to mask telluric regions
        self.region_mask = wavemask.read_wavelength_masks(par['spec_mask_files'])

        # Get the atmospheric extinction
        # TODO: Move extinct_file into the main sensfunc parameter set
        self.atmext = self.spectrograph.get_atmospheric_extinction(par['UVIS']['extinct_file'])

    # TODO: REVISIT
    def _bundle(self):
        """
        Bundle the object for writing using
        :func:`~pypeit.datamodel.DataContainer.to_hdu`.
        """
        # All of the datamodel elements that can be written as a header are
        # written to the SENS extension.
        d = []
        for key in self.keys():
            if self[key] is None:
                continue
            if isinstance(self[key], datamodel.DataContainer) \
                    or isinstance(self[key], table.Table):
                d += [{key: self[key]}]
                continue
            if self.datamodel[key]['otype'] == np.ndarray:
                # TODO: We might want a general solution in
                # DataContainer that knows to convert (and revert)
                # boolean arrays to np.uint8 types. Does anyone know a
                # way around this? It's annoying that ImageHDUs cannot
                # have boolean type arrays.
                if issubclass(self[key].dtype.type, (bool, np.bool_)):
                    d += [{key: self[key].astype(np.uint8)}]
#                elif issubclass(self[key].dtype.type, (float, np.floating)):
#                    d += [{key: self[key].astype(self.output_float_dtype)}]
                else:
                    d += [{key: self[key]}]

        # If the spliced arrays have been set, use these to replace the
        # multislit/echelle wave, zeropoint, and throughput arrays
        if self.splice_multi_det:
            # TODO: I added this neurotic check, just to make sure...
            if self.wave_splice is None or self.zeropoint_splice is None \
                    or self.throughput_splice is None:
                raise PypeItError('CODING ERROR: Assumed if splice_multi_det is True, then the *_splice '
                           'arrays have all been defined.  Found a case where this is not true!')
            # Loop through this list of dictionaries
            for _d in d:
                if list(_d.keys())[0] not in ['wave', 'zeropoint', 'throughput']:
                    # Keep going if it's not one of the relevant attributes
                    continue
                # Replace with the spliced version.  Warning: This requires that
                # the relevant attribute names are paired as self.attr and
                # self.attr_splice
                attr = list(_d.keys())[0]
                _d[attr] = getattr(self, f'{attr}_splice')

        # Add all the non-array, non-DataContainer, non-Table elements to the
        # 'sens' extension. This is done after the first loop to just to make
        # sure that the order of extensions matches the order in the datamodel.
        for _d in d:
            if list(_d.keys()) != ['sens']:
                continue
            for key in self.keys():
                if self[key] is None or self.datamodel[key]['otype'] == np.ndarray \
                        or isinstance(self[key], datamodel.DataContainer) \
                        or isinstance(self[key], table.Table):
                    continue
                _d[key] = self[key]

        return d

    # TODO: REVISIT
    @classmethod
    def from_hdu(cls, hdu, hdu_prefix=None, chk_version=True):
        """
        Instantiate the object from an HDU extension.

        This overrides the base-class method, essentially just to handle the
        fact that the 'TELLURIC' extension is not called 'MODEL'.

        Args:
            hdu (`astropy.io.fits.HDUList`_, `astropy.io.fits.ImageHDU`_, `astropy.io.fits.BinTableHDU`_):
                The HDU(s) with the data to use for instantiation.
            hdu_prefix (:obj:`str`, optional):
                Maintained for consistency with the base class but is
                not used by this method.
            chk_version (:obj:`bool`, optional):
                If True, raise an error if the datamodel version or
                type check failed. If False, throw a warning only.
        """
        # Run the default parser to get most of the data. This correctly parses
        # everything except for the Telluric.model data table.
        d, version_passed, type_passed, parsed_hdus = cls._parse(hdu, allow_subclasses=True)
        # Check
        cls._check_parsed(version_passed, type_passed, chk_version=chk_version)

        # Load the telluric model, if it exists
        if 'TELLURIC' in [h.name for h in hdu]:
            # Instantiate the Telluric model from the header, if it exists
            # TODO: I don't like this, but this is the fastest work around
            d['telluric'] = telluric.Telluric.from_hdu(hdu['TELLURIC'], ext_pseudo='MODEL',
                                                       chk_version=chk_version)
        # Return the constructed object
        return super().from_dict(d=d)

    def set_model_flat(self):
        """
        Construct a model of the flat-field spectrum extracted at the location
        of the standard-star observations.
        """
        if not self.par['use_flat']:
            self.model_flat = None
            return

        self.model_flat = []
        for s in self.obs_spec_twk:
            try:
                flat_spec = s.assoc_spectrum('flat', copy_gpm=False)
            except (PypeItError, KeyError) as e:
                log.warning(
                    'Unable to model flat-field spectrum for standard star observation.  '
                    f'Continuing with calculation but flat-field spectrum will *not* be used.  '
                    f'Original exception was: {e}'
                )
                break
            flat_fit_gpm, flat_fit_gpm_rej, flat_bspl = spectrum.fit_spectrum_bspline(
                flat_spec, resolution=self.par['UVIS']['resolution'],
                nresln=self.par['UVIS']['nresln'], lower=1., upper=1.
            )
            if self.debug:
                spectrum.fit_spectrum_bspline_qa(
                    flat_spec, flat_fit_gpm, flat_fit_gpm_rej, flat_bspl, ylabel='Flat Flux'
                )
            self.model_flat += [flat_bspl.value(flat_spec.wave)[0]]

        if len(self.model_flat) != len(self.obs_spec_twk):
            self.model_flat = None

    def compute_zeropoint(self):
        """
        Dummy method overloaded by subclasses
        """
        raise PypeItError(
            f'CODING ERROR: This subclass of SensFunc ({self.__class__.__name__}) does not define '
            'the compute_zeropoint method!'
        )

    def run(self):
        """
        Execute the sensitivity function calculations.
        """
        # Get a model of the flat-field spectrum, as used to account for
        # relative throughput variations.
        self.set_model_flat()

        # Compute the sensitivity function
        self.compute_zeropoint()

        embed()
        exit()

        # Regrid the sensitivity function to:
        #   - Extrapolate the wavelength range so that it can be applied to
        #     observations with modestly different spectral range
        #   - Splice together spectra that span multiple detectors or orders
        self.regrid()

        # Flux the standard star with this sensitivity function and add it to the output table
        self.flux_std()

        embed(header='after flux')
        exit()

        # Compute the throughput
        self.throughput, self.throughput_splice = self.compute_throughput()


    def regrid(self):
        """
        Regrid the sensitivity function, splicing together multi-detector
        observations and extrapolating to account for modest differences in
        spectral range for observed spectra.
        """

        # Initialized using the model fit to the zeropoint data
        self.wave = self.sens['WAVE'].data
        self.zeropoint = self.sens['ZEROPOINT_FIT'].data
        self.gpm = self.sens['ZEROPOINT_FIT_GPM'].data
        self.model_flat = self.sens['MODEL_FLAT'].data if self.par['use_flat'] else None

        # Splice together the spectra from multiple detectors
        if self.splice_multi_det:
            self.splice()

        # Extrapolate the wavelength range
        self.extrapolate()

    def flux_std(self):
        """
        Flux the standard star and add it to the sensitivity function table

        """
        # Construct the relative throughput functions
        relative_throughput = None
        if self.par['use_flat']:
            relative_throughput = []
            for i in range(ns):
                try:
                    flat_spec = self.obs_spec[i].assoc['flat']
                except KeyError as e:
                    break
                relative_throughput += [flat_spec / interpolate.interp1d(
                    self.wave[i], self.model_flat[i], bounds_error=False,
                    fill_value='extrapolate'
                )(self.obs_spec.wave[i])]
            if len(relative_throughput) != len(self.obs_spec):
                relative_throughput = None

        ns = len(self.obs_spec)
        fluxed_spec = [None] * ns
        for i in range(len(self.obs_spec)):
            zp_spec = spectrum.Spectrum(
                self.obs_spec.wave, interpolate.interp1d(
                    self.wave[i], self.zeropoint[i], bounds_err=False, fill_value='extrapolate'
                )(self.obs_spec.wave)
            )
            fluxed_spec[i] = flux_calib_refactor.flux_calibrate(
                self.obs_spec[i], zp_spec, exptime=self.exptime, atm_extinction=self.atmext,
                airmass=self.airmass, relative_throughput=relative_throughput[i]
            )

#def flux_calibrate(
    #obs_spec, zp_spec, exptime=1., atm_extinction=None, airmass=1., telluric_model=None,
    #relative_throughput=None,
#):

        # Now flux the standard star
        self.sobjs_std.apply_flux_calib(self.par_fluxcalib, self.spectrograph, self, tell=self.algorithm=='IR')
        # TODO assign this to the data model

        # Unpack the fluxed standard
        _wave, _flam, _flam_ivar, _flam_mask, _blaze, _, _ \
            = self.sobjs_std.unpack_object(ret_flam=True, extract_type=self.extr)
        # Reshape to 2d arrays
        wave, flam, flam_ivar, flam_mask, _, _, _ \
                = utils.spec_atleast_2d(_wave, _flam, _flam_ivar, _flam_mask)
        # Store in the sens table
        self.sens['SENS_FLUXED_STD_WAVE'] = wave.T
        self.sens['SENS_FLUXED_STD_FLAM'] = flam.T
        self.sens['SENS_FLUXED_STD_FLAM_IVAR'] = flam_ivar.T
        self.sens['SENS_FLUXED_STD_MASK'] = flam_mask.T

        #save the model that was used
        model_interp_func = interpolate.interp1d(self.std_spec.wave, self.std_spec.flux, bounds_error=False, fill_value='extrapolate')
        model_flux_sav = np.zeros_like(self.sens['SENS_FLUXED_STD_FLAM'])
        for iorddet in range(self.sens['SENS_FLUXED_STD_WAVE'].shape[0]):
            wave_gpm = self.sens['SENS_FLUXED_STD_WAVE'][iorddet] > 1.0
            model_flux_sav[iorddet][wave_gpm] = model_interp_func(self.sens['SENS_FLUXED_STD_WAVE'][iorddet][wave_gpm])

        self.sens['SENS_STD_MODEL_FLAM'] = model_flux_sav

    def eval_zeropoint(self, wave, indx):
        """
        Evaluate the sensitivity function zero-points at the input wavelength.

        This base-class method should always be overwridden by subclasses.

        Parameters
        ----------
        wave : :class:`numpy.ndarray`
            Wavelengths at which to evaluate the zeropoint
        indx : :obj:`int`, optional
            For calculations based on multiple spectra (cross-dispersed echelles
            or spectra across multiple, independently processed detectors), this
            selects the zeropoint calculation to use.

        Returns
        -------
        :class:`numpy.ndarray`
            Zeropoint data.  Shape is identical to ``wave``.
        """
        raise PypeItError(
            f'CODING ERROR: This subclass of SensFunc ({self.__class__.__name__}) does not define '
            'the eval_zeropoint method!'
        )

    def extrapolate(self, samp_fact=1.5):
        """
        Extrapolates the sensitivity function(s) to cover an extra wavelength range.

        set by the ``extrapl_blu`` and ``extrap_red`` parameters. This is
        important for making sure that the sensitivity function can be applied
        to data with slightly different wavelength coverage etc.

        Parameters
        ----------
        samp_fact : :obj:`float`
            Parameter governing the sampling of the wavelength grid used for the
            extrapolation.

        Returns
        -------
        wave_extrap : `numpy.ndarray`_
            Extrapolated wavelength array
        zeropoint_extrap : `numpy.ndarray`_
            Extrapolated sensitivity function
        """
        # Create a new set of oversampled and padded wavelength grids for the
        # extrapolation
        wave_start = np.min(self.wave, axis=1) * (1.0 - self.par['extrap_blu'])
        wave_end = np.max(self.wave, axis=1) * (1.0 + self.par['extrap_red'])

        dwave = np.array([np.median(np.diff(w)) for w in self.wave])
        nspec = np.max(np.ceil(samp_fact * (wave_end - wave_start) / dwave).astype(int))
        self.wave = np.vstack(tuple(
            np.linspace(s, e, num=nspec) for s, e in zip(wave_start, wave_end)
        ))
        self.zeropoint = np.vstack(tuple(
            self.eval_zeropoint(w, i) for i, w in enumerate(self.wave)
        ))
        if self.par['use_flat']:
            self.model_flat = np.vstack(tuple(
                interpolate.interp1d(
                    self.wave[i], self.model_flat[i], bound_error=False, fill_value='extrapolate'
                ) for i in range(len(self.wave))
            ))
        self.gpm = np.ones(self.wave.shape, dtype=bool)

    def splice(self):
        """
        Routine to splice together sensitivity functions into one global
        sensitivity function for spectrographs with multiple detectors extending
        across the wavelength direction.
        """

        log.info(f"Merging sensfunc for {self.norderdet} detectors {self.par['multi_spec_det']}")
        wave, _, _ = wvutils.get_wave_grid(
            waves=self.wave, wave_method='linear', wave_grid_min=np.amin(self.wave),
            wave_grid_max=np.amax(self.wave), spec_samp_fact=1.0
        )

        # Construct a masked array with all the values initially masked
        self.zeropoint = np.ma.masked_all(wave.size, dtype=float)
        _model_flat = np.ma.masked_all(wave.size, dtype=float) if self.par['use_flat'] else None
        for i in range(self.norderdet):
            indx = (wave >= self.sens['WAVE_MIN'][i]) & (wave <= self.sens['WAVE_MAX'][i])
            # This unmasks the relevant data
            self.zeropoint[indx] = self.eval_zeropoint(wave[indx], i)
            if self.par['use_flat']:
                _model_flat[indx] = interpolate.interp1d(
                    self.wave[i], self.model_flat[i], bounds_error=False,
                    fill_value='extrapolate'
                )(wave[indx])

        # Interpolate over remaining masked pixels
        if np.any(np.ma.getmaskarray(self.zeropoint)):
            log.info('Interpolating/Extrapolating over gaps')
            self.zeropoint = utils.interpolate_masked_vector(self.zeropoint)

        if self.par['use_flat'] and np.any(np.ma.getmaskarray(_model_flat)):
            self.model_flat = utils.interpolate_masked_vector(_model_flat)

        self.wave = np.expand_dims(wave, 0)
        self.zeropoint = np.expand_dims(self.zeropoint, 0)
        self.model_flat = np.expand_dims(self.model_flat, 0) if self.par['use_flat'] else None
        self.gpm = np.ones(self.wave.shape, dtype=bool)

    def compute_throughput(self):
        """
        Compute the spectroscopic throughput

        Returns
        -------
        throughput : `numpy.ndarray`_, :obj:`float`, shape is (nspec, norders)
            Throughput measurements

        throughput_splice : `numpy.ndarray`_, :obj:`float`, shape is (nspec_splice, norders)
            Throughput measurements for spliced spectra
        """

        # Set the throughput to be -1 in places where it is not defined.
        throughput = np.full_like(self.zeropoint, -1.0)
        for idet in range(self.wave.shape[1]):
            wave_gpm = (self.wave[:,idet] >= self.sens['WAVE_MIN'][idet]) \
                            & (self.wave[:,idet] <= self.sens['WAVE_MAX'][idet]) \
                            & (self.wave[:,idet] > 1.0)
            throughput[:,idet][wave_gpm] \
                    = flux_calib_refactor.zeropoint_to_throughput(self.wave[:,idet][wave_gpm],
                                                         self.zeropoint[:,idet][wave_gpm],
                                                         self.spectrograph.telescope.eff_aperture())
        if self.splice_multi_det:
            wave_gpm = (self.wave_splice >= np.amin(self.sens['WAVE_MIN'])) \
                            & (self.wave_splice <= np.amax(self.sens['WAVE_MAX'])) \
                            & (self.wave_splice > 1.0)
            throughput_splice = np.zeros_like(self.wave_splice)
            throughput_splice[wave_gpm] \
                    = flux_calib_refactor.zeropoint_to_throughput(self.wave_splice[wave_gpm],
                                                         self.zeropoint_splice[wave_gpm],
                                                         self.spectrograph.telescope.eff_aperture())
        else:
            throughput_splice = None

        return throughput, throughput_splice

    def write_QA(self):
        """
        Write out zeropoint QA files
        """
        utils.pyplot_rcparams()

        # Plot QA for zeropoint
        if 'Echelle' in self.spectrograph.pypeline:
            order_or_det = self.meta_spec['ECH_ORDERS']
            order_or_det_str = 'order'
        else:
            order_or_det = np.arange(self.norderdet) + 1
            order_or_det_str = 'det'

        spec_str = f' {self.spectrograph.name} {self.spectrograph.pypeline} ' \
                   f'{self.spectrograph.dispname} '
        zp_title = ['PypeIt Zeropoint QA for' + spec_str + order_or_det_str
                    + f'={order_or_det[idet]}' for idet in range(self.norderdet)]
        thru_title = [order_or_det_str + f'={order_or_det[idet]}'
                        for idet in range(self.norderdet)]

        is_odd = self.norderdet % 2 != 0
        npages = int(np.ceil(self.norderdet/2)) if is_odd else self.norderdet//2 + 1

        # TODO: PDF page logic is a bit complicated becauase we want to plot two
        # plots per page, but the number of pages depends on the number of
        # order/det. Consider just dumping out a set of plots or revamp once we
        # have a dashboard.
        with PdfPages(self.qafile) as pdf:
            for ipage in range(npages):
                figure, (ax1, ax2) = plt.subplots(2, figsize=(8.27, 11.69))
                if (2 * ipage) < self.norderdet:
                    flux_calib_refactor.zeropoint_qa_plot(self.sens['SENS_WAVE'][2*ipage],
                                                 self.sens['SENS_ZEROPOINT'][2*ipage],
                                                 self.sens['SENS_ZEROPOINT_GPM'][2*ipage],
                                                 self.sens['SENS_ZEROPOINT_FIT'][2*ipage],
                                                 self.sens['SENS_ZEROPOINT_FIT_GPM'][2*ipage],
                                                 title=zp_title[2*ipage], axis=ax1)
                if (2*ipage + 1) < self.norderdet:
                    flux_calib_refactor.zeropoint_qa_plot(self.sens['SENS_WAVE'][2*ipage+1],
                                                 self.sens['SENS_ZEROPOINT'][2*ipage+1],
                                                 self.sens['SENS_ZEROPOINT_GPM'][2*ipage+1],
                                                 self.sens['SENS_ZEROPOINT_FIT'][2*ipage+1],
                                                 self.sens['SENS_ZEROPOINT_FIT_GPM'][2*ipage+1],
                                                 title=zp_title[2*ipage+1], axis=ax2)

                if self.norderdet == 1:
                    # For single order/det just finish up after the first page
                    ax2.remove()
                    pdf.savefig()
                    plt.close('all')
                elif (self.norderdet > 1) & (ipage < npages-1):
                    # For multi order/det but not on the last page, finish up.
                    # No need to remove ax2 since there are always 2 plots per
                    # page except on the last page
                    pdf.savefig()
                    plt.close('all')
                else:
                    # For multi order/det but on the last page, add order/det
                    # summary plot to the final page Deal with even/odd page
                    # logic for axes
                    if is_odd:
                        # add order/det summary plot to axis 2 of current page
                        axis=ax2
                    else:
                        axis=ax1
                        ax2.remove()
                    # Initialise these variables to None values
                    _wave_min, _wave_max, tmin, tmax = None, None, None, None
                    for idet in range(self.norderdet):
                        # define the color
                        rr = (np.max(order_or_det) - order_or_det[idet]) \
                                / np.maximum(np.max(order_or_det) - np.min(order_or_det), 1)
                        gg = 0.0
                        bb = (order_or_det[idet] - np.min(order_or_det)) \
                                / np.maximum(np.max(order_or_det) - np.min(order_or_det), 1)
                        sens_wave_gpm = self.sens['SENS_WAVE'][idet] > 1.0
                        if not np.any(sens_wave_gpm):
                            continue
                        axis.plot(self.sens['SENS_WAVE'][idet,sens_wave_gpm],
                                  self.sens['SENS_ZEROPOINT_FIT'][idet,sens_wave_gpm],
                                  color=(rr, gg, bb), linestyle='-', linewidth=2.5,
                                  label=thru_title[idet], zorder=5 * idet)

                        wave_gpm = (self.wave[:, idet] >= self.sens['WAVE_MIN'][idet]) \
                                   & (self.wave[:, idet] <= self.sens['WAVE_MAX'][idet]) \
                                   & (self.wave[:, idet] > 1.0)
                        if (_wave_min is None) or (np.min(self.wave[wave_gpm, idet]) < _wave_min):
                            _wave_min = np.min(self.wave[wave_gpm, idet])
                        if (_wave_max is None) or (np.max(self.wave[wave_gpm, idet]) > _wave_max):
                            _wave_max = np.max(self.wave[wave_gpm, idet])
                        if (tmin is None) or (np.min(self.sens['SENS_ZEROPOINT_FIT'][idet,sens_wave_gpm]) < tmin):
                            tmin = np.min(self.sens['SENS_ZEROPOINT_FIT'][idet,sens_wave_gpm])
                        if (tmax is None) or (np.max(self.sens['SENS_ZEROPOINT_FIT'][idet,sens_wave_gpm]) > tmax):
                            tmax = np.fmin(22, np.max(self.sens['SENS_ZEROPOINT_FIT'][idet,sens_wave_gpm]))

                    # If we are splicing, overplot the spliced zeropoint
                    if self.splice_multi_det:
                        wave_slice_gpm = (self.wave_splice >= _wave_min) \
                                            & (self.wave_splice <= _wave_max) \
                                            & (self.wave_splice > 1.0)
                        axis.plot(self.wave_splice[wave_slice_gpm].flatten(),
                                  self.zeropoint_splice[wave_slice_gpm].flatten(), color='black',
                                  linestyle=':', linewidth=2.5, label='Spliced Zeropoint', zorder=30)

                    axis.set_xlim((0.98 * _wave_min, 1.02 * _wave_max))
                    axis.set_ylim((0.95 * tmin, 1.05 * tmax))
                    axis.legend(fontsize=14)
                    axis.set_xlabel('Wavelength (Angstroms)')
                    axis.set_ylabel('Zeropoint (AB mag)')
                    axis.set_title('PypeIt Zeropoints for' + spec_str, fontsize=12)

                    pdf.savefig()
                    plt.close('all')

        # Plot throughput curve(s) for all orders/det
        fig = plt.figure(figsize=(12,8))
        axis = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        # Initialise these variables to None values
        _wave_min, _wave_max, tmax = None, None, None
        for idet in range(self.wave.shape[1]):
            # define the color
            rr = (np.max(order_or_det) - order_or_det[idet]) \
                    / np.maximum(np.max(order_or_det) - np.min(order_or_det), 1)
            gg = 0.0
            bb = (order_or_det[idet] - np.min(order_or_det)) \
                    / np.maximum(np.max(order_or_det) - np.min(order_or_det), 1)
            gpm = (self.throughput[:, idet] >= 0.0)
            if not np.any(gpm):
                continue
            axis.plot(self.wave[gpm,idet], self.throughput[gpm,idet], color=(rr, gg, bb),
                      linestyle='-', linewidth=2.5, label=thru_title[idet], zorder=5*idet)
            # Determine the wavelength limits for the plot
            if (_wave_min is None) or (np.min(self.wave[gpm,idet]) < _wave_min):
                _wave_min = np.min(self.wave[gpm,idet])
            if (_wave_max is None) or (np.max(self.wave[gpm,idet]) > _wave_max):
                _wave_max = np.max(self.wave[gpm,idet])
            if (tmax is None) or (np.max(self.throughput[gpm,idet]) > tmax):
                tmax = np.fmin(0.5, np.max(self.throughput[gpm,idet]))
        if self.splice_multi_det:
            axis.plot(self.wave_splice[wave_slice_gpm].flatten(),
                      self.throughput_splice[wave_slice_gpm].flatten(), color='black',
                      linestyle=':', linewidth=2.5, label='Spliced Throughput', zorder=30)

        axis.set_xlim((0.98*_wave_min, 1.02*_wave_max))
        axis.set_ylim((0.0, 1.05*tmax))
        axis.legend()
        axis.set_xlabel('Wavelength (Angstroms)')
        axis.set_ylabel('Throughput')
        axis.set_title('PypeIt Throughput for' + spec_str)
        fig.savefig(self.thrufile)

        # Plot fluxed standard star for all orders/det
        fig = plt.figure(figsize=(12,8))
        axis = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        axis.plot(self.std_spec.wave, self.std_spec.flux, color='green',linewidth=3.0,
                  label=self.std_spec.meta['Name'], zorder=100, alpha=0.7)
        for iorddet in range(self.sens['SENS_FLUXED_STD_WAVE'].shape[0]):
            # define the color
            rr = (np.max(order_or_det) - order_or_det[iorddet]) \
                    / np.maximum(np.max(order_or_det) - np.min(order_or_det), 1)
            gg = 0.0
            bb = (order_or_det[iorddet] - np.min(order_or_det)) \
                    / np.maximum(np.max(order_or_det) - np.min(order_or_det), 1)
            sens_wave_gpm = (self.sens['SENS_FLUXED_STD_WAVE'][iorddet] > 1.0) & self.sens['SENS_FLUXED_STD_MASK'][iorddet]
            axis.plot(self.sens['SENS_FLUXED_STD_WAVE'][iorddet][sens_wave_gpm], self.sens['SENS_FLUXED_STD_FLAM'][iorddet][sens_wave_gpm],
                      color=(rr, gg, bb), drawstyle='steps-mid', linewidth=1.0,
                      label=thru_title[iorddet], zorder=idet, alpha=0.7)

        if _wave_min is None or _wave_max is None:
            wave_gpm_global = self.sens['SENS_FLUXED_STD_WAVE'] > 1.0
            wave_min = (self.sens['SENS_FLUXED_STD_WAVE'][wave_gpm_global]).min()
            wave_max = (self.sens['SENS_FLUXED_STD_WAVE'][wave_gpm_global]).max()
        else:
            wave_min = 0.98*_wave_min
            wave_max = 1.02*_wave_max
        pix_wave_std = (self.std_spec.wave >= wave_min) & (self.std_spec.wave <= wave_max)
        flux_min = -1.0
        flux_max = 1.10*np.max(self.std_spec.flux[pix_wave_std])
        axis.set_xlim((wave_min, wave_max))
        axis.set_ylim((flux_min, flux_max))
        axis.legend()
        axis.set_xlabel('Wavelength (Angstroms)')
        axis.set_ylabel(r'$f_{{\lambda}}~~~(10^{{-17}}~{{\rm erg~s^{-1}~cm^{{-2}}~\AA^{{-1}}}})$')
        axis.set_title('Fluxed Std Compared to True Spectrum:' + spec_str)
        fig.savefig(self.fstdfile)


    @classmethod
    def sensfunc_weights(cls, sensfile, waves, ech_order_vec=None, debug=False, extrap_sens=True, chk_version=True):
        """
        Get the weights based on the sensfunc

        Args:
            sensfile (str):
                the name of your fits format sensfile
            waves (`numpy.ndarray`_):
                wavelength grid for your output weights.  Shape is (nspec,
                norders, nexp) or (nspec, norders).
            ech_order_vec (`numpy.ndarray`_, optional):
                Vector of echelle orders.  Only used for echelle data.
            debug (bool): default=False
                show the weights QA
            extrap_sens (bool): default=True
                Extrapolate the sensitivity function
            chk_version (:obj:`bool`, optional):
                When reading in existing files written by PypeIt, perform strict
                version checking to ensure a valid file.  If False, the code
                will try to keep going, but this may lead to faults and quiet
                failures.  User beware!

        Returns:
            `numpy.ndarray`_: sensfunc weights evaluated on the input waves
            wavelength grid, shape = same as waves
        """
        sens = cls.from_file(sensfile, chk_version=chk_version)

        if waves.ndim == 2:
            nspec, norder = waves.shape
            if ech_order_vec is not None and ech_order_vec.size != norder:
                log.warning('The number of orders in the wave grid does not match the '
                          'number of orders in the unpacked sobjs. Echelle order vector not used.')
                ech_order_vec = None
            nexp = 1
            waves_stack = np.reshape(waves, (nspec, norder, 1))
        elif waves.ndim == 3:
            nspec, norder, nexp = waves.shape
            waves_stack = waves
        elif waves.ndim == 1:
            nspec = waves.size
            norder, nexp = 1, 1
            waves_stack = np.reshape(waves, (nspec, 1, 1))
        else:
            raise PypeItError('Unrecognized dimensionality for waves')

        weights_stack = np.ones_like(waves_stack)

        if norder != sens.zeropoint.shape[1] and ech_order_vec is None:
            raise PypeItError('The number of orders in {:} does not agree with your data. Wrong sensfile?'.format(sensfile))
        elif norder != sens.zeropoint.shape[1] and ech_order_vec is not None:
            log.warning('The number of orders in {:} does not match the number of orders in the data. '
                      'Using only the matching orders.'.format(sensfile))

        # array of order to loop through
        orders = np.arange(norder) if ech_order_vec is None else ech_order_vec
        for iord,this_ord in enumerate(orders):
            if ech_order_vec is None:
                isens = iord
            # find the index of the sensfunc for this order
            elif np.any(sens.sens['ECH_ORDERS'].value == this_ord):
                isens = np.where(sens.sens['ECH_ORDERS'].value == this_ord)[0][0]
            else:
                # if the order is not in the sensfunc file, skip it
                continue
            for iexp in range(nexp):
                sensfunc_iord = flux_calib_refactor.get_sensfunc_factor(waves_stack[:,iord,iexp],
                                                               sens.wave[:,isens],
                                                               sens.zeropoint[:,isens], 1.0,
                                                               extrap_sens=extrap_sens)
                weights_stack[:,iord,iexp] = utils.inverse(sensfunc_iord)

        if debug:
            coadd.weights_qa(utils.echarr_to_echlist(waves_stack)[0], utils.echarr_to_echlist(weights_stack)[0],
                             utils.echarr_to_echlist(waves_stack > 1.0)[0], title='sensfunc_weights')

        # Reshape to be the same size/dimension as the input spectrum
        if waves.ndim == 2:
            weights_stack = np.reshape(weights_stack, (nspec, norder))
        elif waves.ndim == 1:
            weights_stack = np.reshape(weights_stack, (nspec))

        return weights_stack


# TODO Add a method which optionally merges sensfunc using the nsens > 1 logic


class IRSensFunc(SensFunc):
    r"""
    Determine a sensitivity functions from standard-star spectra. Should only
    be used with NIR spectra (:math:`\lambda > 7000` angstrom).

    Args:
        spec1dfiles (:obj:`str`):
            PypeIt spec1d file for the standard file.
        sensfile (:obj:`str`):
            File name for the sensitivity function data.
        par (:class:`~pypeit.par.pypeitpar.SensFuncPar`, optional):
            The parameters required for the sensitivity function computation.
        debug (:obj:`bool`, optional):
            Run in debug mode.
    """

    _algorithm = 'IR'
    """Algorithm used for the sensitivity calculation."""

    def compute_zeropoint(self):
        """
        Calls routine to compute the sensitivity function.

        Returns
        -------
        TelObj : :class:`~pypeit.core.telluric.Telluric`
            Best-fitting telluric model
        """

        embed(header='in IRSensFunc compute_zerpoint')
        exit()

        self.telluric = telluric.sensfunc_telluric(self.wave_cnts, self.counts, self.counts_ivar,
                                                   self.counts_mask, self.meta_spec['EXPTIME'],
                                                   self.meta_spec['AIRMASS'], self.std_spec,
                                                   self.par['IR']['tel_file'],
                                                   log10_blaze_function=self.log10_blaze_function,
                                                   polyorder=self.par['polyorder'],
                                                   ech_orders=self.meta_spec['ECH_ORDERS'],
                                                   only_orders=self.par['IR']['only_orders'],
                                                   resln_guess=self.par['IR']['resln_guess'],
                                                   resln_frac_bounds=self.par['IR']['resln_frac_bounds'],
                                                   pix_shift_bounds=self.par['IR']['pix_shift_bounds'],
                                                   sn_clip=self.par['IR']['sn_clip'],
                                                   teltype=self.par['IR']['teltype'],
                                                   tell_npca=self.par['IR']['tell_npca'],
                                                   mask_hydrogen_lines=self.par['mask_hydrogen_lines'],
                                                   maxiter=self.par['IR']['maxiter'],
                                                   lower=self.par['IR']['lower'],
                                                   upper=self.par['IR']['upper'],
                                                   delta_coeff_bounds=self.par['IR']['delta_coeff_bounds'],
                                                   minmax_coeff_bounds=self.par['IR']['minmax_coeff_bounds'],
                                                   tol=self.par['IR']['tol'],
                                                   popsize=self.par['IR']['popsize'],
                                                   recombination=self.par['IR']['recombination'],
                                                   polish=self.par['IR']['polish'],
                                                   disp=self.par['IR']['disp'], debug=self.debug,
                                                   debug_init=self.debug)

        # Copy the relevant metadata
        self.std_name = self.telluric.std_name
        self.std_cal = self.telluric.std_cal
        self.std_ra = self.telluric.std_ra
        self.std_dec = self.telluric.std_dec
        self.airmass = self.telluric.airmass
        self.exptime = self.telluric.exptime

        # Instantiate the main output data table
        self.sens = self.empty_sensfunc_table(self.telluric.norders, self.telluric.wave_grid.size, self.nspec_in,
                                              ncoeff=self.telluric.max_ntheta_obj)

        # For stupid reasons related to how astropy tables will let me store
        # this data I have to redundantly write out the wavelength grid for each
        # order.  In actuality a variable length wavelength grid is needed which
        # is a subset of the original fixed grid, but there is no way to write
        # this to disk.  Perhaps the better solution would be a list of of
        # astropy tables, one for each order, that would save some space, since
        # we could then write out only the subset used to the table column. I'm
        # going with this inefficient approach for now, since eventually we will
        # want to create a data container for these outputs. That said, coming
        # up with that standarized model is a bit tedious given the somewhat
        # heterogenous outputs of the two different fluxing algorithms.

        # Extract and construct the relevant data using the telluric model
        self.sens['SENS_COEFF'] = self.telluric.model['OBJ_THETA']
        self.sens['WAVE_MIN'] = self.telluric.model['WAVE_MIN']
        self.sens['WAVE_MAX'] = self.telluric.model['WAVE_MAX']
        self.sens['ECH_ORDERS'] = self.telluric.model['ECH_ORDERS']
        s = self.telluric.model['IND_LOWER']
        e = self.telluric.model['IND_UPPER']+1
        self.sens['POLYORDER_VEC'] = self.telluric.model['POLYORDER_VEC']
        for i in range(self.norderdet):
            if self.telluric.obj_dict_list[i] is None:
                continue
            # Compute and assign the zeropint_data from the input data and the
            # best-fit telluric model
            self.sens['SENS_WAVE'][i,s[i]:e[i]] = self.telluric.wave_grid[s[i]:e[i]]
            if self.log10_blaze_function is not None:
                log10_blaze_function_iord = self.telluric.log10_blaze_func_arr[s[i]:e[i],i]
                self.sens['SENS_LOG10_BLAZE_FUNCTION'][i,s[i]:e[i]] = self.telluric.log10_blaze_func_arr[s[i]:e[i],i]
            else:
                log10_blaze_function_iord = None
            self.sens['SENS_ZEROPOINT_GPM'][i,s[i]:e[i]] = self.telluric.mask_arr[s[i]:e[i],i]
            self.sens['SENS_COUNTS_PER_ANG'][i,s[i]:e[i]] = self.telluric.flux_arr[s[i]:e[i],i]
            N_lam = self.sens['SENS_COUNTS_PER_ANG'][i,s[i]:e[i]] / self.exptime
            self.sens['SENS_ZEROPOINT'][i,s[i]:e[i]], _ \
                    = flux_calib_refactor.compute_zeropoint(self.sens['SENS_WAVE'][i,s[i]:e[i]], N_lam,
                                                   self.sens['SENS_ZEROPOINT_GPM'][i,s[i]:e[i]],
                                                   self.telluric.obj_dict_list[i]['flam_true'],
                                                   tellmodel=self.telluric.tellmodel_list[i])
            # TODO: func is always 'legendre' because that is what's set by
            # sensfunc_telluric
            self.sens['SENS_ZEROPOINT_FIT'][i,s[i]:e[i]] \
                    = flux_calib_refactor.eval_zeropoint(
                self.sens['SENS_COEFF'][i,:self.sens['POLYORDER_VEC'][i]+2],
                self.telluric.func, self.sens['SENS_WAVE'][i,s[i]:e[i]],
                self.sens['WAVE_MIN'][i], self.sens['WAVE_MAX'][i],
                log10_blaze_func_per_ang=log10_blaze_function_iord)
            self.sens['SENS_ZEROPOINT_FIT_GPM'][i,s[i]:e[i]] = self.telluric.outmask_list[i]

    def eval_zeropoint(self, wave, iorddet):
        """
        Evaluate the sensitivity function zero-points at the input wavelength

        Parameters
        ----------
        wave : `numpy.ndarray`_, shape is (nspec)
            Wavelength array
        iorddet : :obj:`int`
            Order or detector (0-indexed)

        Returns
        -------
        zeropoint : `numpy.ndarray`_, shape is (nspec,)
            Zeropoint array evaluated at the input wavelength grid and with the gpm applied.

        """
        s = self.telluric.model['IND_LOWER']
        e = self.telluric.model['IND_UPPER']+1
        # TODO: Not sure what else to do here
        if self.log10_blaze_function is not None:
            log10_blaze_function = interpolate.interp1d(
                self.sens['SENS_WAVE'][iorddet,s[iorddet]:e[iorddet]],
                self.sens['SENS_LOG10_BLAZE_FUNCTION'][iorddet,s[iorddet]:e[iorddet]],
                kind='linear', bounds_error=False, fill_value='extrapolate')(wave)
        else:
            log10_blaze_function = None

        return flux_calib_refactor.eval_zeropoint(
            self.sens['SENS_COEFF'][iorddet,:self.telluric.model['POLYORDER_VEC'][iorddet]+2],
            self.telluric.func, wave, self.sens['WAVE_MIN'][iorddet], self.sens['WAVE_MAX'][iorddet],
            log10_blaze_func_per_ang=log10_blaze_function)


class UVISSensFunc(SensFunc):
    r"""
    Determine a sensitivity functions from standard-star spectra. Should only
    be used with UVIS spectra (:math:`\lambda < 7000` angstrom).

    Args:
        spec1dfiles (:obj:`str`):
            PypeIt spec1d file for the standard file.
        sensfile (:obj:`str`):
            File name for the sensitivity function data.
        par (:class:`~pypeit.par.pypeitpar.SensFuncPar`, optional):
            The parameters required for the sensitivity function computation.
        debug (:obj:`bool`, optional):
            Run in debug mode.
    """

    _algorithm = 'UVIS'
    """Algorithm used for the sensitivity calculation."""

    def compute_zeropoint(self):
        """
        Calls routine to compute the sensitivity function.
        """

        # Get the zeropoints
        # TODO:
        #   - Need to figure out what to do with trans_thresh and polycorrect
        #   - Make parameters that specify the location of the breakpoints (not
        #     just resolution based) available to the user?
        #   - keep the bspline model?
        #   - revisit construction of the sensfunc table
        #       - keep the GPM before any rejection iterations?
        #       - deal with spectra that have different lengths

        # Instantiate the main output data table
        norder = len(self.obs_spec_twk)
        nspec = max([s.size for s in self.obs_spec_twk])
        # TODO: Need to revisit this
        self.sens = self.empty_sensfunc_table(norder, nspec, nspec)

        # Calculate the zeropoints
        for i, _spec in enumerate(self.obs_spec_twk):

            if self.model_flat is None:
                self.sens['MODEL_FLAT'][i,:] = 1.
                self.sens['REL_THROUGHPUT'][i,:] = 1.
            else:
                # If the model flat exists, the original flat spectrum *must* exist.
                # TODO: Perform desired smoothing of the flat (_spec.assoc['flat']) here
                self.sens['MODEL_FLAT'][i] = self.model_flat[i]
                self.sens['REL_THROUGHPUT'][i] = _spec.assoc['flat'] / self.model_flat[i]

            zp_spec, fit_gpm, fit_gpm_rej, zp_bspl = flux_calib_refactor.standard_zeropoint(
                _spec, self.std_spec.resample(_spec.wave), exptime=self.exptime,
                atm_extinction=self.atmext, airmass=self.airmass,
                relative_throughput=self.sens['REL_THROUGHPUT'][i],
                nresln=self.par['UVIS']['nresln'], resolution=self.par['UVIS']['resolution'],
                region_mask=self.region_mask, qa_plot='show'
            )

            # Copy the relevant data
            self.sens['WAVE_MIN'][i], self.sens['WAVE_MAX'][i] = zp_spec.wave[[0,-1]]
            self.sens['WAVE'][i] = zp_spec.wave
            self.sens['ZEROPOINT'][i] = zp_spec.flux
            self.sens['ZEROPOINT_GPM'][i] = fit_gpm_rej
            self.sens['ZEROPOINT_FIT'][i], self.sens['ZEROPOINT_FIT_GPM'] = zp_bspl.value(
                zp_spec.wave
            )
            self.sens['THROUGHPUT'][i] = flux_calib_refactor.zeropoint_to_throughput(
                self.sens['WAVE'][i], self.sens['ZEROPOINT_FIT'][i],
                self.spectrograph.telescope.eff_aperture()
            )

    def eval_zeropoint(self, wave, indx=0):
        """
        Evaluate the sensitivity function zero-points at the input wavelength.

        This is a simple linear interpolation/extrapolation of the bspline model
        fit to the zeropoint data.

        Parameters
        ----------
        wave : :class:`numpy.ndarray`
            Wavelengths at which to evaluate the zeropoint
        indx : :obj:`int`, optional
            For calculations based on multiple spectra (cross-dispersed echelles
            or spectra across multiple, independently processed detectors), this
            selects the zeropoint calculation to use.

        Returns
        -------
        :class:`numpy.ndarray`
            Interpolated zeropoint data.  Shape is identical to ``wave``.
        """
        return interpolate.interp1d(
            self.sens['WAVE'][indx], self.sens['ZEROPOINT_FIT'][indx], bounds_error=False,
            fill_value='extrapolate'
        )(wave)



