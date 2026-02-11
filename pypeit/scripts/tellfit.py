"""
Fit telluric absorption to observed spectra

.. include common links, assuming primary doc root is up one directory
.. include:: ../include/links.rst
"""
from pypeit.scripts import scriptbase

class TellFit(scriptbase.ScriptBase):

    @classmethod
    def get_parser(cls, width=None):
        parser = super().get_parser(
            description='Telluric correct a spectrum', width=width,
            formatter=scriptbase.SmartFormatter, default_log_file=True
        )
        parser.add_argument(
            'spec1dfile', type=str,
            help="spec1d or coadd file that will be used for telluric correction."
        )
        parser.add_argument(
            'tell_file', type=str,
            help='Configuration file used to set the telluric.  This can be a ".pypeit" file '
                 'that includes the desired telluric parameters or a ".tell" file with a set '
                 'of parameters specific to the provided 1D spectrum.'
        )
        parser.add_argument(
            '--par_outfile', default=None,
            help='File name for parameters used by the fit.  No file is written if no name is '
                 'given.'
        )
        parser.add_argument("--debug", default=False, action="store_true",
                            help="show debug plots?")
        parser.add_argument("--plot", default=False, action="store_true",
                            help="Show the telluric corrected spectrum")
        parser.add_argument('--chk_version', default=False, action='store_true',
                            help='Ensure the datamodels are from the current PypeIt version. '
                                 'By default (consistent with previous functionality) this is '
                                 'not enforced and crashes may ensue ...')
        return parser

    @classmethod
    def main(cls, args):
        """
        Executes telluric correction.
        """

        from pathlib import Path

        from astropy.io import fits
        from IPython import embed

        from pypeit import log
        from pypeit import PypeItError
        from pypeit import dataPaths
        from pypeit.par import pypeitpar
        from pypeit.spectrographs.util import load_spectrograph
        from pypeit.core import telluric
        from pypeit import inputfiles

        # Initialize the log
        cls.init_log(args)

        # Check the input spec1d file
        _spec1dfile = Path(args.spec1dfile).absolute()
        if not _spec1dfile.is_file():
            raise FileNotFoundError(f'Spec1d file not found: {_spec1dfile}')

        # Load the parameters.  First try to read the input file as a .pypeit
        # file.
        try:
            ifile = inputfiles.PypeItFile.from_file(args.tell_file)
        except PypeItError as e:
            log.warning(
                f'Could not read {args.tell_file} as a .pypeit file.  Attempting to read as a '
                f'.tell file.  Error was: {e}'
            )
            par = None
        else:
            par = ifile.get_pypeitpar()[1]['telluric']

        if par is None:
            # That failed, so now attempt a .tell file.
            try:
                ifile = inputfiles.TelluricFile.from_file(args.tell_file)
            except PypeItError as e:
                raise PypeItError(
                    f'Unable to parse {args.tell_file} as a .pypeit file or a .tell file!'
                )
            else:
                with fits.open(_spec1dfile) as hdu:
                    par = ifile.get_pypeitpar(
                        config_specific_file=hdu, spectrograph_name=hdu[0].header['PYP_SPEC'],
                        pypeit_fits=True
                    )[1]
                par = par['telluric']

        # NOTE: Code should not be able to get here with par = None.

        if par['telgridfile'] is None:
            raise PypeItError(
                'No telluric grid file is specified.  This means it has not been specific in '
                'your input file and there is no default for your spectrograph.  You must set '
                'the telgridfile parameter; see the pypeit documentation for options.'
            )

        # Write the par to disk
        if args.par_outfile is not None:
            log.info(f'Writing the telluric fitting parameters to {args.par_outfile}')
            par.to_config(args.par_outfile, section_name='telluric', include_descr=False)

        # Set the output files
        # TODO: Make one output file
        outfile = _spec1dfile.name.replace('.fits','_tellcorr.fits')
        log.info(f'Telluric-corrected spectrum will be saved to: {outfile}.')

        modelfile = _spec1dfile.name.replace('.fits','_tellmodel.fits')
        log.info(f'Best-fit telluric model will be saved to: {modelfile}.')

        embed()
        exit()

        # Run the telluric fitting procedure.
        if par['telluric']['objmodel']=='qso':
            # run telluric.qso_telluric to get the final results
            TelQSO = telluric.qso_telluric(
                args.spec1dfile,
                par['telluric']['telgridfile'],
                par['telluric']['pca_file'],
                par['telluric']['redshift'],
                modelfile,
                outfile,
                npca=par['telluric']['npca'],
                teltype=par['telluric']['teltype'],
                tell_npca=par['telluric']['tell_npca'],
                pca_lower=par['telluric']['pca_lower'],
                pca_upper=par['telluric']['pca_upper'],
                bounds_norm=par['telluric']['bounds_norm'],
                tell_norm_thresh=par['telluric']['tell_norm_thresh'],
                only_orders=par['telluric']['only_orders'],
                bal_wv_min_max=par['telluric']['bal_wv_min_max'],
                resln_frac_bounds=par['telluric']['resln_frac_bounds'],
                pix_shift_bounds=par['telluric']['pix_shift_bounds'],
                maxiter=par['telluric']['maxiter'],
                popsize=par['telluric']['popsize'],
                tol=par['telluric']['tol'],
                debug_init=args.debug,
                disp=args.debug,
                debug=args.debug,
                show=args.plot,
                chk_version=args.chk_version,
            )

        elif par['telluric']['objmodel']=='star':
            TelStar = telluric.star_telluric(
                args.spec1dfile,
                par['telluric']['telgridfile'],
                modelfile,
                outfile,
                star_type=par['telluric']['star_type'],
                star_mag=par['telluric']['star_mag'],
                star_ra=par['telluric']['star_ra'],
                star_dec=par['telluric']['star_dec'],
                func=par['telluric']['func'],
                model=par['telluric']['model'],
                polyorder=par['telluric']['polyorder'],
                only_orders=par['telluric']['only_orders'],
                teltype=par['telluric']['teltype'],
                tell_npca=par['telluric']['tell_npca'],
                mask_hydrogen_lines=par['sensfunc']['mask_hydrogen_lines'],
                mask_helium_lines=par['sensfunc']['mask_helium_lines'],
                hydrogen_mask_wid=par['sensfunc']['hydrogen_mask_wid'],
                delta_coeff_bounds=par['telluric']['delta_coeff_bounds'],
                minmax_coeff_bounds=par['telluric']['minmax_coeff_bounds'],
                resln_frac_bounds=par['telluric']['resln_frac_bounds'],
                pix_shift_bounds=par['telluric']['pix_shift_bounds'],
                maxiter=par['telluric']['maxiter'],
                popsize=par['telluric']['popsize'],
                tol=par['telluric']['tol'],
                debug_init=args.debug,
                disp=args.debug,
                debug=args.debug,
                show=args.plot,
                chk_version=args.chk_version,
            )
        elif par['telluric']['objmodel']=='poly':
            TelPoly = telluric.poly_telluric(
                args.spec1dfile,
                par['telluric']['telgridfile'],
                modelfile,
                outfile,
                z_obj=par['telluric']['redshift'],
                func=par['telluric']['func'],
                model=par['telluric']['model'],
                polyorder=par['telluric']['polyorder'],
                teltype=par['telluric']['teltype'],
                tell_npca=par['telluric']['tell_npca'],
                fit_wv_min_max=par['telluric']['fit_wv_min_max'],
                mask_lyman_a=par['telluric']['mask_lyman_a'],
                delta_coeff_bounds=par['telluric']['delta_coeff_bounds'],
                minmax_coeff_bounds=par['telluric']['minmax_coeff_bounds'],
                only_orders=par['telluric']['only_orders'],
                resln_frac_bounds=par['telluric']['resln_frac_bounds'],
                pix_shift_bounds=par['telluric']['pix_shift_bounds'],
                maxiter=par['telluric']['maxiter'],
                popsize=par['telluric']['popsize'],
                tol=par['telluric']['tol'],
                debug_init=args.debug,
                disp=args.debug,
                debug=args.debug,
                show=args.plot,
                chk_version=args.chk_version,
            )
        else:
            raise PypeItError("Object model is not supported yet. Must be 'qso', 'star', or 'poly'.")


