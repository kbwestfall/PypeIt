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
        parser.add_argument(
            '--extract', type=str, default=None, choices=[None, 'BOX', 'OPT'],
            help='The extraction to use.  Must be either None, BOX (for a boxcar extraction), '
                 'or OPT (for optimal extraction).  If None, the optimal extraction will be '
                 'used, if it exists, otherwise the boxcar extraction will be used.  This is '
                 'only relevant if the input is a spec1d file (as opposed to a onespec/coadd '
                 'file).'
        )
        parser.add_argument(
            '--fluxed', default=False, action='store_true',
            help='If True, use the flux-calibrated spectrum, if it exists.  If the flux '
                 'calibration has not been performed or fluxed is False, the spectrum used is '
                 'in counts.  This is only relevant if the input is a spec1d file (as opposed '
                 'to a onespec/coadd file).'
        )
        parser.add_argument('--show', default=False, action='store_true',
                            help='Show the telluric corrected spectrum')
        parser.add_argument('--debug', default=False, action='store_true',
                            help='show debug plots?')
        parser.add_argument(
            '--try_old', dest='chk_version', default=True, action='store_false',
            help='Ensure the datamodels are from the current PypeIt version.'
        )
        return parser

    @classmethod
    def main(cls, args):
        """
        Executes telluric correction.
        """

        from pathlib import Path

        from astropy.io import fits
        from IPython import embed

        from pypeit import inputfiles
        from pypeit import log
        from pypeit import PypeItError
        from pypeit import telluric

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

        if par['tel_file'] is None:
            raise PypeItError(
                'No telluric grid file is specified.  This means it has not been specific in '
                'your input file and there is no default for your spectrograph.  You must set '
                'the tel_file parameter; see the pypeit documentation for options.'
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

        tell_corr = telluric.correction.TelluricCorrection(
            _spec1dfile, par, extract=args.extract, fluxed=args.fluxed,
            chk_version=args.chk_version
        )

        embed()
        exit()

        tell_corr.fit(show=args.show, debug=args.debug)


