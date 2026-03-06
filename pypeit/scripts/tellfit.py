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
            '-t', '--tell_file', type=str, default=None,
            help=(
                'Configuration file used to set the parameters used to determine the telluric '
                'correction.  This can be a ".pypeit" file that includes the desired "telluric" '
                'parameter group or a ".tell" file.'
            )
        )
        parser.add_argument(
            '--par_outfile', type=str, default=None,
            help=(
                'File name for parameters used by the fit.  No file is written if no name is '
                'given.'
            )
        )
        parser.add_argument(
            '--extract', type=str, default=None, choices=[None, 'BOX', 'OPT'],
            help=(
                'The extraction to use.  Must be either None, BOX (for a boxcar extraction), or '
                'OPT (for optimal extraction).  If None, the optimal extraction will be used, if '
                'it exists, otherwise the boxcar extraction will be used.  This is only relevant '
                'if the input is a spec1d file (as opposed to a onespec/coadd file).'
            )
        )
        parser.add_argument(
            '--fluxed', default=False, action='store_true',
            help=(
                'If True, use the flux-calibrated spectrum, if it exists.  If the flux '
                'calibration has not been performed or fluxed is False, the spectrum used is '
                'in counts.  This is only relevant if the input is a spec1d file (as opposed to '
                'a onespec/coadd file).'
            )
        )
        parser.add_argument(
            '--show', default=False, action='store_true', help='Show the fit results'
        )
        parser.add_argument(
            '--debug', default=False, action='store_true', help='Run in debugging mode'
        )
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
        from pypeit.scripts import loader

        # Initialize the log
        cls.init_log(args)

        # Check the input spec1d file
        _spec1dfile = Path(args.spec1dfile).absolute()
        if not _spec1dfile.is_file():
            raise FileNotFoundError(f'Spec1d file not found: {_spec1dfile}')

        # Get the parameters and the spectrograph
        par, spec = loader.get_pypeitpar(
            _spec1dfile, ifile=args.tell_file, secondary_ifile_class=inputfiles.TelluricFile
        )
        # We only need the set of telluric parameters
        par = par['telluric']

        # Make sure the file with the telluric models is defined
        if par['tel_file'] is None:
            raise PypeItError(
                'No telluric grid file is specified.  This means it has not been specified in '
                'your input file and there is no default for your spectrograph.  You must set '
                'the tel_file parameter; see the pypeit documentation for options.'
            )
        
        # Write the par to disk
        if args.par_outfile is not None:
            log.info(f'Writing the telluric fitting parameters to {args.par_outfile}')
            par.to_config(cfg_file=args.par_outfile, include_descr=False)

        # Load the spectra to fit
        hdr, spectra = loader.load_spectra(
            _spec1dfile, extract=args.extract, fluxed=args.fluxed, chk_version=args.chk_version
        )

        # Instantiate the object used to determine the telluric correction
        tell_corr = telluric.correction.TelluricCorrection(spectra, par)
        # ... and run the fit
        tell_corr.fit(show=args.show, debug=args.debug)

        # Save the results
        modelfile = _spec1dfile.parent / _spec1dfile.name.replace('.fits','_tellmodel.fits')
        log.info(f'Best-fit telluric model will be saved to: {modelfile}.')
        if modelfile.is_file():
            log.warning(f'{modelfile} will be overwritten!')
        tell_corr.to_file(modelfile, overwrite=True)

        # Set the output files, and put them in the same directory that has the
        # spectrum file
#        outfile = _spec1dfile.parent / _spec1dfile.name.replace('.fits','_tellcorr.fits')
#        log.info(f'Telluric-corrected spectrum will be saved to: {outfile}.')

        # TODO: Write the telluric-corrected spectrum to a file.

