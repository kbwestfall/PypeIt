"""
Script to determine the sensitivity function for a PypeIt 1D spectrum.

.. include common links, assuming primary doc root is up one directory
.. include:: ../include/links.rst
"""
from IPython import embed

from pypeit.scripts import scriptbase

class SensFunc(scriptbase.ScriptBase):

    # TODO: Need an option here for multi_spec_det detectors, passed as a list
    # of numbers in the SensFunc parset, or as --det 3 7 on the command line
    @classmethod
    def get_parser(cls, width=None):
        parser = super().get_parser(
            description='Compute a sensitivity function', width=width,
            formatter=scriptbase.SmartFormatter, default_log_file=True
        )
        parser.add_argument(
            'spec1dfiles', type=str, nargs='+',
            help='file(s) of the reduced standard star spectrum.  These can be either '
                 'spec1d*.fits files or the output of `pypeit_coadd_1dspec` (except for '
                 'cross-dispersed echelle data).  Multiple files can be provided, but they are '
                 'helpful only if they cover different wavelength ranges, since this script will '
                 'splice (not combine) them together.'
        )
        parser.add_argument(
            '-s', '--sens_file', type=str,
            help=(
                'Configuration file used to set the parameters used to calculate the sensitivity '
                'function.  This can be a ".pypeit" file that includes the desired "sensfunc" '
                'parameter group or a ".sens" file.  This argument is *required*.'
            )
        )
        parser.add_argument(
            '--par_outfile', default=None,
            help='File name for parameters used by the fit.  No file is written if no name is '
                 'given.'
        )
        parser.add_argument(
            "-o", "--outfile", type=str,
            help='Output file for sensitivity function. If not specified, the sensitivity '
                 'function will be written out to a standard filename in the current working '
                 'directory, i.e. if the standard spec1d file is named ' \
                 'spec1d_b24-Feige66_KASTb_foo.fits the sensfunc will be written to '
                 'sens_b24-Feige66_KASTb_foo.fits. A QA file will also be written as '
                 'sens_spec1d_b24-Feige66_KASTb_foo_QA.pdf and a file showing throughput plots to '
                 'sens_spec1d_b24-Feige66_KASTb_foo_throughput.pdf. The same extensions for QA '
                 'and throughput will be used if outfile is provided but with .fits trimmed off '
                 'if it is in the filename.'
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
        """Executes sensitivity function computation."""

        from pathlib import Path

        from astropy.io import fits
        import numpy as np

        from pypeit import log
        from pypeit import PypeItError
        from pypeit import inputfiles
#        from pypeit import io
#        from pypeit.par import pypeitpar
        from pypeit import sensfunc
#        from pypeit.spectrographs.util import load_spectrograph

        # Initialize the log
        cls.init_log(args)

        if args.sens_file is None:
            raise PypeItError('Must provide either a .pypeit or .sens configuration file')

        # Check the input files exist
        _spec1dfiles = [Path(sf).absolute() for sf in np.atleast_1d(args.spec1dfiles)]
        badfiles = [sf for sf in _spec1dfiles if not sf.is_file()]
        if len(badfiles) > 0:
            raise FileNotFoundError(f'The following spec1d files were not found: {badfiles}')

        # NOTE: This block of code that instantiates the parameter set is very
        # similar to what is used in pypeit/scripts/tellfit.py.  We might want
        # a function that can be used by both scripts.
        # Load the parameters.  First try to read the input file as a .pypeit
        # file.
        # TODO: Do we need to save both the 'senfunc' and 'fluxcalib' groups?
        try:
            ifile = inputfiles.PypeItFile.from_file(args.sens_file)
        except PypeItError as e:
            log.warning(
                f'Could not read {args.sens_file} as a .pypeit file.  Attempting to read as a '
                f'.sens file.  Error was: {e}'
            )
            par = None
        else:
            spectrograph, par, config_specific_file = ifile.get_pypeitpar()

        if par is None:
            # That failed, so now attempt a .sens file.
            try:
                ifile = inputfiles.SensFile.from_file(args.sens_file)
            except PypeItError as e:
                raise PypeItError(
                    f'Unable to parse {args.sens_file} as a .pypeit file or a .sens file!'
                )
            else:
                # Get the set of parameters
                # NOTE: We set `pypeit_fits=True` below because, *by definition*,
                # the input files to sensfunc are PypeIt output files.
                with fits.open(_spec1dfiles[0]) as hdu:
                    spectrograph, par, config_specific_file = ifile.get_pypeitpar(
                        config_specific_file=hdu, spectrograph_name=hdu[0].header['PYP_SPEC'],
                        pypeit_fits=True
                    )

        # To get here, par and spectrograph cannot be None

        # TODO: Return to this.  How much of this is already in the header of
        # the spec1d files and the coadd output files?  Can we make sure the
        # headers of *those* files are complete?
#        # Determine the spectrograph and generate the primary FITS header
#        with io.fits_open(args.spec1dfiles[0]) as hdul:
#            spectrograph = load_spectrograph(hdul[0].header['PYP_SPEC'], pypeit_fits=True)
#            spectrograph_config_par = spectrograph.config_specific_par(hdul)
#
#            # Construct a primary FITS header that includes the spectrograph's
#            #   config keys for inclusion in the output sensfunc file
#            primary_hdr = io.initialize_header()
#            add_keys = (
#                ['PYP_SPEC', 'DATE-OBS', 'TELESCOP', 'INSTRUME', 'DETECTOR']
#                + spectrograph.configuration_keys() + spectrograph.raw_header_cards()
#            )
#            for key in add_keys:
#                if key.upper() in hdul[0].header.keys():
#                    primary_hdr[key.upper()] = hdul[0].header[key.upper()]
        primary_hdr = None

#        # If the .sens file was passed in read it and overwrite default parameters
#        if args.sens_file is not None:
#            sensFile = inputfiles.SensFile.from_file(args.sens_file)
#            # Read sens file
#            par = pypeitpar.PypeItPar.from_cfg_lines(
#                        cfg_lines=spectrograph_config_par.to_config(),
#                        merge_with=(sensFile.cfg_lines,))
#        else:
#            par = pypeitpar.PypeItPar.from_cfg_lines(cfg_lines=spectrograph_config_par.to_config())

        # Write the par to disk
        if args.par_outfile is not None:
            log.info(f'Writing the sensfunc parameters to {args.par_outfile}')
            par.to_config(cfg_file=args.par_outfile, include_descr=False)

        # Parse the output filename
        if args.outfile is not None:
            outfile = args.outfile
        else:
            # read the filenames and parse
            _names = [f.name for f in _spec1dfiles]
            # if spec1d_ in the filename, remove it
            _names = [n.split('spec1d_')[-1] if n.startswith('spec1d') else n for n in _names]
            spec1dname = _names[0] if len(_names) == 1 else f"{_names[0].split('.fits')[0]}-{_names[-1]}"
            outfile = 'sens_' + spec1dname

        # Read the spectra
        embed()
        exit()

        # Instantiate the relevant class for the requested algorithm
        sensobj = sensfunc.SensFunc.get_instance(args.spec1dfiles, outfile, par['sensfunc'],
                                                 par_fluxcalib=par['fluxcalib'], debug=args.debug,
                                                 chk_version=par['rdx']['chk_version'])
        # Generate the sensfunc
        sensobj.run()
        # Write it out to a file, including the new primary FITS header
        sensobj.to_file(outfile, primary_hdr=primary_hdr, overwrite=True)

        #TODO JFH Add a show_sensfunc option here and to the sensfunc classes.      

