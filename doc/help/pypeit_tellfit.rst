.. code-block:: console

    $ pypeit_tellfit -h
    usage: pypeit_tellfit [-h] [-v VERBOSITY] [--log_file LOG_FILE]
                          [--log_level LOG_LEVEL] [--par_outfile PAR_OUTFILE]
                          [--extract {None,BOX,OPT}] [--fluxed] [--show] [--debug]
                          [--try_old]
                          spec1dfile tell_file
    
    Telluric correct a spectrum
    
    positional arguments:
      spec1dfile            spec1d or coadd file that will be used for telluric
                            correction.
      tell_file             Configuration file used to set the telluric. This can be
                            a ".pypeit" file that includes the desired telluric
                            parameters or a ".tell" file with a set of parameters
                            specific to the provided 1D spectrum.
    
    options:
      -h, --help            show this help message and exit
      -v, --verbosity VERBOSITY
                            Verbosity level, which must be 0, 1, or 2. Level 0
                            includes warning and error messages, level 1 adds
                            informational messages, and level 2 adds debugging
                            messages and the calling sequence.
      --log_file LOG_FILE   Name for the log file. If set to "default", a default
                            name is used. If None, a log file is not produced.
      --log_level LOG_LEVEL
                            Verbosity level for the log file. If a log file is
                            produce and this is None, the file log will match the
                            console stream log.
      --par_outfile PAR_OUTFILE
                            File name for parameters used by the fit. No file is
                            written if no name is given.
      --extract {None,BOX,OPT}
                            The extraction to use. Must be either None, BOX (for a
                            boxcar extraction), or OPT (for optimal extraction). If
                            None, the optimal extraction will be used, if it exists,
                            otherwise the boxcar extraction will be used. This is
                            only relevant if the input is a spec1d file (as opposed
                            to a onespec/coadd file).
      --fluxed              If True, use the flux-calibrated spectrum, if it exists.
                            If the flux calibration has not been performed or fluxed
                            is False, the spectrum used is in counts. This is only
                            relevant if the input is a spec1d file (as opposed to a
                            onespec/coadd file).
      --show                Show the telluric corrected spectrum
      --debug               show debug plots?
      --try_old             Ensure the datamodels are from the current PypeIt
                            version.
    