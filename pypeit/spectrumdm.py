"""
Implements a spectrum data container
"""

from astropy.io import fits
from IPython import embed
import numpy as np

from pypeit import log
from pypeit.core import spectrum
from pypeit.datamodel import DataContainer
from pypeit.datamodel import DataContainerList
from pypeit.datamodel import define_datamodel_component


class SpectrumContainer(DataContainer, spectrum.Spectrum):
    """
    A :class:`~pypeit.datamodel.DataContainer` object that holds data for a
    single spectrum.

    This is largely a convenience class for handling IO.

    Parameters
    ----------
    *args
        Passed directly to the instantiation of the
        :class:`~pypeit.core.spectrum.Spectrum` base-class attributes.
    name : str, optional
        A string identifier for the spectrum.
    **kwargs
        Passed directly to the instantiation of the
        :class:`~pypeit.core.spectrum.Spectrum` base-class attributes.
    """

    version = '1.0.0'
    """
    Version of the datamodel
    """

    datamodel = {
        'wave' : define_datamodel_component(
            otype=np.ndarray, atype=np.floating, descr='Wavelength array'
        ),
        'flux' : define_datamodel_component(
            otype=np.ndarray, atype=np.floating, descr='Flux array'
        ),
        'ivar' : define_datamodel_component(
            otype=np.ndarray, atype=np.floating, descr='Inverse variance in flux'
        ),
        'gpm' : define_datamodel_component(
            otype=np.ndarray, atype=(bool, np.bool), descr='Good-pixel mask'
        ),
    }
    """
    Components of the data model and their type.  This should essentially match
    the array attributes of the :class:`~pypeit.core.spectrum.Spectrum` base
    class.
    """

    internals = [
        'name',
        'meta'
    ]
    """
    Internals that are not part of the data model.  Note the metadata dictionary held by 
    :class:`~pypeit.core.spectrum.Spectrum` is included here.
    """

    allowed_metadata_types = (int, np.integer, float, np.floating, bool, np.bool, str)
    """
    The allowed types for metadata that will be written to an output file.  The
    :attr:`meta` dictionary can hold other data, but it will be lost when the
    object is output to a file.
    """

    def __init__(self, *args, name=None, **kwargs):
        DataContainer.__init__(self)
        self.name = name
        spectrum.Spectrum.__init__(self, *args, **kwargs)

    def copy(self):
        """
        Override the :func:`~pypeit.core.spectrum.Spectrum.copy` method.
        """
        # NOTE: This should call the Spectrum.copy method (i.e., there is no
        # DataContainer.copy method).
        cp = super().copy()
        cp.name = self.name
        return cp

    def _bundle(self, ext=None, transpose_arrays=False):
        """
        Override the :func:`~pypeit.datamodel.DataContainer._bundle` method.
        Function arguments are passed directly to the base-class method.
        """
        # Set the name of the HDU to self.name if `ext` is not provided.
        if ext is None:
            ext = self.name
        # If it is still none, set it to 'SPECTRUM'
        if ext is None:
            ext = 'SPECTRUM'

        # Use the base class function, which will just collect the datamodel
        # arrays into a dictionary.  The returned object is a list with one
        # entry, a dictionary with one key (ext), that itself contains a
        # dictionary with the spectrum vectors.
        data = super()._bundle(ext=ext, transpose_arrays=transpose_arrays)

        # The extension name and the spectrum name should be the same, but save
        # the original
        if self.name is not None:
            data[0][ext]['NAME'] = self.name

        # Add the metadata
        if self.meta is not None:
            meta_keys = []
            for key, value in self.meta.items():
                if not isinstance(self.meta[key], self.allowed_metadata_types):
                    log.warning(
                        f'{key} metadata is not a type ({type(self.meta[key])}) that can be '
                        'parsed into the header of a FITS extension.'
                    )
                    continue
                data[0][ext][key] = value
                meta_keys += [key]
            # Finally add the list of metadata keys that were included
            data[0][ext]['METAKEYS'] = ', '.join(meta_keys)

        return data

    @classmethod
    def from_hdu(cls, hdu, chk_version=True, **kwargs):
        """
        Override :func:`~pypeit.datamodel.DataContainer.from_hdu` method to
        enable parsing header data into the :attr:`meta` dictionary.
        """
        # This reproduces *all* of the lines in the base class function.  We
        # need to know which hdus were parsed to setup the metadata dictionary
        # and name.

        # The following three lines are identical to DataContainer.from_hdu
        d, dm_version_passed, dm_type_passed, parsed_hdus = cls._parse(hdu, **kwargs)
        cls._check_parsed(dm_version_passed, dm_type_passed, chk_version=chk_version)
        self = cls.from_dict(d=d)

        # Only one hdu should be parsed
        if len(parsed_hdus) != 1:
            log.warning(
                f'The {cls.__name__} data should be contained in a single HDU, but multiple HDUs '
                'were parsed.  Just taking data from the first one.'
            )
        parsed_hdus = parsed_hdus[0]

        # Get the header
        hdr = hdu[parsed_hdus].header if isinstance(hdu, fits.HDUList) else hdu.header

        if 'NAME' in hdr:
            # If the name was in the header, use it
            self.name = hdr['NAME']
        elif isinstance(parsed_hdus, str):
            # Or, if the extension identifier is a string (instead of an
            # integer), use that
            self.name = parsed_hdus

        # Parse the metadata, if available
        if 'METAKEYS' in hdr:
            keys = list(map(lambda x : x.strip(), hdr['METAKEYS'].split(',')))
            self.meta = {key : hdr[key.upper()] for key in keys}

        return self

    @classmethod
    def from_spectrum(cls, spec, name=None):
        """
        Instantiate from a :class:`~pypeit.core.spectrum.Spectrum` object.
        """
        return cls(spec.wave, spec.flux, ivar=spec.ivar, gpm=spec.gpm, meta=spec.meta, name=name)

    def to_spectrum(self):
        """
        Construct a :class:`~pypeit.core.spectrum.Spectrum` object from the
        internal data.

        Returns
        -------
        :class:`~pypeit.core.spectrum.Spectrum`
            An object with a copy of the relevant attributes from this instance
            of :class:`~pypeit.core.spectrumdm.SpectrumContainer`.
        """
        return spectrum.Spectrum(
            self.wave, self.flux, ivar=self.ivar, gpm=self.gpm, meta=self.meta
        )


class SpectrumContainerList(DataContainerList):
    """
    A subclass of :class:`~pypeit.datamodel.DataContainerList` for lists of
    :class:`~pypeit.core.spectrumdm.SpectrumContainer` objects.
    """

    list_type = SpectrumContainer
    """
    The type for elements in instances of this list.
    """
