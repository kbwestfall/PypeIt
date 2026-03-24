"""
Provides pypeit specific exceptions.
"""

__all__ = [
    'PypeItError',
    'PypeItBitMaskError',
    'PypeItCodingError',
    'PypeItDataModelError',
    'PypeItPathError'
]

class PypeItError(Exception):
    pass

class PypeItBitMaskError(PypeItError):
    pass

class PypeItCodingError(PypeItError):
    pass

class PypeItDataModelError(PypeItError):
    pass

class PypeItPathError(PypeItError):
    pass
