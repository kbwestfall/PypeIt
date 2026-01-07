
import numpy as np
from pypeit.core import pixels

def test_convert_to_pixel_range():
    x = np.arange(5, dtype=float) + 1.
    # Test with both min and max
    s, e = pixels.convert_to_pixel_range(x, x_min=2.5, x_max=4.5)
    assert s == 2
    assert e == 4
    # Test with only min
    s, e = pixels.convert_to_pixel_range(x, x_min=3.5)
    assert s == 3
    assert e == 5
    # Test with only max
    s, e = pixels.convert_to_pixel_range(x, x_max=2.5)
    assert s == 0
    assert e == 2
    # Test with neither min nor max
    s, e = pixels.convert_to_pixel_range(x)
    assert s == 0
    assert e == 5
