
from IPython import embed

import numpy as np

from pypeit import sampling

def test_centers_borders():

    # Use arange
    x = np.arange(10, dtype=float)
    borders = sampling.centers_to_borders(x)
    assert np.array_equal(np.arange(-0.5, 10.5, 1.), borders), 'Border calculation is wrong'
    assert np.array_equal(x, sampling.borders_to_centers(borders)), 'Did not recover centers'

    # Use linspace
    x = np.linspace(0., 9., 19)
    borders = sampling.centers_to_borders(x)
    assert np.array_equal(np.arange(-0.25, 9.75, 0.5), borders), 'Border calculation is wrong'
    assert np.array_equal(x, sampling.borders_to_centers(borders)), 'Did not recover centers'

    # Use geomspace
    x = np.geomspace(1., 2**16, 17)
    borders = sampling.centers_to_borders(x, log=True)
    log_step = np.diff(np.log(borders))
    assert np.allclose(log_step, log_step[0]), \
        'Step should be constant in log to numerical precision'
    assert np.allclose(x, sampling.borders_to_centers(borders, log=True)), \
        'Should recover centers to numerical precision'
    
    # Irregularly spaced sampling is generally not recoverable, but the
    # differences should be small
    rng = np.random.default_rng(99)
    borders = np.arange(11, dtype=float) + 0.1*rng.normal(size=11)
    width = np.diff(borders)
    x = sampling.borders_to_centers(borders)
    rec_borders = sampling.centers_to_borders(x)
    assert np.all(np.absolute(borders - rec_borders)[:-1] / width < 0.2), \
        'Poor irregular grid recovery'

