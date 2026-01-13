import numpy as np
import pytest

from pypeit.core import pydl


def make_test_arrays(seed=99):
    """Create test arrays for pydl tests using fixed size and seed.

    Returns
    -------
    data : np.ndarray
        Normal-distributed data with unity standard deviation and length 1000.
    invvar : np.ndarray
        Inverse variance array of ones.
    mask : np.ndarray
        Boolean mask array set to True everywhere.
    """
    n = 1000
    rng = np.random.default_rng(seed)
    data = rng.normal(loc=0.0, scale=1.0, size=n)
    invvar = np.ones_like(data)
    mask = np.ones_like(data, dtype=bool)
    return data, invvar, mask


def add_outliers(data, n_outliers=100, sigma=10.0, seed=99):
    """Return a copy of `data` with `n_outliers` randomly replaced by
    samples drawn from a normal distribution with standard deviation
    `sigma`.

    Parameters
    ----------
    data : np.ndarray
        Input 1-D data array.
    n_outliers : int
        Number of values to replace.
    sigma : float
        Standard deviation of the outlier normal distribution.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    new_data : np.ndarray
        Copy of input data with outliers inserted.
    mask : np.ndarray
        Boolean array with True at positions of the injected outliers.
    """
    rng = np.random.default_rng(seed)
    n = data.size
    if n_outliers >= n:
        raise ValueError('n_outliers must be less than data length')
    inds = rng.choice(n, size=n_outliers, replace=False)
    new_data = data.copy()
    new_data[inds] = rng.normal(loc=0.0, scale=sigma, size=n_outliers)
    mask = np.zeros(n, dtype=bool)
    mask[inds] = True
    return new_data, mask





def test_no_rejection_when_equal():
    data, invvar, mask = make_test_arrays()
    model = data.copy()
    outmask_in = np.ones_like(data, dtype=bool)
    outmask, qdone = pydl.djs_reject(data, model, outmask=outmask_in.copy(), invvar=invvar)
    assert qdone, "Expected qdone True when data equals model"
    assert np.all(outmask), "Expected all points to be kept when data equals model"


def test_detect_injected_outliers():
    rng = np.random.default_rng(99)
    data, invvar, mask = make_test_arrays(seed=rng)
    data2, injected = add_outliers(data, n_outliers=100, sigma=10.0, seed=rng)
    model = np.zeros_like(data2)
    outmask, qdone = pydl.djs_reject(data2, model, outmask=np.ones_like(data2, dtype=bool), invvar=invvar, upper=5)
    # At least one injected outlier should be detected as rejected
    detected = np.logical_and(np.logical_not(outmask), injected)
    assert detected.sum() > 0, "Expected at least one injected outlier to be detected as rejected"


def test_upper_rejection_invvar():
    data, invvar, mask = make_test_arrays()
    # inject a strong outlier at index 0
    data[0] = 10.0
    model = np.zeros_like(data)
    outmask, qdone = pydl.djs_reject(data, model, outmask=np.ones_like(data, dtype=bool), invvar=invvar, upper=3)
    assert not qdone, "Expected qdone False when an outlier is present"
    assert not outmask[0], "Expected the injected outlier at index 0 to be rejected"


def test_use_mad_with_invvar_raises():
    data, invvar, mask = make_test_arrays()
    model = np.zeros_like(data)
    with pytest.raises(ValueError):
        pydl.djs_reject(data, model, invvar=invvar, use_mad=True)


def test_use_mad_rejection():
    # Create data where the median absolute deviation is non-zero
    data, invvar, mask = make_test_arrays()
    # create three large outliers
    data[:3] = 30.0
    model = np.zeros_like(data)
    # use_mad=True computes sigma from data-model and should allow rejection
    outmask, qdone = pydl.djs_reject(data, model, outmask=np.ones_like(data, dtype=bool), use_mad=True, upper=0.6)
    # The three large points should be rejected
    assert not qdone, "Expected qdone False with MAD-based rejection"
    assert not np.any(outmask[:3]), "Expected the first three large points to be rejected"


def test_grow_and_sticky():
    data, invvar, mask = make_test_arrays()
    # inject an outlier in the middle
    mid = data.size // 2
    data[mid] = 10.0
    model = np.zeros_like(data)
    outmask_in = np.ones_like(data, dtype=bool)
    # Reject the outlier and grow by 1 should also reject its neighbors
    outmask, qdone = pydl.djs_reject(data, model, outmask=outmask_in.copy(), invvar=invvar, upper=3, grow=1)
    assert (not outmask[mid-1]) and (not outmask[mid]) and (not outmask[mid+1]), \
        "Expected outlier and its immediate neighbors to be rejected when grow=1"
    # Now test sticky: start with a previously rejected pixel
    prev = outmask.copy()
    prev[0] = False
    outmask2, qdone2 = pydl.djs_reject(data, model, outmask=prev, invvar=invvar, upper=3, sticky=True)
    # The sticky False at index 0 must remain
    assert not outmask2[0], "Expected previously rejected pixel to remain rejected when sticky=True"
