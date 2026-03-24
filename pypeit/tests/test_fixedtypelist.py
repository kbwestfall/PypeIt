
import numpy as np
from IPython import embed
import pytest

from pypeit.core import fixedtypelist

class StrList(fixedtypelist.FixedTypeList):
    list_type = str

def test_init():
    l = StrList()

    assert len(l) == 0, 'Should have zero length'

    l = StrList(['this', 'should', 'work'])
    assert len(l) == 3, 'Should have three strings'

    with pytest.raises(TypeError):
        l = StrList(['this', 'should', 'not', 'work', 1.])


def test_append_extend():

    l = StrList(['this', 'should', 'work'])
    address = hex(id(l))

    l.append('and')
    assert len(l) == 4, 'Should now have 4 strings'
    assert hex(id(l)) == address, 'Address should not change'

    l.extend(['this'])
    assert len(l) == 5, 'Should now have 5 strings'
    assert hex(id(l)) == address, 'Address should not change'

    l.extend(['should', 'also'])
    assert len(l) == 7, 'Should now have 7 strings'
    assert hex(id(l)) == address, 'Address should not change'


def test_dunder():

    l = StrList(['this', 'should', 'work'])
    address = hex(id(l))

    l += ['and']
    assert len(l) == 4, 'Should now have 4 strings'
    assert hex(id(l)) == address, 'Address should not change'

    l += ['this', 'should', 'also']
    assert len(l) == 7, 'Should now have 7 strings'
    assert hex(id(l)) == address, 'Address should not change'

    l = l + ['this', 'changes', 'the', 'address']
    assert len(l) == 11, 'Should now have 11 strings'
    assert hex(id(l)) != address, 'Address should change'

    with pytest.raises(TypeError):
        # Should not be able to append an incorrect type
        l += [1]

    with pytest.raises(TypeError):
        # Should not be able to append an incorrect type
        l = l + [1]


def test_slicing_indexing():
    l = StrList(['this', 'should', 'work'])

    # Select a single element
    sub = l[0]
    assert isinstance(sub, str), 'Should return a string'

    # Select using a slice
    sub = l[1:3]
    assert isinstance(sub, StrList), 'Should return a StrList'
    assert len(sub) == 2, 'Should have 2 elements'

    # Select a single element using a single-element list
    sub = l[[0]]
    assert isinstance(sub, StrList), 'Should return a StrList'
    assert len(sub) == 1, 'Should only have a single element'

    # Select multiple elements by their index number using a list
    indx = [0,2]
    sub = l[indx]
    assert isinstance(sub, StrList), 'Should return a StrList'
    assert len(sub) == 2, 'Should have 2 elements'
    assert sub[0] == l[indx[0]], 'Mismatch 0'
    assert sub[1] == l[indx[1]], 'Mismatch 1'

    # Select multiple elements by their index number using a numpy array
    indx = np.array([2,1])
    sub = l[indx]
    assert isinstance(sub, StrList), 'Should return a StrList'
    assert len(sub) == 2, 'Should have 2 elements'
    assert sub[0] == l[indx[0]], 'Mismatch 0'
    assert sub[1] == l[indx[1]], 'Mismatch 1'

    # Select multiple elements by their index number using a boolean array
    indx = np.array([True, True, False])
    sub = l[indx]
    _indx = np.where(indx)[0]
    assert isinstance(sub, StrList), 'Should return a StrList'
    assert len(sub) == 2, 'Should have 2 elements'
    assert sub[0] == l[_indx[0]], 'Mismatch 0'
    assert sub[1] == l[_indx[1]], 'Mismatch 1'
