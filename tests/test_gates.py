# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The qtflow authors
"""Native gate definitions: unitarity, level-selective action, IBM gates on the qubit subspace."""
import itertools
import numpy as np
import pytest
from qtflow.gates import native_unitary, level_selective, LEVEL_SELECTIVE, D, LOGICAL_1Q, embed_1q

NAMES = ['rz', 'sx', 'x', 'cz', 'c1x12', 'c2x01', 'c2x12']

@pytest.mark.parametrize('name', NAMES)
def test_unitary(name):
    U = native_unitary(name, (0.37,))
    assert np.allclose(U.conj().T @ U, np.eye(len(U)))

@pytest.mark.parametrize('name', ['c1x12', 'c2x01', 'c2x12'])
def test_level_selective_definition(name):
    """C^c X_jk = sum_{i != c} |i><i| (x) I + |c><c| (x) X_jk"""
    c0, j, k = LEVEL_SELECTIVE[name]
    Xjk = np.eye(D); Xjk[[j, k]] = Xjk[[k, j]]
    ref = sum(np.kron(np.outer(np.eye(D)[i], np.eye(D)[i]), Xjk if i == c0 else np.eye(D)) for i in range(D))
    assert np.allclose(level_selective(name), ref)

@pytest.mark.parametrize('name', ['c1x12', 'c2x01', 'c2x12'])
def test_level_selective_self_inverse_permutation(name):
    U = level_selective(name)
    assert np.allclose(U @ U, np.eye(D*D))
    assert set(np.unique(U.real)) <= {0.0, 1.0} and np.allclose(U.imag, 0)

def test_basis_maps():
    def act(name, c, t):
        v = np.zeros(D*D); v[c*D+t] = 1; w = level_selective(name) @ v; i = int(np.argmax(w)); return divmod(i, D)
    assert act('c1x12', 1, 1) == (1, 2) and act('c1x12', 1, 2) == (1, 1) and act('c1x12', 0, 1) == (0, 1)
    assert act('c2x01', 2, 0) == (2, 1) and act('c2x01', 1, 0) == (1, 0) and act('c2x01', 2, 2) == (2, 2)
    assert act('c2x12', 2, 1) == (2, 2) and act('c2x12', 1, 1) == (1, 1) and act('c2x12', 2, 0) == (2, 0)

def test_single_qubit_natives_leave_level2():
    for nm in ('rz', 'sx', 'x'):
        U = native_unitary(nm, (1.1,)); assert np.isclose(U[2, 2], 1) and np.allclose(U[2, :2], 0)

def test_sx_squared_is_x_and_cz_phase():
    assert np.allclose(native_unitary('sx') @ native_unitary('sx'), native_unitary('x'))
    U = native_unitary('cz')
    for a, b in itertools.product(range(D), repeat=2):
        assert np.isclose(U[a*D+b, a*D+b], -1 if (a, b) == (1, 1) else 1)

@pytest.mark.parametrize('name', sorted(LOGICAL_1Q))
def test_logical_1q_unitary(name):
    U = LOGICAL_1Q[name]((0.9,)); assert np.allclose(U.conj().T @ U, np.eye(2))
    assert np.allclose(embed_1q(U)[:2, :2], U)
