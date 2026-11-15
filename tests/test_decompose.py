# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The qtflow authors
"""Decompositions into the native set are exact (vs Qiskit Operator) and use the claimed pulse counts."""
import numpy as np
import pytest
from qiskit.quantum_info import Operator
from qiskit import QuantumCircuit
from qtflow import QCircuit, HeavyHexBackend, transpile
from qtflow.decompose import one_qubit_native, merge_rz, expand
from qtflow.circuit import Instruction
from qtflow.gates import LOGICAL_1Q, native_unitary
from qtflow.verify import check_equivalence

B = HeavyHexBackend(3, noise='ideal')
ANGLES = [0.0, 0.3, np.pi/2, 2.1, np.pi, -1.7]

def _equal_up_to_phase(A, B_):
    k = np.unravel_index(np.argmax(np.abs(B_)), B_.shape); ph = A[k]/B_[k]
    return np.isclose(abs(ph), 1) and np.allclose(A, ph*B_, atol=1e-10)

@pytest.mark.parametrize('name', sorted(LOGICAL_1Q))
@pytest.mark.parametrize('theta', [0.3, -2.4])
def test_one_qubit_euler(name, theta):
    U = np.eye(2, dtype=complex)
    for ins in one_qubit_native(name, 0, (theta,)):
        assert ins.name in ('rz', 'sx', 'x')
        U = native_unitary(ins.name, ins.params)[:2, :2] @ U
    assert _equal_up_to_phase(U, LOGICAL_1Q[name]((theta,)))

def test_cx_cz_swap():
    for build in (lambda q: q.cx(0, 1), lambda q: q.cx(1, 0), lambda q: q.cz(0, 1), lambda q: q.swap(0, 1)):
        qc = QCircuit(2); build(qc); r = check_equivalence(qc, transpile(qc, B))
        assert r['max_error'] < 1e-9 and r['max_leakage'] < 1e-9

def test_toffoli_three_pulses():
    qc = QCircuit(3); qc.ccx(0, 1, 2); tc = transpile(qc, B)
    assert tc.count_ops() == {'c1x12': 2, 'c2x01': 1}
    assert check_equivalence(qc, tc)['max_error'] < 1e-9

@pytest.mark.parametrize('k', [2, 3, 4, 5, 6])
def test_mcx_exact_and_2k_minus_1(k):
    qc = QCircuit(k+1); qc.mcx(list(range(k)), k); tc = transpile(qc, B)
    assert tc.two_body_pulses() == 2*k - 1 and tc.n_swaps == 0
    if k <= 5:
        r = check_equivalence(qc, tc); assert r['max_error'] < 1e-9 and r['max_leakage'] < 1e-9

@pytest.mark.parametrize('k', [2, 3, 4, 5])
@pytest.mark.parametrize('theta', ANGLES)
def test_mcry_exact_and_2k(k, theta):
    qc = QCircuit(k+1); qc.mcry(theta, list(range(k)), k); tc = transpile(qc, B)
    assert tc.two_body_pulses() == 2*k
    assert check_equivalence(qc, tc)['max_error'] < 1e-9

@pytest.mark.parametrize('k', [2, 3, 4])
@pytest.mark.parametrize('theta', ANGLES)
def test_mcrx_exact_and_2k(k, theta):
    qc = QCircuit(k+1); qc.mcrx(theta, list(range(k)), k); tc = transpile(qc, B)
    assert tc.two_body_pulses() == 2*k
    assert check_equivalence(qc, tc)['max_error'] < 1e-9

def test_single_control_mcry_uses_cx():
    qc = QCircuit(2); qc.mcry(0.8, [0], 1); tc = transpile(qc, B)
    assert tc.count_ops().get('cz') == 2 and check_equivalence(qc, tc)['max_error'] < 1e-9

def test_mcx_with_permuted_qubits():
    qc = QCircuit(5); qc.mcx([4, 0, 2], 1); tc = transpile(qc, B)
    assert check_equivalence(qc, tc)['max_error'] < 1e-9

def test_merge_rz():
    seq = [Instruction('rz', (0,), (0.2,)), Instruction('rz', (0,), (0.3,)), Instruction('sx', (0,)), Instruction('rz', (0,), (0.1,))]
    out = merge_rz(seq)
    assert [i.name for i in out] == ['rz', 'sx', 'rz'] and np.isclose(out[0].params[0], 0.5)

def test_expand_rejects_unknown():
    with pytest.raises(KeyError): expand(Instruction('foo', (0, 1)))

def test_from_qiskit():
    q = QuantumCircuit(4); q.h(0); q.cx(0, 1); q.ccx(0, 1, 2); q.mcx([0, 1, 2], 3); q.rz(0.4, 3); q.measure_all()
    qc = QCircuit.from_qiskit(q); tc = transpile(qc, B)
    assert check_equivalence(qc, tc)['max_error'] < 1e-9
    qref = q.remove_final_measurements(inplace=False)
    assert _equal_up_to_phase(Operator(qref).data, __import__('qtflow.verify', fromlist=['x']).logical_unitary(qc))
