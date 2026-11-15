# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The qtflow authors
"""Simulator: ideal results, noise channels, conventions."""
import numpy as np
import pytest
from qtflow import QCircuit, HeavyHexBackend, NoiseModel, transpile, run, ideal_probabilities
from qtflow.simulator import _kraus_relax, _weyl

B = HeavyHexBackend(3, noise='reference')

def test_bit_order_qubit0_rightmost():
    qc = QCircuit(3); qc.x(0); tc = transpile(qc, B)
    assert ideal_probabilities(tc) == pytest.approx({'001': 1.0})

def test_level2_reported():
    qc = QCircuit(2); qc.x(0); qc.x(1); qc.c1x12(0, 1); tc = transpile(qc, B)
    assert ideal_probabilities(tc) == pytest.approx({'21': 1.0})

def test_ghz_ideal():
    qc = QCircuit(3); qc.h(0); qc.cx(0, 1); qc.cx(1, 2); tc = transpile(qc, B)
    assert ideal_probabilities(tc) == pytest.approx({'000': 0.5, '111': 0.5})

def test_zero_noise_trajectories_equal_ideal():
    nm = NoiseModel(1e12, 1e12, 1e12, gate_error={})
    qc = QCircuit(3); qc.h(0); qc.ccx(0, 2, 1); qc.ry(0.7, 2); tc = transpile(qc, B)
    r = run(tc, shots=100, noise=nm, trajectories=20)
    ideal = ideal_probabilities(tc)
    assert set(r['probabilities']) == set(ideal)
    for k in ideal: assert r['probabilities'][k] == pytest.approx(ideal[k], abs=1e-6)

@pytest.mark.parametrize('t', [0.0, 10.0, 500.0, 5e4])
def test_kraus_trace_preserving(t):
    K = _kraus_relax(t, NoiseModel())
    assert np.allclose(sum(k.conj().T @ k for k in K), np.eye(3), atol=1e-10)

def test_kraus_populations_decay():
    nm = NoiseModel(); K = _kraus_relax(17000.0, nm)
    rho = np.diag([0, 0, 1.0]).astype(complex); out = sum(k @ rho @ k.conj().T for k in K)
    assert np.isclose(np.trace(out).real, 1) and out[2, 2].real == pytest.approx(np.exp(-1), rel=1e-6)

def test_weyl_operators():
    W = _weyl(); assert len(W) == 9
    for i, A in enumerate(W):
        for j, Bm in enumerate(W):
            assert np.isclose(abs(np.trace(A.conj().T @ Bm)), 3 if i == j else 0)

def test_counts_sum_and_normalized():
    qc = QCircuit(3); qc.h(0); qc.ccx(0, 1, 2); tc = transpile(qc, B)
    r = run(tc, shots=1234, trajectories=40)
    assert sum(r['counts'].values()) == 1234 and sum(r['probabilities'].values()) == pytest.approx(1.0)

def test_longer_T1_gives_higher_success():
    qc = QCircuit(5); qc.x(0); qc.x(1); qc.x(2); qc.x(3); qc.mcx([0, 1, 2, 3], 4)
    tc = transpile(qc, B); ps = []
    for T in (3.0, 27.0, 300.0):                      # relaxation only, so the trend is not hidden by gate errors
        nm = NoiseModel(T, T*17/27, 1e12, gate_error={})
        ps.append(run(tc, shots=10, noise=nm, trajectories=400, seed=1)['probabilities'].get('11111', 0))
    assert ps[0] + 0.05 < ps[1] < ps[2] + 0.02 and ps[2] > 0.97

def test_seed_reproducible():
    qc = QCircuit(3); qc.h(0); qc.ccx(0, 1, 2); tc = transpile(qc, B)
    assert run(tc, shots=200, trajectories=30, seed=5) == run(tc, shots=200, trajectories=30, seed=5)

def test_readout_error_spreads_counts():
    nm = NoiseModel(1e12, 1e12, 1e12, gate_error={}, readout_error=0.2)
    qc = QCircuit(2); tc = transpile(qc, B)
    r = run(tc, shots=2000, noise=nm, trajectories=5)
    assert r['counts'].get('00', 0) < 2000 and r['counts'].get('00', 0) > 1100
