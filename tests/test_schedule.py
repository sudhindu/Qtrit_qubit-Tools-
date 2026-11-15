# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The qtflow authors
"""Scheduling: durations, flux role moves and the analytic C^k-NOT duration 360k-520 ns."""
import pytest
from qtflow import QCircuit, HeavyHexBackend, transpile

B = HeavyHexBackend(3, noise='ideal')

def test_toffoli_200ns_two_flux_moves():
    qc = QCircuit(3); qc.ccx(0, 1, 2); tc = transpile(qc, B)
    assert tc.duration_ns == pytest.approx(200.0)
    assert sum(1 for x in tc.timed if x.name == 'flux') == 2

@pytest.mark.parametrize('k', [3, 4, 5, 6])
def test_cknot_duration_formula(k):
    qc = QCircuit(k+1); qc.mcx(list(range(k)), k); tc = transpile(qc, B)
    assert tc.duration_ns == pytest.approx(360*k - 520)

@pytest.mark.parametrize('k', [3, 4, 5])
def test_ckry_duration_with_fast_c2x12(k):
    b = HeavyHexBackend(3, noise='ideal', durations={'c2x12': 60.0, 'sx': 0.0})
    qc = QCircuit(k+1); qc.mcx(list(range(k)), k); tc = transpile(qc, b)
    assert tc.duration_ns == pytest.approx(140*k - 80)

def test_parallel_gates_overlap():
    qc = QCircuit(4); qc.x(0); qc.x(3); tc = transpile(qc, B)
    assert tc.duration_ns == pytest.approx(32.0)

def test_measure_at_makespan():
    qc = QCircuit(3); qc.ccx(0, 1, 2); qc.measure_all(); tc = transpile(qc, B)
    ms = [x for x in tc.timed if x.name == 'measure']
    assert len(ms) == 3 and all(x.t0 == pytest.approx(tc.duration_ns) for x in ms)

def test_no_overlap_per_qubit():
    qc = QCircuit(5); qc.h(0); qc.mcx([0, 1, 2, 3], 4); qc.ry(0.3, 2); qc.mcry(0.5, [4, 0], 1); tc = transpile(qc, B)
    busy = {}
    for x in sorted(tc.timed, key=lambda x: x.t0):
        if x.name == 'measure' or x.duration == 0: continue
        for q in x.qubits:
            assert x.t0 >= busy.get(q, 0.0) - 1e-9; busy[q] = x.t0 + x.duration
