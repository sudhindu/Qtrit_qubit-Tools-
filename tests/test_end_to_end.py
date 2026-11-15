# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The qtflow authors
"""End-to-end: user circuit -> transpile -> simulate -> pulses -> plots."""
import numpy as np
import pytest
import matplotlib.pyplot as plt
from qtflow import QCircuit, HeavyHexBackend, transpile, run, PulseProgram, plotting, ideal_probabilities

B = HeavyHexBackend(3, noise='reference')

@pytest.mark.parametrize('inp', range(8))
def test_toffoli_truth_table(inp):
    qc = QCircuit(3)
    for q in range(3):
        if (inp >> q) & 1: qc.x(q)
    qc.ccx(0, 1, 2); tc = transpile(qc, B)
    out = inp ^ (4 if (inp & 3) == 3 else 0)
    assert ideal_probabilities(tc) == pytest.approx({format(out, '03b'): 1.0})

def grover3(marked=5):
    qc = QCircuit(3, 'grover3')
    for q in range(3): qc.h(q)
    for _ in range(2):
        for q in range(3):
            if not (marked >> q) & 1: qc.x(q)
        qc.h(2); qc.ccx(0, 1, 2); qc.h(2)
        for q in range(3):
            if not (marked >> q) & 1: qc.x(q)
        for q in range(3): qc.h(q); qc.x(q)
        qc.h(2); qc.ccx(0, 1, 2); qc.h(2)
        for q in range(3): qc.x(q); qc.h(q)
    qc.measure_all(); return qc

@pytest.mark.parametrize('marked', [0, 5, 7])
def test_grover_peak(marked):
    tc = transpile(grover3(marked), B)
    P = ideal_probabilities(tc); assert P[format(marked, '03b')] > 0.94
    r = run(tc, shots=2000, trajectories=60); assert max(r['counts'], key=r['counts'].get) == format(marked, '03b')

def test_cknot_noisy_success():
    qc = QCircuit(4); [qc.x(q) for q in range(3)]; qc.mcx([0, 1, 2], 3); tc = transpile(qc, B)
    p = run(tc, shots=100, trajectories=200)['probabilities'].get('1111', 0)
    assert 0.85 < p < 1.0

@pytest.mark.filterwarnings('ignore::UserWarning')
def test_full_flow_and_plots(tmp_path):
    qc = QCircuit(4, 'flow'); qc.h(0); qc.mcry(0.9, [0, 1], 2); qc.mcrx(0.4, [0, 2], 3); qc.c1x12(0, 1); qc.measure_all()
    tc = transpile(qc, B); r = run(tc, shots=500, trajectories=30); p = PulseProgram(tc)
    fig, ax = plt.subplots(4, 1, figsize=(4, 8))
    plotting.draw_logical(qc, ax[0]); plotting.draw_native(tc, ax[1])
    plotting.plot_histogram([r['counts']], ax[2]); plotting.plot_pulse_timeline(p, ax[3])
    fig.savefig(tmp_path / 'f.png'); assert (tmp_path / 'f.png').stat().st_size > 1000
    assert 'two-body pulses' in tc.summary()
