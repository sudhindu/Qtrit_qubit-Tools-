# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The qtflow authors
"""Toffoli with three native pulses: truth table (ideal), noisy run and pulse export."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from qtflow import QCircuit, HeavyHexBackend, transpile, run, PulseProgram, ideal_probabilities

backend = HeavyHexBackend(distance=3, noise='reference')
for inp in range(8):
    qc = QCircuit(3, 'toffoli_%03d' % inp)
    for q in range(3):
        if (inp >> q) & 1: qc.x(q)
    qc.ccx(0, 1, 2); qc.measure_all()
    tc = transpile(qc, backend)
    print(format(inp, '03b'), '->', ideal_probabilities(tc), ' noisy:', run(tc, shots=2000, trajectories=200)['probabilities'])
print(tc.summary()); print([str(i) for i in tc.native])
PulseProgram(tc).to_qua('toffoli_qua.py', config_path='toffoli_qua_config.json')
