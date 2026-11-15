# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The qtflow authors
"""Import a Qiskit circuit, transpile it with qtflow and export the pulse program as JSON."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from qiskit import QuantumCircuit
from qtflow import QCircuit, HeavyHexBackend, transpile, run, PulseProgram

q = QuantumCircuit(4)
q.h(0); q.cx(0, 1); q.ccx(0, 1, 2); q.mcx([0, 1, 2], 3); q.measure_all()
qc = QCircuit.from_qiskit(q)
tc = transpile(qc, HeavyHexBackend(3)); print(tc.summary())
print(run(tc, shots=2000, trajectories=200)['counts'])
PulseProgram(tc).to_json('from_qiskit_pulses.json')
