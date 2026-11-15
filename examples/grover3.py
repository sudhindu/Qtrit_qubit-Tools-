# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The qtflow authors
"""3-qubit Grover search (2 iterations, marked state 101) with Toffoli-based oracle and diffuser."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from qtflow import QCircuit, HeavyHexBackend, transpile, run, plotting

def ccz(qc): qc.h(2); qc.ccx(0, 1, 2); qc.h(2)
marked = '101'                                   # qubit 0 rightmost
qc = QCircuit(3, 'grover3')
for q in range(3): qc.h(q)
for _ in range(2):
    flips = [q for q in range(3) if marked[::-1][q] == '0']
    for q in flips: qc.x(q)
    ccz(qc)
    for q in flips: qc.x(q)
    for q in range(3): qc.h(q); qc.x(q)
    ccz(qc)
    for q in range(3): qc.x(q); qc.h(q)
qc.measure_all()

tc = transpile(qc, HeavyHexBackend(3, noise='reference')); print(tc.summary())
ideal = run(tc, shots=4000, noise=None); noisy = run(tc, shots=4000, trajectories=300)
print('ideal', ideal['probabilities']); print('noisy', noisy['counts'])
ax = plotting.plot_histogram([ideal['counts'], noisy['counts']], labels=['ideal', 'noisy'], title='Grover, marked ' + marked)
plt.tight_layout(); plt.savefig('grover3_hist.png', dpi=200)
