# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The qtflow authors
"""3-qutrit demo used in the paper figure.

Input: IBM single-qubit gates (h, x) and the three level-selective native gates.
  h(q0); x(q1); C1X12(q0,q1); C2X01(q1,q2); C2X12(q1,q2)
Ideal output: (|0,1,0> + |1,2,2>)/sqrt(2)  ->  bitstrings '010' and '221' (qubit 0 rightmost), 0.5 each.
Writes: demo_3q.png, demo_3q_pulses.json, demo_3q_qua.py, demo_3q_qua_config.json in the current folder.
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from qtflow import QCircuit, HeavyHexBackend, transpile, run, PulseProgram, plotting

qc = QCircuit(3, 'demo_3q')
qc.h(0); qc.x(1); qc.c1x12(0, 1); qc.c2x01(1, 2); qc.c2x12(1, 2); qc.measure_all()

backend = HeavyHexBackend(distance=3, noise='reference')        # T1 27/17 us, Tphi 40 us, gate errors (README)
tc = transpile(qc, backend)
ideal = run(tc, shots=4000, noise=None)
noisy = run(tc, shots=4000, trajectories=400)
prog = PulseProgram(tc)
prog.to_json('demo_3q_pulses.json'); prog.to_qua('demo_3q_qua.py', config_path='demo_3q_qua_config.json')
print(tc.summary()); print('ideal', ideal['probabilities']); print('noisy', noisy['probabilities'])

plt.rcParams.update({'font.size': 7})
fig = plt.figure(figsize=(7.0, 2.3))                       # two-column (figure*) layout
gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 0.85, 1.15], hspace=0.35, wspace=0.42)
plotting.draw_logical(qc, ax=fig.add_subplot(gs[0, 0]), title='(a) input circuit')
plotting.draw_native(tc, ax=fig.add_subplot(gs[1, 0]), title='(b) native circuit (heavy-hex)')
axh = plotting.plot_histogram([ideal['counts'], noisy['counts']], ax=fig.add_subplot(gs[:, 1]), labels=['ideal', 'noisy'],
                              title='(c) outcomes, 4000 shots', top=8)
axh.legend(frameon=False, fontsize=6, loc='center', bbox_to_anchor=(0.6, 0.62))
plotting.plot_pulse_timeline(prog, ax=fig.add_subplot(gs[:, 2]), title='(d) pulse program')
fig.savefig('demo_3q.png', dpi=300, bbox_inches='tight')
json.dump(dict(summary=tc.summary(), ideal=ideal['probabilities'], noisy=noisy['probabilities'], counts=noisy['counts']),
          open('demo_3q_results.json', 'w'), indent=1)
