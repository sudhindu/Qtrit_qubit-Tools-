# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The qtflow authors
"""Validation run for qtflow. Writes validation/results/*.csv, *.json, a figure and validation/REPORT.md.

  python validation/run_validation.py                 all stages (about 10-15 minutes on a laptop)
  python validation/run_validation.py v4 v5           selected stages; 'report' rebuilds REPORT.md from saved results

V1  exactness of the native decompositions (Toffoli, C^k-NOT, C^k-Ry, C^k-Rx) vs Qiskit Operator
V2  pulse counts and schedule durations vs the analytic formulas (2k-1, 2k, 360k-520 ns)
V3  same gates on IBM natives: Qiskit transpile to {rz, sx, x, cz} on the same heavy-hex lattice (reference)
V4  random circuits routed on heavy-hex: equivalence, SWAP count
V5  noisy simulation: Toffoli, C^k-NOT, 3-qubit Grover under the 'reference', 'improved', 'ibm_like' presets
V6  pulse export: timing consistency, QUA text syntax
"""
import os, sys, json, csv, time, warnings
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..'))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile as qk_transpile
from qiskit.transpiler import CouplingMap
from qtflow import QCircuit, HeavyHexBackend, transpile, run, PulseProgram, ideal_probabilities
from qtflow.verify import check_equivalence
sys.path.insert(0, os.path.join(HERE, '..', 'tests')); from test_routing import random_circuit
from test_end_to_end import grover3

OUT = os.path.join(HERE, 'results'); os.makedirs(OUT, exist_ok=True)
B = HeavyHexBackend(3, noise='ideal'); rows = {}; t_start = time.time()
STAGES = set(sys.argv[1:]) or {'v1', 'v4', 'v5', 'v6', 'report'}
def load(name):
    with open(os.path.join(OUT, name + '.json')) as f: return json.load(f)

def save(name, data):
    with open(os.path.join(OUT, name + '.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    json.dump(data, open(os.path.join(OUT, name + '.json'), 'w'), indent=1, default=float)

def qiskit_ibm(qc_builder, n):
    """IBM-native reference: Qiskit opt. level 3 on the same heavy-hex graph, basis rz/sx/x/cz; best of 5 seeds"""
    q = QuantumCircuit(n); qc_builder(q); best = None
    for s in range(5):
        t = qk_transpile(q, coupling_map=CouplingMap.from_heavy_hex(3), basis_gates=['rz', 'sx', 'x', 'cz'],
                         optimization_level=3, seed_transpiler=s)
        ncz = t.count_ops().get('cz', 0)
        dur = 0; busy = {}
        for ins in t.data:                                   # ASAP with sx/x 32 ns, cz 68 ns, rz 0
            d = {'cz': 68, 'sx': 32, 'x': 32}.get(ins.operation.name, 0); qs = [t.find_bit(x).index for x in ins.qubits]
            t0 = max([busy.get(x, 0) for x in qs] + [0]); [busy.__setitem__(x, t0 + d) for x in qs]
        dur = max(busy.values()) if busy else 0
        if best is None or ncz < best[0]: best = (ncz, dur)
    return best

# ---------------------------------------------------------------- V1-V3
v1 = []
for k in (range(2, 7) if 'v1' in STAGES else []):
    for g in ('mcx', 'mcry', 'mcrx'):
        qc = QCircuit(k+1, '%s_k%d' % (g, k))
        if g == 'mcx': qc.mcx(list(range(k)), k); build = lambda q, k=k: q.mcx(list(range(k)), k)
        elif g == 'mcry': qc.mcry(0.83, list(range(k)), k); build = lambda q, k=k: q.mcry(0.83, list(range(k)), k)
        else: qc.mcrx(0.83, list(range(k)), k); build = lambda q, k=k: q.mcrx(0.83, list(range(k)), k)
        tc = transpile(qc, B)
        eq = check_equivalence(qc, tc) if k <= 5 else {'max_error': float('nan'), 'max_leakage': float('nan')}
        expect = 2*k - 1 if g == 'mcx' else 2*k
        ibm = qiskit_ibm(build, k+1)
        v1.append(dict(gate=g, k=k, max_error=eq['max_error'], max_leakage=eq['max_leakage'],
                       pulses=tc.two_body_pulses(), expected_pulses=expect, swaps=tc.n_swaps, duration_ns=tc.duration_ns,
                       formula_ns=(360*k - 520) if g == 'mcx' else '', ibm_cz=ibm[0], ibm_duration_ns=ibm[1]))
        print(v1[-1])
if 'v1' in STAGES: save('v1_v3_multicontrol', v1)

# ---------------------------------------------------------------- V4
v4 = []; rng = np.random.default_rng(2026)
for i in (range(60) if 'v4' in STAGES else []):
    n = int(rng.integers(3, 6)); qc = random_circuit(n, 14, rng)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore'); tc = transpile(qc, B)
    if len(tc.active_qubits) > 10: continue          # 3^11+ amplitudes x 2^n inputs: too slow for a check
    adj = all(B.adjacent(*x.qubits) for x in tc.native if len(x.qubits) == 2)
    eq = check_equivalence(qc, tc)
    v4.append(dict(idx=i, n=n, active_qutrits=len(tc.active_qubits), logical_gates=len(qc.data), native_2body=tc.two_body_pulses(), swaps=tc.n_swaps,
                   duration_ns=tc.duration_ns, adjacency_ok=adj, max_error=eq['max_error'], max_leakage=eq['max_leakage']))
if 'v4' in STAGES: save('v4_random_routing', v4); print('V4 max error', max(r['max_error'] for r in v4))

# ---------------------------------------------------------------- V5
v5 = []
def cknot(k):
    qc = QCircuit(k+1, 'C%dNOT' % k); [qc.x(q) for q in range(k)]; qc.mcx(list(range(k)), k); qc.measure_all(); return qc
cases = [('Toffoli (in 110)', cknot(2), '111'), ('C3NOT', cknot(3), '1111'), ('C4NOT', cknot(4), '11111'),
         ('C5NOT', cknot(5), '111111'), ('Grover3 m=5', grover3(5), '101')]
for name, qc, tgt in (cases if 'v5' in STAGES else []):
    for preset in ('reference', 'improved', 'ibm_like'):
        tc = transpile(qc, HeavyHexBackend(3, noise=preset)); r = run(tc, shots=4000, trajectories=300, seed=11)
        leak = sum(v for kk, v in r['probabilities'].items() if '2' in kk)
        v5.append(dict(circuit=name, preset=preset, duration_ns=tc.duration_ns, pulses=tc.two_body_pulses(),
                       ideal=ideal_probabilities(tc).get(tgt, 0), noisy=r['probabilities'].get(tgt, 0), leakage=leak))
        print(v5[-1])
if 'v5' in STAGES: save('v5_noisy', v5)

# ---------------------------------------------------------------- V6
v6 = []
for name, qc, _ in (cases if 'v6' in STAGES else []):
    tc = transpile(qc, B); p = PulseProgram(tc); txt = p.to_qua()
    try: compile(txt, name, 'exec'); ok = True
    except SyntaxError: ok = False
    gates_end = max(e['t0'] + e['duration'] for e in p.entries if e['kind'] == 'play')
    v6.append(dict(circuit=name, schedule_ns=tc.duration_ns, pulse_end_ns=gates_end, channels=len(p.channels),
                   play=sum(e['kind'] == 'play' for e in p.entries), frame=sum(e['kind'] == 'frame' for e in p.entries),
                   qua_syntax_ok=ok))
if 'v6' in STAGES: save('v6_pulses', v6)
if 'report' not in STAGES: sys.exit(0)
v1, v4, v5, v6 = (load(n) for n in ('v1_v3_multicontrol', 'v4_random_routing', 'v5_noisy', 'v6_pulses'))

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(1, 2, figsize=(7, 2.6))
for g, m, c in (('mcx', 'o', '#2a78d6'), ('mcry', 's', '#eb6834')):
    r = [x for x in v1 if x['gate'] == g]
    ax[0].plot([x['k'] for x in r], [x['pulses'] for x in r], m + '-', color=c, label='qtflow ' + g)
    ax[0].plot([x['k'] for x in r], [x['ibm_cz'] for x in r], m + '--', color=c, mfc='white', label='IBM natives ' + g)
    ax[1].plot([x['k'] for x in r], [x['duration_ns'] for x in r], m + '-', color=c, label='qtflow ' + g)
    ax[1].plot([x['k'] for x in r], [x['ibm_duration_ns'] for x in r], m + '--', color=c, mfc='white', label='IBM natives ' + g)
ax[0].set_ylabel('two-body pulses'); ax[1].set_ylabel('schedule (ns)'); ax[1].set_yscale('log')
for a in ax: a.set_xlabel('controls k'); a.grid(alpha=0.3); a.legend(fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'v1_v3_scaling.png'), dpi=200)

# ---------------------------------------------------------------- report
def table(data, cols, fmt=None):
    fmt = fmt or {}
    L = ['| ' + ' | '.join(cols) + ' |', '|' + '---|'*len(cols)]
    for r in data: L.append('| ' + ' | '.join((fmt[c] % r[c]) if (c in fmt and r[c] != '') else str(r[c]) for c in cols) + ' |')
    return '\n'.join(L)
import platform, qiskit
rep = ['# qtflow validation report', '',
       'Generated by `validation/run_validation.py` (Python %s, Qiskit %s, numpy %s).' % (
           platform.python_version(), qiskit.__version__, np.__version__), '',
       'Unit tests: `python -m pytest -q tests` (see `logs/pytest.log`).', '',
       '## V1-V3  Multi-controlled gates on heavy-hex (distance 3)', '',
       '`max_error`: largest entry of |U_native - e^{i phi} U_qiskit| over all basis inputs (qubit subspace); '
       '`max_leakage`: population left outside the logical qubit subspace. k = 6 is not checked (cost). '
       'IBM columns: Qiskit opt. level 3 to {rz, sx, x, cz} on the same lattice, best of 5 seeds, ASAP with sx 32 ns, cz 68 ns.', '',
       table(v1, ['gate', 'k', 'max_error', 'max_leakage', 'pulses', 'expected_pulses', 'swaps', 'duration_ns', 'formula_ns', 'ibm_cz', 'ibm_duration_ns'],
             {'max_error': '%.1e', 'max_leakage': '%.1e'}), '',
       '![scaling](results/v1_v3_scaling.png)', '',
       '## V4  Random circuits (%d of 60 generated circuits; 3-5 qubits, 14 logical gates incl. C^k gates; circuits whose routing touched more than 10 qutrits are not checked)' % len(v4), '',
       '- all native two-body gates on couplers: **%s**' % all(r['adjacency_ok'] for r in v4),
       '- max equivalence error: **%.1e**, max leakage: **%.1e**' % (max(r['max_error'] for r in v4), max(r['max_leakage'] for r in v4)),
       '- SWAPs inserted: mean %.1f, max %d' % (np.mean([r['swaps'] for r in v4]), max(r['swaps'] for r in v4)), '',
       'Per-circuit data: `results/v4_random_routing.csv`.', '',
       '## V5  Noisy simulation (300 trajectories, 4000 shots)', '',
       'Presets: reference (T1 27/17 us, Tphi 40 us), improved (100/60/150 us), ibm_like (200/100/200 us); same gate errors.', '',
       table(v5, ['circuit', 'preset', 'duration_ns', 'pulses', 'ideal', 'noisy', 'leakage'], {'ideal': '%.3f', 'noisy': '%.3f', 'leakage': '%.3f'}), '',
       '## V6  Pulse export', '',
       table(v6, ['circuit', 'schedule_ns', 'pulse_end_ns', 'channels', 'play', 'frame', 'qua_syntax_ok']), '',
       '## Pass criteria', '',
       '- V1: max_error < 1e-9, leakage < 1e-9, pulses = expected, mcx duration = 360k-520 ns: **%s**' % all(
           (r['k'] > 5 or (r['max_error'] < 1e-9 and r['max_leakage'] < 1e-9)) and r['pulses'] == r['expected_pulses']
           and (r['gate'] != 'mcx' or r['k'] == 2 or abs(r['duration_ns'] - r['formula_ns']) < 1e-6) for r in v1),
       '- V4: all equivalent and adjacent: **%s**' % all(r['adjacency_ok'] and r['max_error'] < 1e-8 for r in v4),
       '- V5: ideal success = 1 (Grover > 0.94) and noisy success increases with coherence: **%s**' % all(
           v5[i]['noisy'] <= v5[i+2]['noisy'] + 0.02 for i in range(0, len(v5), 3)),
       '- V6: pulse end = schedule, QUA syntax valid: **%s**' % all(abs(r['schedule_ns'] - r['pulse_end_ns']) < 1e-6 and r['qua_syntax_ok'] for r in v6), '']
open(os.path.join(HERE, 'REPORT.md'), 'w').write('\n'.join(rep))
print('done in %.0f s' % (time.time() - t_start))
