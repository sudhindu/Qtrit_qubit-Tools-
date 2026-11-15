# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The qtflow authors
"""Heavy-hex routing: every native two-body gate is on a coupler and the routed circuit is equivalent."""
import numpy as np
import pytest
from qtflow import QCircuit, HeavyHexBackend, transpile
from qtflow.verify import check_equivalence

B3 = HeavyHexBackend(3, noise='ideal')
B5 = HeavyHexBackend(5, noise='ideal')

def random_circuit(n, depth, rng, multi=True):
    qc = QCircuit(n, 'rand')
    for _ in range(depth):
        r = rng.random()
        if r < 0.3: getattr(qc, rng.choice(['h', 'x', 's', 't', 'sx']))(int(rng.integers(n)))
        elif r < 0.4: qc.ry(float(rng.uniform(-np.pi, np.pi)), int(rng.integers(n)))
        elif r < 0.65 or not multi or n < 3:
            a, b = rng.choice(n, 2, replace=False); qc.cx(int(a), int(b))
        else:
            k = int(rng.integers(2, min(n - 1, 4) + 1)); qs = [int(x) for x in rng.choice(n, k + 1, replace=False)]
            g = rng.choice(['mcx', 'mcry', 'mcrx'])
            if g == 'mcx': qc.mcx(qs[:-1], qs[-1])
            elif g == 'mcry': qc.mcry(float(rng.uniform(-3, 3)), qs[:-1], qs[-1])
            else: qc.mcrx(float(rng.uniform(-3, 3)), qs[:-1], qs[-1])
    return qc

def test_coupling_map_heavy_hex():
    assert B3.num_qubits == 19 and B3.graph.number_of_edges() == 20
    assert max(d for _, d in B3.graph.degree()) == 3

@pytest.mark.parametrize('seed', range(12))
def test_random_circuits_equivalent(seed):
    rng = np.random.default_rng(seed); n = int(rng.integers(3, 6))
    qc = random_circuit(n, 14, rng); tc = transpile(qc, B3)
    for ins in tc.native:
        if len(ins.qubits) == 2: assert B3.adjacent(*ins.qubits)
    r = check_equivalence(qc, tc)
    assert r['max_error'] < 1e-8 and r['max_leakage'] < 1e-8

@pytest.mark.parametrize('seed', range(4))
def test_random_on_larger_lattice(seed):
    rng = np.random.default_rng(100 + seed); qc = random_circuit(5, 12, rng)
    tc = transpile(qc, B5); assert check_equivalence(qc, tc)['max_error'] < 1e-8

def test_bad_initial_layout_needs_swaps():
    qc = QCircuit(2); qc.cx(0, 1)
    far = max(((a, b) for a in range(B3.num_qubits) for b in range(B3.num_qubits)), key=lambda e: abs(e[0] - e[1]))
    tc = transpile(qc, B3, initial_layout={0: 0, 1: 18})
    assert tc.n_swaps > 0 and check_equivalence(qc, tc)['max_error'] < 1e-9

def test_layouts_are_injective():
    rng = np.random.default_rng(3); qc = random_circuit(5, 20, rng); tc = transpile(qc, B3)
    assert len(set(tc.initial_layout.values())) == 5 and len(set(tc.final_layout.values())) == 5

def test_chain_is_a_line():
    qc = QCircuit(6); qc.mcx([0, 1, 2, 3, 4], 5); tc = transpile(qc, B3)
    chain = tc.routed[-1].qubits; assert B3.is_path(list(chain))

def test_unsupported_gate():
    qc = QCircuit(2); qc.append('u3', [0], [0.1, 0.2, 0.3])
    with pytest.raises(ValueError): transpile(qc, B3)

def test_native_qutrit_gates_routed_without_swaps():
    qc = QCircuit(3); qc.h(0); qc.x(1); qc.c1x12(0, 1); qc.c2x01(1, 2); qc.c2x12(1, 2)
    tc = transpile(qc, B3); assert tc.n_swaps == 0
    r = check_equivalence(qc, tc); assert r['max_error'] < 1e-9 and r['dim'] == (27, 8)
