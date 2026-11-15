# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The qtflow authors
"""Equivalence checks between a logical circuit and its transpiled native circuit.

* qubit circuits (cx, ccx, mcx, mcry, mcrx, 1q gates ...): the reference unitary is built with Qiskit's
  Operator (an independent implementation); the native circuit is simulated on qutrits for every computational
  basis input. Leakage out of the qubit subspace, and population left on routing ancillas, is reported.
* qutrit circuits (1q gates + cz + c1x12/c2x01/c2x12): the reference is the qutrit unitary built directly from
  the gate definitions. Inputs are the 2^n qubit basis states (circuits start in the qubit subspace); outputs are
  compared over all 3^n levels, so states that end in |2> (e.g. the demo circuit) are checked too.
Both checks allow one global phase. (A single-qubit gate applied while a qutrit holds |2> would make the Euler
global phase physical; the transpiler output is exact for every circuit in which that does not happen.)
"""
import itertools
import numpy as np
from .gates import native_unitary, embed_1q, LOGICAL_1Q, QUTRIT_NATIVE, D
from .simulator import _apply

def logical_unitary(qc):
    """2^n x 2^n unitary of a qubit-level QCircuit (Qiskit little-endian: index = sum_l b_l 2^l)"""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    q = QuantumCircuit(qc.num_qubits)
    for ins in qc.data:
        nm, qs, p = ins.name, list(ins.qubits), list(ins.params)
        if nm in QUTRIT_NATIVE: raise ValueError('qutrit gate %s has no qubit-level reference' % nm)
        if nm == 'id': continue
        if nm in ('mcx', 'ccx'): q.mcx(qs[:-1], qs[-1])
        elif nm == 'mcry': q.mcry(p[0], qs[:-1], qs[-1])
        elif nm == 'mcrx': q.mcrx(p[0], qs[:-1], qs[-1])
        else: getattr(q, nm)(*(p + qs))
    return Operator(q).data

def qutrit_unitary(qc):
    """3^n x 3^n unitary of a circuit made of 1q gates, cz and the level-selective gates (index = sum_l t_l 3^l)"""
    n = qc.num_qubits; U = np.zeros((D**n, D**n), complex)
    for col in range(D**n):
        lev = [(col // D**l) % D for l in range(n)]
        psi = np.zeros((D,)*n, complex); psi[tuple(lev)] = 1
        for ins in qc.data:
            if len(ins.qubits) == 1: G = embed_1q(LOGICAL_1Q[ins.name](ins.params))
            elif ins.name in QUTRIT_NATIVE or ins.name == 'cz': G = native_unitary(ins.name)
            else: raise ValueError('gate %s not allowed in a qutrit reference circuit' % ins.name)
            psi = _apply(psi, G, list(ins.qubits), n)
        for out in np.ndindex(*psi.shape):
            U[sum(out[l]*D**l for l in range(n)), col] = psi[out]
    return U

def _run_native(tc, levels_logical):
    act = tc.active_qubits; idx = {p: i for i, p in enumerate(act)}; n = len(act)
    lev = [0]*n
    for l, v in enumerate(levels_logical): lev[idx[tc.initial_layout[l]]] = v
    psi = np.zeros((D,)*n, complex); psi[tuple(lev)] = 1
    for ins in tc.native:
        psi = _apply(psi, native_unitary(ins.name, ins.params), [idx[q] for q in ins.qubits], n)
    return psi, idx

def _match_phase(M, U):
    k = np.unravel_index(np.argmax(np.abs(U)), U.shape)
    ph = M[k]/U[k] if abs(U[k]) > 0 else 1.0
    ph = ph/abs(ph) if abs(ph) > 1e-12 else 1.0
    return float(np.max(np.abs(M - ph*U)))

def check_equivalence(qc, tc, levels=None):
    """returns dict(max_error, max_leakage, dim). levels=2 (qubit check) or 3 (qutrit check); default: auto"""
    n = qc.num_qubits
    if len(tc.active_qubits) > 12: raise ValueError('too many active qutrits for a dense check (%d)' % len(tc.active_qubits))
    if levels is None: levels = 3 if any(i.name in QUTRIT_NATIVE for i in qc.data) else 2
    U = qutrit_unitary(qc) if levels == 3 else logical_unitary(qc)
    if levels == 3:                                   # keep qubit-basis input columns only
        cols3 = [sum(((c >> l) & 1)*D**l for l in range(n)) for c in range(2**n)]; U = U[:, cols3]
    M = np.zeros_like(U); leak = 0.0
    for col in range(2**n):
        lev = [(col >> l) & 1 for l in range(n)]
        psi, idx = _run_native(tc, lev)
        tot = 0.0
        for row in range(levels**n):
            out = [0]*len(idx)
            for l in range(n): out[idx[tc.final_layout[l]]] = (row // levels**l) % levels
            M[row, col] = psi[tuple(out)]; tot += abs(M[row, col])**2
        leak = max(leak, 1.0 - tot)
    return dict(max_error=_match_phase(M, U), max_leakage=float(leak), dim=(levels**n, 2**n))

def basis_inputs(n):
    return list(itertools.product((0, 1), repeat=n))
