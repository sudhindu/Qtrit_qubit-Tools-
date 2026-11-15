# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The qtflow authors
"""Expansion of routed (physical) circuits into the native gate set.

Multi-controlled gates use the level-selective native gates on a LINE of qutrits (the router guarantees it):
  Toffoli on chain (a, b, t)          : c1x12(a,b) . c2x01(b,t) . c1x12(a,b)                 3 two-qutrit pulses
  C^k-X on chain (q1..qk, t)          : fold [c1x12(q1,q2), c2x12(q2,q3) .. c2x12(q_{k-1},q_k)]
                                        flip c2x01(q_k,t) ; unfold (mirror of fold)            2k-1 pulses
  C^k-Ry(theta) on chain (q1..qk, t)  : ry(theta/2)_t, fold, c2x01, ry(-theta/2)_t, c2x01, unfold   2k pulses
  C^k-Rx(theta)                       : rz(pi/2)_t . C^k-Ry(theta) . rz(-pi/2)_t                 2k pulses
Qubit gates: cx = H_t cz H_t ; swap = 3 cx ; single-qubit gates -> rz/sx (Euler ZSX), x kept native.
"""
import numpy as np
from qiskit.synthesis import OneQubitEulerDecomposer
from .gates import LOGICAL_1Q, NATIVE_1Q
from .circuit import Instruction

_ZSX = OneQubitEulerDecomposer('ZSX')

def one_qubit_native(name, q, params=()):
    if name == 'x': return [Instruction('x', (q,))]
    if name == 'sx': return [Instruction('sx', (q,))]
    if name == 'rz': return [Instruction('rz', (q,), tuple(params))]
    if name == 'id': return []
    U = LOGICAL_1Q[name](params)
    out = []
    for ins in _ZSX(U).data:
        nm = ins.operation.name
        if nm == 'rz':
            th = float(ins.operation.params[0])
            if abs(np.mod(th + np.pi, 2*np.pi) - np.pi) > 1e-12: out.append(Instruction('rz', (q,), (th,)))
        elif nm == 'sx': out.append(Instruction('sx', (q,)))
        else: raise RuntimeError('unexpected Euler gate ' + nm)
    return out

def _h(q): return one_qubit_native('h', q)

def cx_native(c, t): return _h(t) + [Instruction('cz', (c, t))] + _h(t)

def chain_fold(chain):
    """chain = (q1, ..., qk, t); returns the fold list (native two-qutrit instructions)"""
    ctr = chain[:-1]
    fold = [Instruction('c1x12', (ctr[0], ctr[1]))]
    for j in range(2, len(ctr)):
        fold.append(Instruction('c2x12', (ctr[j-1], ctr[j])))
    return fold

def mcx_native(chain):
    ctr, t = chain[:-1], chain[-1]
    if len(ctr) == 1: return cx_native(ctr[0], t)
    fold = chain_fold(chain)
    return fold + [Instruction('c2x01', (ctr[-1], t))] + fold[::-1]

def mcry_native(theta, chain):
    ctr, t = chain[:-1], chain[-1]
    if len(ctr) == 1:      # controlled-Ry with one control: ry(t/2) cx ry(-t/2) cx
        return (one_qubit_native('ry', t, (theta/2,)) + cx_native(ctr[0], t) +
                one_qubit_native('ry', t, (-theta/2,)) + cx_native(ctr[0], t))
    fold = chain_fold(chain)
    return (one_qubit_native('ry', t, (theta/2,)) + fold + [Instruction('c2x01', (ctr[-1], t))] +
            one_qubit_native('ry', t, (-theta/2,)) + [Instruction('c2x01', (ctr[-1], t))] + fold[::-1])

def mcrx_native(theta, chain):
    t = chain[-1]
    return [Instruction('rz', (t,), (np.pi/2,))] + mcry_native(theta, chain) + [Instruction('rz', (t,), (-np.pi/2,))]

def expand(ins):
    """physical macro instruction -> list of native instructions"""
    nm, q, p = ins.name, ins.qubits, ins.params
    if len(q) == 1: return one_qubit_native(nm, q[0], p)
    if nm == 'cz': return [Instruction('cz', q)]
    if nm == 'cx': return cx_native(*q)
    if nm == 'swap':
        a, b = q; return cx_native(a, b) + cx_native(b, a) + cx_native(a, b)
    if nm in ('c1x12', 'c2x01', 'c2x12'): return [Instruction(nm, q)]
    if nm in ('ccx', 'mcx'): return mcx_native(q)
    if nm == 'mcry': return mcry_native(p[0], q)
    if nm == 'mcrx': return mcrx_native(p[0], q)
    raise KeyError('cannot expand ' + nm)

def merge_rz(native):
    """merge consecutive rz on the same qubit (virtual Z bookkeeping)"""
    out = []; last = {}
    for ins in native:
        if ins.name == 'rz':
            q = ins.qubits[0]
            if q in last and out[last[q]] is not None and out[last[q]].name == 'rz':
                prev = out[last[q]]; out[last[q]] = Instruction('rz', (q,), (prev.params[0] + ins.params[0],)); continue
            out.append(ins); last[q] = len(out)-1
        else:
            out.append(ins)
            for q in ins.qubits: last[q] = len(out)-1
    return out
