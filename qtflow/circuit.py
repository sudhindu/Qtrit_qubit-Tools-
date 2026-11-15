# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The qtflow authors
"""QCircuit: a small Qiskit-like circuit container.

Logical gates : h x y z s sdg t tdg sx rx ry rz id cx cz ccx (toffoli) mcx mcry mcrx swap
Native gates  : rz sx x cz c1x12 c2x01 c2x12 (can be used directly, e.g. for qutrit-level programs)
Example
    qc = QCircuit(3)
    qc.h(0); qc.x(1)
    qc.ccx(0, 1, 2)
    qc.mcry(0.7, [0, 1], 2)
    qc.measure_all()
"""
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class Instruction:
    name: str
    qubits: Tuple[int, ...]
    params: Tuple[float, ...] = ()
    def __repr__(self):
        p = '(' + ', '.join('%.4g' % v for v in self.params) + ')' if self.params else ''
        return '%s%s %s' % (self.name, p, list(self.qubits))

class QCircuit:
    def __init__(self, num_qubits, name='circuit'):
        self.num_qubits = int(num_qubits); self.name = name
        self.data: List[Instruction] = []; self.measured = False

    # ------------------------------------------------------------- generic
    def append(self, name, qubits, params=()):
        qubits = tuple(int(q) for q in qubits)
        if len(set(qubits)) != len(qubits): raise ValueError('repeated qubit in %s %s' % (name, qubits))
        if any(q < 0 or q >= self.num_qubits for q in qubits): raise ValueError('qubit index out of range')
        self.data.append(Instruction(name, qubits, tuple(float(p) for p in params))); return self

    # ------------------------------------------------------------- single-qubit
    def h(self, q): return self.append('h', [q])
    def x(self, q): return self.append('x', [q])
    def y(self, q): return self.append('y', [q])
    def z(self, q): return self.append('z', [q])
    def s(self, q): return self.append('s', [q])
    def sdg(self, q): return self.append('sdg', [q])
    def t(self, q): return self.append('t', [q])
    def tdg(self, q): return self.append('tdg', [q])
    def sx(self, q): return self.append('sx', [q])
    def rx(self, theta, q): return self.append('rx', [q], [theta])
    def ry(self, theta, q): return self.append('ry', [q], [theta])
    def rz(self, theta, q): return self.append('rz', [q], [theta])

    # ------------------------------------------------------------- multi-qubit (logical)
    def cx(self, c, t): return self.append('cx', [c, t])
    def cz(self, a, b): return self.append('cz', [a, b])
    def swap(self, a, b): return self.append('swap', [a, b])
    def ccx(self, c1, c2, t): return self.append('ccx', [c1, c2, t])
    toffoli = ccx
    def mcx(self, controls, target):
        controls = list(controls)
        if len(controls) == 1: return self.cx(controls[0], target)
        if len(controls) == 2: return self.ccx(controls[0], controls[1], target)
        return self.append('mcx', controls + [target])
    def mcry(self, theta, controls, target): return self.append('mcry', list(controls) + [target], [theta])
    def mcrx(self, theta, controls, target): return self.append('mcrx', list(controls) + [target], [theta])

    # ------------------------------------------------------------- native qutrit gates (control first)
    def c1x12(self, c, t): return self.append('c1x12', [c, t])
    def c2x01(self, c, t): return self.append('c2x01', [c, t])
    def c2x12(self, c, t): return self.append('c2x12', [c, t])

    def measure_all(self): self.measured = True; return self

    # ------------------------------------------------------------- utilities
    def count_ops(self):
        out = {}
        for ins in self.data: out[ins.name] = out.get(ins.name, 0) + 1
        return out

    def __repr__(self):
        return 'QCircuit(%d qubits, %d instructions)\n  ' % (self.num_qubits, len(self.data)) + '\n  '.join(map(repr, self.data))

    @classmethod
    def from_qiskit(cls, qc):
        """import a Qiskit QuantumCircuit made of supported gates"""
        out = cls(qc.num_qubits, qc.name)
        alias = {'mcx_gray': 'mcx', 'mcx_recursive': 'mcx', 'mcx_vchain': 'mcx', 'c3x': 'mcx', 'c4x': 'mcx', 'u1': 'rz', 'p': 'rz'}
        for ins in qc.data:
            nm = alias.get(ins.operation.name, ins.operation.name)
            qs = [qc.find_bit(q).index for q in ins.qubits]
            if nm in ('barrier', 'measure'): continue
            if nm == 'mcry' or (nm.startswith('c') and 'ry' in nm and len(qs) > 2): nm = 'mcry'
            out.append(nm, qs, [float(p) for p in ins.operation.params])
        return out
