# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The qtflow authors
"""transpile(): logical QCircuit -> routed, native, scheduled circuit on a HeavyHexBackend."""
from dataclasses import dataclass, field
from typing import Dict, List
from .circuit import QCircuit, Instruction
from .routing import Router
from .decompose import expand, merge_rz
from .schedule import schedule
from .gates import NATIVE_2Q

SUPPORTED = {'h', 'x', 'y', 'z', 's', 'sdg', 't', 'tdg', 'sx', 'rx', 'ry', 'rz', 'id', 'cx', 'cz', 'swap',
             'ccx', 'mcx', 'mcry', 'mcrx', 'c1x12', 'c2x01', 'c2x12'}

@dataclass
class TranspiledCircuit:
    logical: QCircuit
    backend: object
    initial_layout: Dict[int, int]
    final_layout: Dict[int, int]
    routed: List[Instruction]
    native: List[Instruction]
    timed: list
    duration_ns: float
    n_swaps: int

    @property
    def active_qubits(self):
        s = set(self.initial_layout.values()) | set(self.final_layout.values())
        for ins in self.native: s |= set(ins.qubits)
        return sorted(s)

    def count_ops(self):
        out = {}
        for ins in self.native: out[ins.name] = out.get(ins.name, 0) + 1
        return out

    def two_body_pulses(self):
        return sum(1 for ins in self.native if ins.name in NATIVE_2Q)

    def summary(self):
        c = self.count_ops()
        return ('%s on %s: %d two-body pulses (%s), %d single-qubit pulses, %d SWAPs, schedule %.0f ns (+ readout)' % (
            self.logical.name, self.backend.name, self.two_body_pulses(),
            ', '.join('%s %d' % (k, c[k]) for k in ('cz', 'c1x12', 'c2x01', 'c2x12') if k in c),
            c.get('sx', 0) + c.get('x', 0), self.n_swaps, self.duration_ns))

def transpile(qc: QCircuit, backend, initial_layout=None, optimize=True):
    for ins in qc.data:
        if ins.name not in SUPPORTED: raise ValueError('unsupported gate: ' + ins.name)
    router = Router(backend, qc.num_qubits, initial_layout)
    routed = router.run(qc.data)
    native = [n for ins in routed for n in expand(ins)]
    if optimize: native = merge_rz(native)
    for ins in native:                       # hardware constraint check
        if len(ins.qubits) == 2 and not backend.adjacent(*ins.qubits):
            raise RuntimeError('non-adjacent native gate ' + repr(ins))
    timed, T = schedule(native, backend.durations, measure=qc.measured, qubits=list(router.l2p.values()))
    return TranspiledCircuit(qc, backend, router.initial_layout, dict(router.l2p), routed, native, timed, T, router.n_swaps)
