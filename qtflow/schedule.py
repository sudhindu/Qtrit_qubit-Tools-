# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The qtflow authors
"""As-soon-as-possible scheduling of native instructions with durations.

A qutrit taking part in a level-selective gate is parked at the control or the target frequency. When its role
changes from one native qutrit gate to the next (e.g. the middle qutrit of a Toffoli), a 'flux' move of
durations['flux'] (default 10 ns) is inserted on it before the gate.
"""
from dataclasses import dataclass
from typing import Tuple
from .circuit import Instruction

@dataclass
class Timed:
    name: str
    qubits: Tuple[int, ...]
    params: Tuple[float, ...]
    t0: float
    duration: float

def schedule(native, durations, measure=True, qubits=None):
    busy = {}; role = {}; out = []
    def start(qs): return max([busy.get(q, 0.0) for q in qs] + [0.0])
    for ins in native:
        if ins.name in ('c1x12', 'c2x01', 'c2x12'):
            c, t = ins.qubits
            for q, r in ((c, 'C'), (t, 'T')):
                if q in role and role[q] != r:
                    t0 = busy.get(q, 0.0); d = durations['flux']
                    out.append(Timed('flux', (q,), (1.0 if r == 'C' else -1.0,), t0, d)); busy[q] = t0 + d
                role[q] = r
        d = durations.get(ins.name, 0.0); t0 = start(ins.qubits)
        out.append(Timed(ins.name, ins.qubits, ins.params, t0, d))
        for q in ins.qubits: busy[q] = t0 + d
    makespan = max(busy.values()) if busy else 0.0
    if measure:
        qs = sorted(qubits) if qubits is not None else sorted(busy)
        for q in qs: out.append(Timed('measure', (q,), (), makespan, durations['measure']))
    return out, makespan
