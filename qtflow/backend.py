# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The qtflow authors
"""HeavyHexBackend: coupling map, gate durations, frequencies and noise parameters of a heavy-hex qutrit device.

The coupling map is Qiskit's heavy-hex graph (CouplingMap.from_heavy_hex(distance)); edges are undirected.
All numbers are model defaults (see README) and can be overridden.
"""
from dataclasses import dataclass, field
import networkx as nx
from qiskit.transpiler import CouplingMap

DEFAULT_DURATIONS_NS = {'rz': 0.0, 'sx': 32.0, 'x': 32.0, 'cz': 68.0,
                        'c1x12': 60.0, 'c2x01': 60.0, 'c2x12': 170.0, 'flux': 10.0, 'measure': 1000.0}

@dataclass
class NoiseModel:
    """thermal relaxation + gate depolarizing (+ optional readout error), per qutrit / per gate"""
    T1_10_us: float = 27.0            # |1> -> |0>
    T1_21_us: float = 17.0            # |2> -> |1>  (faster than |1>, as measured on transmons)
    Tphi_us: float = 40.0             # pure dephasing; L = sqrt(2/Tphi) n  (|0>-|2> coherence dephases 4x faster)
    gate_error: dict = field(default_factory=lambda: {'sx': 2.4e-4, 'x': 2.4e-4, 'cz': 2.0e-3,
                                                      'c1x12': 6.0e-3, 'c2x01': 3.0e-3, 'c2x12': 1.8e-2})
    readout_error: float = 0.0        # symmetric assignment error between neighbouring levels

    @classmethod
    def preset(cls, name):
        if name == 'ideal': return None
        if name == 'reference': return cls()
        if name == 'improved': return cls(100.0, 60.0, 150.0)
        if name == 'ibm_like': return cls(200.0, 100.0, 200.0)
        raise KeyError(name)

class HeavyHexBackend:
    def __init__(self, distance=3, durations=None, noise='reference', f_control_GHz=5.00, f_target_GHz=5.20):
        self.coupling_map = CouplingMap.from_heavy_hex(distance)
        self.num_qubits = self.coupling_map.size()
        self.graph = nx.Graph(); self.graph.add_nodes_from(range(self.num_qubits))
        self.graph.add_edges_from((a, b) for a, b in self.coupling_map.get_edges())
        self.durations = dict(DEFAULT_DURATIONS_NS); self.durations.update(durations or {})
        self.noise = NoiseModel.preset(noise) if isinstance(noise, str) else noise
        self.f_control_GHz, self.f_target_GHz = f_control_GHz, f_target_GHz
        self.name = 'heavy_hex_d%d' % distance

    def adjacent(self, a, b): return self.graph.has_edge(a, b)
    def is_path(self, nodes): return all(self.adjacent(a, b) for a, b in zip(nodes[:-1], nodes[1:]))
    def __repr__(self):
        return 'HeavyHexBackend(%s, %d qutrits, %d couplers)' % (self.name, self.num_qubits, self.graph.number_of_edges())
