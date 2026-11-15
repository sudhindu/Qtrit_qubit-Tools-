# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The qtflow authors
"""Layout and routing on the heavy-hex lattice.

* Initial layout: logical qubits are placed in order along a long simple path of the lattice.
* Two-qubit gates (cx, cz, swap, native qutrit gates): one operand is moved by SWAPs until the pair is adjacent.
* Multi-controlled gates (ccx, mcx, mcry, mcrx): the level-selective decomposition needs a LINE
  q1 - q2 - ... - qk - target. The router enumerates simple paths of length k+1 near the operands, assigns the
  target to an end of the path and the controls (in any order) to the other nodes with minimum total distance
  (Hungarian assignment), and brings each qubit to its node by SWAPs along shortest paths that avoid nodes already
  placed. Each SWAP costs 3 CZ on the qubit subspace (data must be in {|0>,|1>} between blocks).
This is a simple greedy router (lab version); it is correct by construction and verified in tests/.
"""
import itertools, warnings
import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment
from .circuit import Instruction

def long_path(G, n):
    """a simple path with at least n nodes (DFS from low-degree nodes, longest found)"""
    best = []
    starts = sorted(G.nodes, key=lambda v: G.degree[v])[:8]
    for s in starts:
        stack = [(s, [s])]; seen = 0
        while stack and seen < 20000:
            v, path = stack.pop(); seen += 1
            if len(path) > len(best): best = path
            if len(best) >= n: return best[:n]
            for w in G.neighbors(v):
                if w not in path: stack.append((w, path + [w]))
    if len(best) < n:
        raise ValueError('lattice too small for %d qubits on a path' % n)
    return best[:n]

class Router:
    def __init__(self, backend, n_logical, initial_layout=None):
        self.G = backend.graph; self.backend = backend
        if n_logical > backend.num_qubits: raise ValueError('circuit larger than device')
        phys = list(initial_layout) if initial_layout is not None else long_path(self.G, n_logical)
        self.l2p = {l: p for l, p in enumerate(phys)}
        self.p2l = {p: l for l, p in self.l2p.items()}
        self.initial_layout = dict(self.l2p); self.out = []; self.n_swaps = 0
        self.dist = dict(nx.all_pairs_shortest_path_length(self.G))

    # ---------------------------------------------------------------- primitives
    def _swap(self, a, b):
        self.out.append(Instruction('swap', (a, b))); self.n_swaps += 1
        la, lb = self.p2l.get(a), self.p2l.get(b)
        if la is not None: self.l2p[la] = b
        if lb is not None: self.l2p[lb] = a
        self.p2l.pop(a, None); self.p2l.pop(b, None)
        if la is not None: self.p2l[b] = la
        if lb is not None: self.p2l[a] = lb

    def _move(self, l, v, frozen):
        """move logical l to physical v avoiding frozen nodes when possible"""
        u = self.l2p[l]
        if u == v: return
        H = self.G.subgraph([w for w in self.G.nodes if w not in frozen or w in (u, v)])
        try: path = nx.shortest_path(H, u, v)
        except nx.NetworkXNoPath: path = nx.shortest_path(self.G, u, v)
        for a, b in zip(path[:-1], path[1:]): self._swap(a, b)

    # ---------------------------------------------------------------- two-qubit
    def route_pair(self, ins):
        a, b = (self.l2p[q] for q in ins.qubits)
        if not self.G.has_edge(a, b):
            path = nx.shortest_path(self.G, a, b)
            for x, y in zip(path[:-2], path[1:-1]): self._swap(x, y)
        self.out.append(Instruction(ins.name, tuple(self.l2p[q] for q in ins.qubits), ins.params))

    # ---------------------------------------------------------------- multi-controlled (line)
    def _paths(self, m, seeds, radius):
        nodes = {w for s in seeds for w, d in self.dist[s].items() if d <= radius}
        found = []
        for s in nodes:
            stack = [[s]]
            while stack:
                p = stack.pop()
                if len(p) == m: found.append(p); continue
                for w in self.G.neighbors(p[-1]):
                    if w not in p and w in nodes: stack.append(p + [w])
        return found

    def route_chain(self, ins):
        ctrls, tgt = list(ins.qubits[:-1]), ins.qubits[-1]; m = len(ins.qubits)
        seeds = [self.l2p[q] for q in ins.qubits]; best = None
        for radius in (2, 3, 4, 6, 99):
            for p in self._paths(m, seeds, radius):
                for path in (p, p[::-1]):
                    tnode, cnodes = path[-1], path[:-1]
                    C = np.array([[self.dist[self.l2p[c]][v] for v in cnodes] for c in ctrls], float)
                    r, cidx = linear_sum_assignment(C)
                    cost = C[r, cidx].sum() + self.dist[self.l2p[tgt]][tnode]
                    if best is None or cost < best[0]:
                        assign = {ctrls[i]: cnodes[j] for i, j in zip(r, cidx)}; assign[tgt] = tnode
                        best = (cost, path, assign)
            if best is not None: break
        cost, path, assign = best
        for _ in range(4):                      # place qubits, nearest first; repeat if a later move disturbed an earlier one
            frozen = set()
            for l in sorted(assign, key=lambda q: self.dist[self.l2p[q]][assign[q]]):
                self._move(l, assign[l], frozen); frozen.add(assign[l])
            if all(self.l2p[l] == assign[l] for l in assign): break
        if not all(self.l2p[l] == assign[l] for l in assign): raise RuntimeError('routing failed for ' + repr(ins))
        order = [self.p2l[v] for v in path]                  # logical order along the line, target last
        self.out.append(Instruction(ins.name, tuple(self.l2p[q] for q in order), ins.params))

    def run(self, data):
        has_native_qutrit = any(i.name in ('c1x12', 'c2x01', 'c2x12') for i in data)
        for ins in data:
            if len(ins.qubits) == 1: self.out.append(Instruction(ins.name, (self.l2p[ins.qubits[0]],), ins.params))
            elif len(ins.qubits) == 2 and ins.name not in ('mcry', 'mcrx'): self.route_pair(ins)
            else: self.route_chain(ins)
        if has_native_qutrit and self.n_swaps:
            warnings.warn('SWAPs were inserted in a circuit with explicit native qutrit gates; SWAP acts on the '
                          'qubit subspace only, so results are valid only if no qutrit holds |2> at a SWAP.')
        return self.out
