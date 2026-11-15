# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The qtflow authors
"""Simulation of transpiled circuits on qutrit (3-level) hardware.

* statevector(tc): exact, noiseless, on the active physical qutrits.
* run(tc, shots, ...): noisy Monte-Carlo wave-function (quantum trajectories) simulation with
    - thermal relaxation of every qutrit during idle AND gate time (ladder decay |1>->|0> with T1_10, |2>->|1> with
      T1_21, pure dephasing sqrt(2/Tphi) n), exact Kraus operators of the 3-level Lindblad channel;
    - gate errors: after a gate, with probability p_gate a random non-identity qutrit Weyl operator X^a Z^b is
      applied to each qutrit of the gate (qutrit depolarizing);
    - optional symmetric readout (assignment) error between neighbouring levels.
  Returns counts and probabilities over the LOGICAL qubits. Level |2> at readout appears as the digit '2'
  (leakage is visible). Bitstrings follow Qiskit's convention: qubit 0 is the RIGHTMOST character.
"""
import numpy as np
from scipy.linalg import expm
from .gates import native_unitary, D

def _apply(psi, U, qs, n):
    m = len(qs); Ut = U.reshape((D,)*(2*m))
    psi = np.tensordot(Ut, psi, axes=(list(range(m, 2*m)), list(qs)))
    return np.moveaxis(psi, list(range(m)), list(qs))

def _active(tc):
    act = tc.active_qubits; return act, {p: i for i, p in enumerate(act)}

def statevector(tc):
    act, idx = _active(tc); n = len(act)
    psi = np.zeros((D,)*n, complex); psi[(0,)*n] = 1
    for ins in tc.native:
        psi = _apply(psi, native_unitary(ins.name, ins.params), [idx[q] for q in ins.qubits], n)
    return psi, act

def logical_probabilities(prob_active, tc, act):
    """marginal level distribution of the logical qubits (final layout) -> dict bitstring -> probability"""
    idx = {p: i for i, p in enumerate(act)}; n = tc.logical.num_qubits
    axes = [idx[tc.final_layout[l]] for l in range(n)]
    others = tuple(i for i in range(len(act)) if i not in axes)
    P = prob_active.sum(axis=others) if others else prob_active
    order = sorted(range(n), key=lambda l: sorted(axes).index(axes[l]))      # axis position of each logical qubit
    P = np.moveaxis(P, [sorted(axes).index(axes[l]) for l in range(n)], list(range(n)))
    out = {}
    for lev in np.ndindex(*P.shape):
        if P[lev] > 1e-12: out[''.join(str(v) for v in lev[::-1])] = float(P[lev])   # qubit 0 rightmost
    return out

def ideal_probabilities(tc):
    psi, act = statevector(tc); return logical_probabilities(np.abs(psi)**2, tc, act)

# ---------------------------------------------------------------- noise channels
def _kraus_relax(t_ns, nm):
    G10, G21, gphi = 1/(nm.T1_10_us*1e3), 1/(nm.T1_21_us*1e3), 1/(nm.Tphi_us*1e3)
    E = lambda a, b: np.outer(np.eye(D)[a], np.eye(D)[b])
    cs = [np.sqrt(G10)*E(0, 1), np.sqrt(G21)*E(1, 2), np.sqrt(2*gphi)*np.diag([0., 1., 2.])]
    I = np.eye(D); L = np.zeros((D*D, D*D), complex)
    for c in cs:                                  # row-major vec: vec(A rho B) = (A kron B^T) vec(rho)
        cdc = c.conj().T @ c
        L += np.kron(c, c.conj()) - 0.5*np.kron(cdc, I) - 0.5*np.kron(I, cdc.T)
    S = expm(L*t_ns)
    J = np.zeros((D*D, D*D), complex)            # Choi matrix J[(i,a),(j,b)] = S(|i><j|)[a,b]
    for i in range(D):
        for j in range(D):
            out = S[:, i*D+j].reshape(D, D)
            for a in range(D):
                for b in range(D): J[i*D+a, j*D+b] = out[a, b]
    w, V = np.linalg.eigh((J + J.conj().T)/2); K = []
    for lam, v in zip(w, V.T):
        if lam > 1e-14: K.append(np.sqrt(lam)*v.reshape(D, D).T)
    return K

_W = None
def _weyl():
    global _W
    if _W is None:
        X = np.roll(np.eye(D), 1, axis=0); Z = np.diag(np.exp(2j*np.pi*np.arange(D)/D))
        _W = [np.linalg.matrix_power(X, a) @ np.linalg.matrix_power(Z, b) for a in range(D) for b in range(D)]
    return _W

def run(tc, shots=4000, noise='backend', trajectories=300, seed=7, return_probabilities=True):
    nm = tc.backend.noise if noise == 'backend' else noise
    rng = np.random.default_rng(seed); act, idx = _active(tc); n = len(act)
    if nm is None:
        P = ideal_probabilities(tc); trajectories = 0
    else:
        kcache = {}
        def kraus(t):
            key = round(t, 3)
            if key not in kcache: kcache[key] = _kraus_relax(t, nm)
            return kcache[key]
        def relax(psi, q, t):
            if t <= 1e-9: return psi
            K = kraus(t); outs = [_apply(psi, k, [q], n) for k in K]
            p = np.array([np.vdot(o, o).real for o in outs]); p /= p.sum()
            k = rng.choice(len(K), p=p); return outs[k]/np.sqrt(np.vdot(outs[k], outs[k]).real)
        timed = sorted([x for x in tc.timed if x.name != 'measure'], key=lambda x: x.t0)
        T_end = tc.duration_ns; acc = np.zeros((D,)*n)
        for _ in range(trajectories):
            psi = np.zeros((D,)*n, complex); psi[(0,)*n] = 1; last = {q: 0.0 for q in range(n)}
            for x in timed:
                qs = [idx[q] for q in x.qubits]
                for q in qs: psi = relax(psi, q, x.t0 - last[q])
                if x.name != 'flux': psi = _apply(psi, native_unitary(x.name, x.params), qs, n)
                for q in qs: psi = relax(psi, q, x.duration); last[q] = x.t0 + x.duration
                p = nm.gate_error.get(x.name, 0.0)
                if p > 0 and rng.random() < p:
                    W = _weyl()
                    while True:
                        ch = rng.integers(0, D*D, size=len(qs))
                        if any(c != 0 for c in ch): break
                    for q, c in zip(qs, ch): psi = _apply(psi, W[c], [q], n)
            for q in range(n): psi = relax(psi, q, T_end - last[q])
            acc += np.abs(psi)**2
        P = logical_probabilities(acc/trajectories, tc, act)
    keys = list(P); probs = np.array([P[k] for k in keys]); probs /= probs.sum()
    samples = rng.choice(len(keys), size=shots, p=probs); counts = {}
    for s in samples:
        k = keys[s]
        if nm is not None and nm.readout_error > 0:
            k = ''.join(_readout(c, nm.readout_error, rng) for c in k)
        counts[k] = counts.get(k, 0) + 1
    res = dict(counts=dict(sorted(counts.items())), shots=shots, trajectories=trajectories)
    if return_probabilities: res['probabilities'] = dict(sorted(P.items()))
    return res

def _readout(c, e, rng):
    v = int(c)
    if rng.random() < e:
        v = min(2, v+1) if (v == 0 or (v == 1 and rng.random() < 0.5)) else v-1
    return str(v)
