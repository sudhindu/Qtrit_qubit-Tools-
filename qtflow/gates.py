# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The qtflow authors
"""Gate set of qtflow.

Every physical element is a transmon used as a qutrit (levels |0>, |1>, |2>). Logical qubit data live in
{|0>, |1>}; the level |2> is used only transiently by the native two-qutrit gates.

Native gate set = IBM native gates + three level-selective two-qutrit gates:
    rz(theta)  virtual frame change (zero duration)          [IBM native]
    sx, x      microwave pulses on the |0>-|1> transition    [IBM native]
    cz         two-qubit controlled-Z on the qubit subspace  [IBM native]
    c1x12      C^1X_12 : if control = |1>, swap target |1> <-> |2>
    c2x01      C^2X_01 : if control = |2>, swap target |0> <-> |1>
    c2x12      C^2X_12 : if control = |2>, swap target |1> <-> |2>
Single-qubit native gates act as identity on |2>; cz acts as identity outside |1,1>.
"""
import numpy as np

D = 3
NATIVE_1Q = ('rz', 'sx', 'x')
NATIVE_2Q = ('cz', 'c1x12', 'c2x01', 'c2x12')
QUTRIT_NATIVE = ('c1x12', 'c2x01', 'c2x12')
LEVEL_SELECTIVE = {'c1x12': (1, 1, 2), 'c2x01': (2, 0, 1), 'c2x12': (2, 1, 2)}   # (control level, j, k)

def embed_1q(U2):
    """embed a 2x2 qubit unitary into the 3-level space (identity on |2>)"""
    U = np.eye(D, dtype=complex); U[:2, :2] = U2; return U

def rz(theta):
    return embed_1q(np.diag([np.exp(-0.5j*theta), np.exp(0.5j*theta)]))

def sx():
    return embed_1q(0.5*np.array([[1+1j, 1-1j], [1-1j, 1+1j]]))

def x():
    return embed_1q(np.array([[0, 1], [1, 0]]))

def cz():
    U = np.eye(D*D, dtype=complex); U[1*D+1, 1*D+1] = -1; return U

def level_selective(name):
    """9x9 permutation of the native two-qutrit gate (order: control, target)"""
    c0, j, k = LEVEL_SELECTIVE[name]; U = np.zeros((D*D, D*D), dtype=complex)
    for c in range(D):
        for t in range(D):
            t2 = (k if t == j else (j if t == k else t)) if c == c0 else t
            U[c*D+t2, c*D+t] = 1
    return U

def native_unitary(name, params=()):
    if name == 'rz': return rz(params[0])
    if name == 'sx': return sx()
    if name == 'x': return x()
    if name == 'cz': return cz()
    if name in LEVEL_SELECTIVE: return level_selective(name)
    raise KeyError(name)

# ---- logical (qubit) gates accepted in user circuits, as 2x2 / 4x4 matrices on the qubit subspace
def ry2(t): return np.array([[np.cos(t/2), -np.sin(t/2)], [np.sin(t/2), np.cos(t/2)]], complex)
def rx2(t): return np.array([[np.cos(t/2), -1j*np.sin(t/2)], [-1j*np.sin(t/2), np.cos(t/2)]], complex)
def rz2(t): return np.diag([np.exp(-0.5j*t), np.exp(0.5j*t)])
H2 = np.array([[1, 1], [1, -1]], complex)/np.sqrt(2)
LOGICAL_1Q = {
    'h': lambda p: H2, 'x': lambda p: np.array([[0, 1], [1, 0]], complex), 'y': lambda p: np.array([[0, -1j], [1j, 0]]),
    'z': lambda p: np.diag([1, -1]).astype(complex), 's': lambda p: np.diag([1, 1j]), 'sdg': lambda p: np.diag([1, -1j]),
    't': lambda p: np.diag([1, np.exp(1j*np.pi/4)]), 'tdg': lambda p: np.diag([1, np.exp(-1j*np.pi/4)]),
    'sx': lambda p: 0.5*np.array([[1+1j, 1-1j], [1-1j, 1+1j]]), 'rx': lambda p: rx2(p[0]), 'ry': lambda p: ry2(p[0]),
    'rz': lambda p: rz2(p[0]), 'id': lambda p: np.eye(2, dtype=complex)}
