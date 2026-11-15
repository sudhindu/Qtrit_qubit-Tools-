# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The qtflow authors
"""C^k-NOT (2k-1 pulses), C^k-Ry and C^k-Rx (2k pulses): counts, durations and exactness for k = 2..5."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from qtflow import QCircuit, HeavyHexBackend, transpile
from qtflow.verify import check_equivalence

backend = HeavyHexBackend(3, noise='ideal')
print('%-5s %2s %7s %9s %10s' % ('gate', 'k', 'pulses', 'time(ns)', 'max_err'))
for k in range(2, 6):
    for g in ('mcx', 'mcry', 'mcrx'):
        qc = QCircuit(k + 1)
        if g == 'mcx': qc.mcx(list(range(k)), k)
        else: getattr(qc, g)(np.pi/3, list(range(k)), k)
        tc = transpile(qc, backend)
        print('%-5s %2d %7d %9.0f %10.1e' % (g, k, tc.two_body_pulses(), tc.duration_ns, check_equivalence(qc, tc)['max_error']))
