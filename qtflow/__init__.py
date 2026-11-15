# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The qtflow authors
"""qtflow — a lab-version tool flow for transmon qubit-qutrit circuits.

QCircuit (Qiskit-like) -> transpile (IBM native gates + level-selective qutrit gates, heavy-hex routing, schedule)
-> run (noisy qutrit simulation, histogram) -> PulseProgram (JSON / QUA-style pulse instructions).
"""
from .circuit import QCircuit, Instruction
from .backend import HeavyHexBackend, NoiseModel
from .transpiler import transpile, TranspiledCircuit
from .simulator import run, statevector, ideal_probabilities
from .pulses import PulseProgram
from . import plotting

__version__ = '0.1.0'
__all__ = ['QCircuit', 'Instruction', 'HeavyHexBackend', 'NoiseModel', 'transpile', 'TranspiledCircuit',
           'run', 'statevector', 'ideal_probabilities', 'PulseProgram', 'plotting']
