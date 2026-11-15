# qtflow: a qubit-qutrit tool flow for heavy-hex transmon devices (lab version 0.1.0)

qtflow takes a circuit written in a Qiskit-like Python API and produces three outputs:

1. a **native circuit** routed on a heavy-hex lattice;
2. an **outcome histogram** from a noisy qutrit simulator, in the style of Aer;
3. a **pulse program**, exported as JSON and as QUA-style Python text.

The native gate set is the IBM set {`rz`, `sx`, `x`, `cz`} plus three level-selective two-qutrit gates:

| gate | action (control, target) | default duration |
|---|---|---|
| `c1x12` = C¹X₁₂ | if control is \|1⟩: target \|1⟩ ↔ \|2⟩ | 60 ns |
| `c2x01` = C²X₀₁ | if control is \|2⟩: target \|0⟩ ↔ \|1⟩ | 60 ns |
| `c2x12` = C²X₁₂ | if control is \|2⟩: target \|1⟩ ↔ \|2⟩ | 170 ns |

Multi-controlled gates are compiled with these three gates on a line of qutrits and need **no ancilla**:

| logical gate | native sequence | two-body pulses |
|---|---|---|
| Toffoli `ccx` | c1x12 · c2x01 · c1x12 | 3 |
| C^k-NOT `mcx` | fold (c1x12, c2x12 …) · c2x01 · unfold | 2k − 1 |
| C^k-Ry `mcry` | ry(θ/2) · fold · c2x01 · ry(−θ/2) · c2x01 · unfold | 2k |
| C^k-Rx `mcrx` | rz(π/2) · C^k-Ry(θ) · rz(−π/2) | 2k |

All other gates are compiled to IBM natives: `cx` = H·CZ·H, `swap` = 3 `cx`, and single-qubit gates go to `rz`/`sx` through a ZSX Euler decomposition.

## Install

```bash
git clone <this repository> qtflow && cd qtflow
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt                        # numpy, scipy, networkx, matplotlib, qiskit
pip install -e .                                       # optional: makes `import qtflow` work from anywhere
```

Tested with Python 3.13, Qiskit 2.6.2, numpy 2.5 and scipy 1.18. Python ≥ 3.9 and Qiskit ≥ 1.2 should work.

## Quick start

```python
from qtflow import QCircuit, HeavyHexBackend, transpile, run, PulseProgram, plotting

qc = QCircuit(3)
qc.h(0); qc.x(1)
qc.c1x12(0, 1); qc.c2x01(1, 2); qc.c2x12(1, 2)      # native qutrit gates can be used directly
qc.measure_all()

backend = HeavyHexBackend(distance=3, noise='reference')
tc = transpile(qc, backend)                          # decompose, route, schedule
print(tc.summary())
res = run(tc, shots=4000, trajectories=300)          # noisy Monte-Carlo trajectories
print(res['counts'])                                 # e.g. {'010': ~1900, '221': ~1900, ...}
prog = PulseProgram(tc)
prog.to_json('pulses.json')
prog.to_qua('program_qua.py', config_path='qua_config.json')
plotting.draw_native(tc); plotting.plot_histogram(res['counts']); plotting.plot_pulse_timeline(prog)
```

The logical gates you can use are `h x y z s sdg t tdg sx rx ry rz id cx cz swap ccx/toffoli mcx mcry mcrx c1x12 c2x01 c2x12`. `QCircuit.from_qiskit()` imports a Qiskit `QuantumCircuit` built from these gates.

## How it works

```
QCircuit ──► layout + routing ──► decomposition ──► rz merge ──► ASAP schedule ──► simulator ──► histogram
             (heavy-hex)          (IBM natives +                 (+ flux moves)   (qutrits)
                                   c1x12/c2x01/c2x12)                         └──► PulseProgram ──► JSON / QUA-style
```

| module | role |
|---|---|
| `circuit.py` | `QCircuit` user API |
| `gates.py` | 3-level matrices of the native gates |
| `backend.py` | `HeavyHexBackend`: Qiskit `CouplingMap.from_heavy_hex(d)`, durations, `NoiseModel` presets |
| `routing.py` | initial layout on a long path. Two-qubit gates are routed with SWAPs. A C^k gate is routed by choosing a line of k+1 qutrits (Hungarian assignment) and moving the operands onto it |
| `decompose.py` | native decompositions (tables above) |
| `schedule.py` | ASAP schedule. Inserts a 10 ns flux move when a qutrit changes between the control and target roles |
| `simulator.py` | exact statevector; noisy trajectories with 3-level T1/T2 Kraus channels and qutrit-depolarizing gate errors |
| `pulses.py` | pulse program with channels `q{p}.xy`, `q{p}.z`, `c{a}_{b}` (coupler) and `q{p}.rr`. The waveforms of the qutrit gates are in `data/native_qutrit_pulses.json` |
| `verify.py` | equivalence check of the native circuit against Qiskit `Operator` (qubit circuits) or the qutrit unitary |
| `plotting.py` | circuit drawings, histogram and pulse timeline |

**Conventions.**

- Bitstrings follow Qiskit: qubit 0 is the rightmost character.
- A qutrit read in |2⟩ appears as the digit `2`, so leakage is visible in the histogram.
- `rz` is a virtual frame change.
- Native gates are written (control, target).

## Reproduce

```bash
python -m pytest -q tests                      # 179 unit tests, about 1 min
python validation/run_validation.py            # V1-V6 validation, writes validation/REPORT.md
cd examples/output && python ../demo_3q.py     # 4-panel demo figure (input, native, histogram, pulses)
# every example writes its outputs into the folder you run it from (here: examples/output/)
```

The tests and the validation cover:

- gate definitions;
- exactness of every decomposition against Qiskit, for k = 2…6 and for several angles;
- the pulse counts 2k−1 and 2k, and the C^k-NOT duration 360k − 520 ns;
- equivalence and coupler adjacency for random routed circuits;
- conservation and trace preservation of the noise channels;
- the trend of noisy success with T1;
- pulse timing, the JSON round trip and the syntax of the QUA-style text;
- end-to-end runs: the Toffoli truth table and 3-qubit Grover.

`validation/REPORT.md` also lists the same gates compiled by Qiskit to IBM natives on the same lattice, for reference.

| folder | content |
|---|---|
| `qtflow/` | package |
| `tests/` | pytest unit tests |
| `validation/` | `run_validation.py`, `REPORT.md`, `results/` (CSV, JSON, figure), `logs/` |
| `examples/` | `demo_3q.py`, `toffoli.py`, `grover3.py`, `multicontrol.py`, `from_qiskit.py`; outputs are in `examples/output/` |

## Model assumptions (defaults, all overridable)

- Durations: sx/x 32 ns, cz 68 ns, c1x12 60 ns, c2x01 60 ns, c2x12 170 ns, flux move 10 ns, readout 1 µs.
- Noise presets:
  - `reference`: T1(1→0) 27 µs, T1(2→1) 17 µs, Tφ 40 µs;
  - `improved`: 100/60/150 µs;
  - `ibm_like`: 200/100/200 µs;
  - `ideal`.
- Gate errors are qutrit-depolarizing probabilities per gate: sx/x 2.4e-4, cz 2e-3, c1x12 6e-3, c2x01 3e-3, c2x12 1.8e-2. They are placeholders and should be replaced by device data.
- Readout error is off by default.
- Pulse amplitudes in volts use the placeholder factors in `pulses.CALIBRATION`.

## Limitations (lab version)

- **Routing is greedy.** It does not look ahead and does not use noise-aware layout. SWAPs act on the qubit subspace, so no qutrit may hold |2⟩ when a SWAP is inserted; the router warns when this could happen.
- **The simulator is dense.** It works up to about 12 active qutrits. There is no tensor-network or GPU back-end.
- **The noise model has no crosstalk or spectator coupling.** It also has no time-correlated (1/f) noise or calibration drift.
- **The pulse output is not hardware-tested.** It has not been run on hardware or on the Quantum Machines SDK; only its syntax is checked. It needs per-device calibration: amplitudes, IF/LO plan, port mapping and readout integration weights.
- **Single-qubit gates are not optimized.** Only consecutive `rz` are merged. There is no commutation-based or peephole optimization.
- **No OpenQASM 3 / OpenPulse export.** There is also no mid-circuit measurement or classical feed-forward.

## License

Apache License 2.0; see `LICENSE` and `NOTICE`. Copyright 2026 The qtflow authors. Author names are withheld for double-blind review. qtflow is independent of IBM and Quantum Machines; product names are trademarks of their owners.
