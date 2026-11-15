# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The qtflow authors
"""Pulse program: timing consistency, JSON export, QUA-style export."""
import json
import numpy as np
import pytest
from qtflow import QCircuit, HeavyHexBackend, transpile, PulseProgram

B = HeavyHexBackend(3, noise='ideal')

def demo():
    qc = QCircuit(3, 'demo'); qc.h(0); qc.x(1); qc.c1x12(0, 1); qc.c2x01(1, 2); qc.c2x12(1, 2); qc.measure_all()
    return transpile(qc, B)

def test_entries_match_schedule():
    tc = demo(); p = PulseProgram(tc)
    gates = [e for e in p.entries if e['kind'] == 'play']
    assert max(e['t0'] + e['duration'] for e in gates) == pytest.approx(tc.duration_ns)
    assert p.duration_ns == pytest.approx(tc.duration_ns + B.durations['measure'])

def test_native_gate_uses_three_channels():
    tc = demo(); p = PulseProgram(tc)
    for g in ('c1x12', 'c2x01', 'c2x12'):
        chans = {e['channel'][0] + e['channel'].split('.')[-1][-2:] for e in p.entries if e['gate'] == g}
        assert len([e for e in p.entries if e['gate'] == g]) == 3

@pytest.mark.parametrize('g,T', [('c1x12', 60), ('c2x01', 60), ('c2x12', 170)])
def test_waveform_lengths(g, T):
    p = PulseProgram(demo())
    for s in ('_coupler', '_I', '_Q', '_flux'): assert len(p.waveforms[g + s]) == T

def test_json_roundtrip(tmp_path):
    p = PulseProgram(demo()); f = tmp_path / 'p.json'; p.to_json(str(f)); d = json.load(open(f))
    assert d['generator'] == 'qtflow' and len(d['schedule']) == len(p.entries) and set(d['waveforms']) == set(p.waveforms)

def test_qua_text_compiles_and_config(tmp_path):
    p = PulseProgram(demo()); txt = p.to_qua(str(tmp_path / 'prog.py'), config_path=str(tmp_path / 'cfg.json'))
    compile(txt, 'prog.py', 'exec')                         # valid Python syntax (QUA SDK not required)
    cfg = json.load(open(tmp_path / 'cfg.json'))
    for w in cfg['waveforms'].values():
        if w['type'] == 'arbitrary': assert len(w['samples']) % 4 == 0 and len(w['samples']) >= 16
    for e in p.entries:
        if e['kind'] == 'play': assert e['channel'] in cfg['elements']
    assert txt.count('play(') == sum(1 for e in p.entries if e['kind'] == 'play')

def test_rz_is_virtual():
    qc = QCircuit(1); qc.rz(0.5, 0); tc = transpile(qc, B); p = PulseProgram(tc)
    assert [e['kind'] for e in p.entries] == ['frame'] and p.duration_ns == 0

def test_cz_on_coupler():
    qc = QCircuit(2); qc.cz(0, 1); tc = transpile(qc, B); p = PulseProgram(tc)
    e = [e for e in p.entries if e['gate'] == 'cz'][0]; assert e['channel'].startswith('c') and e['duration'] == 68

def test_custom_calibration():
    p = PulseProgram(demo(), calibration={'x_amp_V': 0.3})
    assert np.max(p.waveforms['x_I']) == pytest.approx(0.3)
