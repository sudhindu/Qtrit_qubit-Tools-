# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The qtflow authors
"""Pulse-level output: scheduled native circuit -> pulse program.

Channels (per physical qutrit p and coupler a-b):
    q{p}.xy   microwave drive (I/Q)            sx, x, and the drive of the level-selective gates
    q{p}.z    flux line of the qutrit           role moves (control <-> target) and the target flux pulse
    c{a}_{b}  flux line of the tunable coupler  cz and the coupler pulse of the level-selective gates
    q{p}.rr   readout resonator
rz is a virtual frame change (zero duration) on q{p}.xy.

Waveforms of c1x12 / c2x01 / c2x12 are the optimized single-window pulses shipped in data/native_qutrit_pulses.json
(MHz units, 1 ns samples). sx/x (DRAG Gaussian), cz (flat-top coupler pulse), flux steps and readout are generic
shapes with PLACEHOLDER amplitudes: they must be calibrated on a device. Conversion MHz -> volts uses the
placeholder factors in CALIBRATION.

Outputs: PulseProgram.to_json(), PulseProgram.to_qua() (QUA-style Python text for Quantum Machines controllers;
qtflow is not affiliated with Quantum Machines and the text is not hardware-tested), and plotting.plot_pulse_timeline().
"""
import json, os
import numpy as np

_DATA = os.path.join(os.path.dirname(__file__), 'data', 'native_qutrit_pulses.json')
CALIBRATION = {'V_per_MHz_drive': 0.0045, 'V_per_MHz_coupler': 0.004, 'V_per_MHz_flux': 0.004,
               'x_amp_V': 0.20, 'sx_amp_V': 0.10, 'drag_beta': 0.5, 'cz_amp_V': 0.25, 'flux_step_V': 0.10,
               'readout_amp_V': 0.05, 'lo_GHz': 5.0, 'readout_IF_MHz': 50.0}

def _gauss_drag(T, amp, beta):
    t = np.arange(T) + 0.5; s = T/4.0
    g = np.exp(-0.5*((t - T/2)/s)**2); g -= g.min(); g /= g.max()
    return amp*g, -beta*amp*np.gradient(g)

def _flat_top(T, amp, ramp=10):
    t = np.arange(T) + 0.5; w = np.ones(T)
    r = t < ramp; w[r] = np.sin(0.5*np.pi*t[r]/ramp)**2
    r = t > T-ramp; w[r] = np.sin(0.5*np.pi*(T-t[r])/ramp)**2
    return amp*w

class PulseProgram:
    def __init__(self, tc, calibration=None):
        self.cal = dict(CALIBRATION); self.cal.update(calibration or {})
        self.native = json.load(open(_DATA))['gates']
        self.tc = tc; self.entries = []; self.waveforms = {}
        self._build()

    def _wf(self, name, samples):
        self.waveforms[name] = [float(v) for v in np.asarray(samples)]; return name

    def _build(self):
        c = self.cal
        xi, xq = _gauss_drag(32, c['x_amp_V'], c['drag_beta']); si, sq = _gauss_drag(32, c['sx_amp_V'], c['drag_beta'])
        self._wf('x_I', xi); self._wf('x_Q', xq); self._wf('sx_I', si); self._wf('sx_Q', sq)
        self._wf('cz', _flat_top(int(self.tc.backend.durations['cz']), c['cz_amp_V']))
        nflux = int(self.tc.backend.durations['flux'])
        self._wf('flux_up', np.full(nflux, c['flux_step_V'])); self._wf('flux_down', np.full(nflux, -c['flux_step_V']))
        self._wf('readout', np.full(int(self.tc.backend.durations['measure']), c['readout_amp_V']))
        for g, d in self.native.items():
            self._wf(g + '_coupler', np.array(d['coupler_MHz'])*c['V_per_MHz_coupler'])
            self._wf(g + '_I', np.array(d['drive_I_MHz'])*c['V_per_MHz_drive'])
            self._wf(g + '_Q', np.array(d['drive_Q_MHz'])*c['V_per_MHz_drive'])
            self._wf(g + '_flux', np.array(d['target_flux_MHz'])*c['V_per_MHz_flux'])
        for x in self.tc.timed:
            q = x.qubits; t0 = x.t0
            if x.name == 'rz':
                self.entries.append(dict(t0=t0, duration=0.0, channel='q%d.xy' % q[0], kind='frame', angle=float(x.params[0]), gate='rz'))
            elif x.name in ('sx', 'x'):
                self.entries.append(dict(t0=t0, duration=x.duration, channel='q%d.xy' % q[0], kind='play', I=x.name + '_I', Q=x.name + '_Q', gate=x.name))
            elif x.name == 'cz':
                a, b = sorted(q); self.entries.append(dict(t0=t0, duration=x.duration, channel='c%d_%d' % (a, b), kind='play', wave='cz', gate='cz'))
            elif x.name == 'flux':
                self.entries.append(dict(t0=t0, duration=x.duration, channel='q%d.z' % q[0], kind='play', wave='flux_up' if x.params[0] > 0 else 'flux_down', gate='flux'))
            elif x.name in self.native:
                ctl, tgt = q; a, b = sorted(q); d = self.native[x.name]
                ifreq = (d['drive_GHz'] - self.cal['lo_GHz'])*1e3
                self.entries.append(dict(t0=t0, duration=x.duration, channel='c%d_%d' % (a, b), kind='play', wave=x.name + '_coupler', gate=x.name))
                self.entries.append(dict(t0=t0, duration=x.duration, channel='q%d.xy' % tgt, kind='play', I=x.name + '_I', Q=x.name + '_Q', gate=x.name, IF_MHz=ifreq))
                self.entries.append(dict(t0=t0, duration=x.duration, channel='q%d.z' % tgt, kind='play', wave=x.name + '_flux', gate=x.name))
            elif x.name == 'measure':
                self.entries.append(dict(t0=t0, duration=x.duration, channel='q%d.rr' % q[0], kind='measure', wave='readout', gate='measure'))
        self.entries.sort(key=lambda e: (e['t0'], e['channel']))

    @property
    def channels(self):
        return sorted({e['channel'] for e in self.entries}, key=lambda s: (s[0] != 'q', s))

    @property
    def duration_ns(self):
        return max((e['t0'] + e['duration'] for e in self.entries), default=0.0)

    def to_json(self, path=None):
        obj = dict(generator='qtflow', units=dict(time='ns', amplitude='V (placeholder calibration)'), calibration=self.cal,
                   layout=self.tc.final_layout, channels=self.channels, schedule=self.entries, waveforms=self.waveforms)
        s = json.dumps(obj, indent=1)
        if path: open(path, 'w').write(s)
        return s

    # ---------------------------------------------------------------- QUA-style export
    @staticmethod
    def _pad(samples):
        n = max(16, int(np.ceil(len(samples)/4.0))*4); return list(samples) + [0.0]*(n - len(samples))

    def qua_config(self):
        els, pulses, wfs = {}, {}, {}
        for name, s in self.waveforms.items(): wfs[name] = {'type': 'arbitrary', 'samples': self._pad(s)}
        wfs['zero'] = {'type': 'constant', 'sample': 0.0}
        def ensure(ch):
            if ch in els: return
            if ch.endswith('.xy'):
                els[ch] = {'mixInputs': {'I': ('con1', 1), 'Q': ('con1', 2), 'lo_frequency': int(self.cal['lo_GHz']*1e9)},
                           'intermediate_frequency': 0, 'operations': {}}
            elif ch.endswith('.rr'):
                els[ch] = {'mixInputs': {'I': ('con1', 9), 'Q': ('con1', 10), 'lo_frequency': int(7.0e9)},
                           'intermediate_frequency': int(self.cal['readout_IF_MHz']*1e6), 'operations': {},
                           'outputs': {'out1': ('con1', 1)}, 'time_of_flight': 200, 'smearing': 0}
            else:
                els[ch] = {'singleInput': {'port': ('con1', 3)}, 'operations': {}}
        for e in self.entries:
            ch = e['channel']; ensure(ch)
            if e['kind'] == 'frame': continue
            op = e['gate'] if e['kind'] != 'measure' else 'readout'
            if 'I' in e:
                pname = '%s_iq' % e['gate']
                pulses[pname] = {'operation': 'control', 'length': len(self._pad(self.waveforms[e['I']])), 'waveforms': {'I': e['I'], 'Q': e['Q']}}
            elif e['kind'] == 'measure':
                pname = 'readout_pulse'
                pulses[pname] = {'operation': 'measurement', 'length': len(self._pad(self.waveforms['readout'])),
                                 'waveforms': {'I': 'readout', 'Q': 'zero'}, 'digital_marker': 'ON'}
            else:
                pname = '%s_single' % e['wave']; op = e['wave']
                pulses[pname] = {'operation': 'control', 'length': len(self._pad(self.waveforms[e['wave']])), 'waveforms': {'single': e['wave']}}
            els[ch]['operations'][op] = pname
        return {'version': 1, 'controllers': {'con1': {'analog_outputs': {i: {'offset': 0.0} for i in range(1, 11)},
                                                       'digital_outputs': {1: {}}, 'analog_inputs': {1: {'offset': 0.0}}}},
                'elements': els, 'pulses': pulses, 'waveforms': wfs, 'digital_waveforms': {'ON': {'samples': [(1, 0)]}}}

    def to_qua(self, path=None, config_path=None):
        """QUA-style program text; instructions grouped by start time, align() keeps the ASAP schedule"""
        L = ['# Auto-generated by qtflow: QUA-style program for the transpiled circuit "%s".' % self.tc.logical.name,
             '# Not affiliated with Quantum Machines; not hardware-tested. Amplitudes are placeholder calibrations.',
             '# Waveforms are zero-padded to multiples of 4 ns (min 16 ns) as required by the controller clock.',
             'import json', 'from qm.qua import *', '',
             "config = json.load(open(%r))" % (config_path or 'qtflow_qua_config.json'), '',
             'with program() as prog:']
        rr = [c for c in self.channels if c.endswith('.rr')]
        if rr: L.append('    I = declare(fixed)')
        groups = {}
        for e in self.entries: groups.setdefault(round(e['t0'], 3), []).append(e)
        for t0 in sorted(groups):
            es = groups[t0]; chans = sorted({e['channel'] for e in es if e['kind'] != 'frame'})
            L.append('    # t = %.1f ns' % t0)
            if len(chans) > 1: L.append('    align(%s)' % ', '.join(repr(c) for c in chans))
            for e in es:
                if e['kind'] == 'frame':
                    L.append('    frame_rotation_2pi(%.6f, %r)  # rz(%.4f)' % (-e['angle']/(2*np.pi), e['channel'], e['angle']))
                elif e['kind'] == 'measure':
                    L.append('    measure("readout", %r, None, demod.full("cos", I))' % e['channel'])
                else:
                    if 'IF_MHz' in e: L.append('    update_frequency(%r, %d)' % (e['channel'], int(round(e['IF_MHz']*1e6))))
                    op = e['gate'] if 'I' in e else e['wave']
                    L.append('    play(%r, %r)  # %s' % (op, e['channel'], e['gate']))
        txt = '\n'.join(L) + '\n'
        if path: open(path, 'w').write(txt)
        if config_path: json.dump(self.qua_config(), open(config_path, 'w'))
        return txt
