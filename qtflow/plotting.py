# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The qtflow authors
"""Plots: circuit diagrams (logical and native), probability histograms and pulse timelines (matplotlib only)."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

INK, INK2, GRID = '#1a1a1a', '#555555', '#e6e5e1'
COL = {'1q': '#dbe8f6', 'cz': '#2a78d6', 'c1x12': '#eb6834', 'c2x01': '#1baf7a', 'c2x12': '#8a6bd1', 'flux': '#9c9a94'}
LEVEL = {'c1x12': ('1', 'X$_{12}$'), 'c2x01': ('2', 'X$_{01}$'), 'c2x12': ('2', 'X$_{12}$')}
LABEL1Q = {'h': 'H', 'x': 'X', 'y': 'Y', 'z': 'Z', 's': 'S', 'sdg': 'S$^\\dagger$', 't': 'T', 'tdg': 'T$^\\dagger$', 'sx': '$\\sqrt{X}$', 'rx': 'R$_x$', 'ry': 'R$_y$', 'rz': 'R$_z$'}

def _layers(instrs, nwires, wire_of):
    cols = []; front = [0]*nwires
    for ins in instrs:
        ws = [wire_of(q) for q in ins.qubits]; lo, hi = min(ws), max(ws)
        c = max(front[w] for w in range(lo, hi+1)); cols.append(c)
        for w in range(lo, hi+1): front[w] = c + 1
    return cols, max(front) if front else 0

def _angle(a):
    for num, den in ((1, 1), (1, 2), (1, 4), (3, 4), (1, 3), (2, 3)):
        for sg in (1, -1):
            if abs(a - sg*num*np.pi/den) < 1e-9:
                return ('-' if sg < 0 else '') + ('' if num == 1 else str(num)) + '$\\pi$' + ('' if den == 1 else '/%d' % den)
    return '%.2f' % a

def draw(instrs, wires, ax=None, title=None, wire_labels=None, dx=0.95):
    """instrs: list of Instruction (logical or native); wires: list of qubit ids in drawing order (top to bottom)"""
    wi = {q: i for i, q in enumerate(wires)}; nw = len(wires)
    cols, ncol = _layers(instrs, nw, lambda q: wi[q])
    W = max(ncol*dx + 0.6, 2.0)
    if ax is None: _, ax = plt.subplots(figsize=(W*0.9, 0.42*nw + 0.4))
    for i in range(nw):
        ax.plot([0, W - 0.2], [-i, -i], color=INK2, lw=0.8, zorder=0)
        ax.text(-0.08, -i, (wire_labels or ['q$_{%d}$' % q for q in wires])[i], ha='right', va='center', fontsize=7)
    for ins, c in zip(instrs, cols):
        x = 0.35 + c*dx; ws = [wi[q] for q in ins.qubits]; nm = ins.name
        if len(ws) == 1:
            lab = LABEL1Q.get(nm, nm)
            if ins.params and nm in ('rx', 'ry', 'rz'): lab += '\n' + _angle(ins.params[0])
            ax.add_patch(Rectangle((x-0.24, -ws[0]-0.24), 0.48, 0.48, fc=COL['1q'], ec=INK, lw=0.6, zorder=2))
            ax.text(x, -ws[0], lab, ha='center', va='center', fontsize=5.2 if '\n' in lab else 6.5, zorder=3)
            continue
        ax.plot([x, x], [-min(ws), -max(ws)], color=INK, lw=0.8, zorder=1)
        if nm in LEVEL:
            lvl, lab = LEVEL[nm]; cw, tw = ws
            ax.add_patch(Circle((x, -cw), 0.16, fc='white', ec=COL[nm], lw=1.0, zorder=2)); ax.text(x, -cw, lvl, ha='center', va='center', fontsize=6, zorder=3)
            ax.add_patch(Rectangle((x-0.26, -tw-0.22), 0.52, 0.44, fc=COL[nm], ec=INK, lw=0.6, zorder=2)); ax.text(x, -tw, lab, ha='center', va='center', fontsize=6, color='white', zorder=3)
        elif nm == 'cz':
            for w in ws: ax.add_patch(Circle((x, -w), 0.07, fc=INK, zorder=2))
        elif nm == 'swap':
            for w in ws: ax.plot([x-0.1, x+0.1], [-w-0.1, -w+0.1], color=INK, lw=0.9); ax.plot([x-0.1, x+0.1], [-w+0.1, -w-0.1], color=INK, lw=0.9)
        else:                                                  # logical controlled gates: dots + target box
            for w in ws[:-1]: ax.add_patch(Circle((x, -w), 0.07, fc=INK, zorder=2))
            lab = {'cx': 'X', 'ccx': 'X', 'mcx': 'X', 'mcry': 'R$_y$', 'mcrx': 'R$_x$'}.get(nm, nm)
            ax.add_patch(Rectangle((x-0.22, -ws[-1]-0.22), 0.44, 0.44, fc=COL['1q'], ec=INK, lw=0.6, zorder=2)); ax.text(x, -ws[-1], lab, ha='center', va='center', fontsize=6.5, zorder=3)
    ax.set_xlim(-0.55, W); ax.set_ylim(-nw + 0.45, 0.6); ax.set_aspect('equal'); ax.axis('off')
    if title: ax.set_title(title, loc='left', fontsize=8)
    return ax

def draw_logical(qc, ax=None, title=None):
    return draw(qc.data, list(range(qc.num_qubits)), ax, title)

def draw_native(tc, ax=None, title=None, show_flux=False):
    wires = [tc.initial_layout[l] for l in range(tc.logical.num_qubits)]
    wires += [q for q in tc.active_qubits if q not in wires]
    labels = ['Q$_{%d}$' % q for q in wires]
    return draw(tc.native, wires, ax, title, wire_labels=labels)

def plot_histogram(results, ax=None, labels=None, title=None, top=16):
    """results: list of dicts bitstring -> probability (or counts)"""
    if isinstance(results, dict): results = [results]
    norm = [{k: v/sum(r.values()) for k, v in r.items()} for r in results]
    keys = sorted(set().union(*norm), key=lambda k: -max(n.get(k, 0) for n in norm))[:top]; keys = sorted(keys)
    if ax is None: _, ax = plt.subplots(figsize=(3.4, 2.0))
    w = 0.8/len(norm); cols = ['#9c9a94', '#2a78d6', '#eb6834', '#1baf7a']
    for i, n in enumerate(norm):
        ax.bar(np.arange(len(keys)) + (i - (len(norm)-1)/2)*w, [n.get(k, 0) for k in keys], w, color=cols[i % 4],
               label=(labels[i] if labels else None))
    ax.set_xticks(range(len(keys))); ax.set_xticklabels(keys, rotation=60, fontsize=6); ax.set_ylabel('probability', fontsize=7)
    ax.grid(axis='y', color=GRID, lw=0.5); [ax.spines[s].set_visible(False) for s in ('top', 'right')]
    if labels: ax.legend(frameon=False, fontsize=6)
    if title: ax.set_title(title, loc='left', fontsize=8)
    return ax

def plot_pulse_timeline(prog, ax=None, title=None, show_readout=False):
    order = lambda c: (int(''.join(ch for ch in c.split('.')[0].split('_')[0] if ch.isdigit()) or 0), c)
    chans = sorted([c for c in prog.channels if show_readout or not c.endswith('.rr')], key=lambda c: (c[0] == 'c', order(c)))
    ci = {c: i for i, c in enumerate(chans)}
    if ax is None: _, ax = plt.subplots(figsize=(3.4, 0.3*len(chans) + 0.6))
    for e in prog.entries:
        if e['channel'] not in ci: continue
        y0 = -ci[e['channel']]
        if e['kind'] == 'frame':
            ax.plot([e['t0']]*2, [y0-0.35, y0+0.35], color=INK2, lw=0.6, ls=':'); continue
        col = COL.get(e['gate'], '#2a78d6' if e['gate'] in ('sx', 'x') else ('#9c9a94' if e['gate'] in ('flux', 'measure') else '#2a78d6'))
        if e['kind'] == 'measure':
            ax.add_patch(Rectangle((e['t0'], y0-0.3), e['duration'], 0.6, fc='#eeeeee', ec=INK2, lw=0.4)); continue
        if 'I' in e:
            s = np.hypot(prog.waveforms[e['I']], prog.waveforms[e['Q']])
        else:
            s = np.asarray(prog.waveforms[e['wave']])
        m = np.max(np.abs(s)) or 1.0; t = e['t0'] + np.arange(len(s)) + 0.5
        ax.fill_between(t, y0, y0 + 0.4*s/m, color=col, lw=0, alpha=0.85)
        ax.plot(t, y0 + 0.4*s/m, color=col, lw=0.5)
    ax.set_yticks([-i for i in range(len(chans))]); ax.set_yticklabels(chans, fontsize=6)
    ax.set_xlabel('time (ns)', fontsize=7); ax.tick_params(axis='x', labelsize=6)
    ax.set_ylim(-len(chans) + 0.4, 0.7); ax.grid(axis='x', color=GRID, lw=0.5)
    if not show_readout:
        tg = max((e['t0'] + e['duration'] for e in prog.entries if e['kind'] != 'measure'), default=1.0)
        ax.set_xlim(-5, tg + 10)
    [ax.spines[s].set_visible(False) for s in ('top', 'right', 'left')]
    if title: ax.set_title(title, loc='left', fontsize=8)
    return ax
