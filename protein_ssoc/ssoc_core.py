#!/usr/bin/env python3
"""
================================================================================
Protein-SSOC Core — Amino Acid Tables & Composition Analysis
================================================================================

Shared foundation for Phase and Energy modules.

Alloy analogy:
  Amino acids → solute atoms in a metallic alloy
  Sequence    → atomic arrangement
  Composition → alloy composition (binary: hydrophobic/polar)

Tables:
  AA_GATE      — Binary gate classification (H=1, P=0)
  AA_GATE_CORE — Core gate (Ala excluded: packing defect dopant)
  AA_CHARGE    — Formal charges at pH 7
  AA_HKD       — Kyte-Doolittle hydropathy (modified)

Author: Masamichi Iizumi + Tamaki
License: MIT
================================================================================
"""

from __future__ import annotations
import numpy as np
from collections import Counter


# ==============================================================================
# Amino Acid Classification Tables
# ==============================================================================

# Binary gate: hydrophobic (GATE) = 1, polar (NonGATE) = 0
AA_GATE = {
    'I': 1, 'V': 1, 'L': 1, 'F': 1, 'C': 1, 'M': 1, 'A': 1, 'W': 1, 'Y': 1,
    'G': 0, 'T': 0, 'S': 0, 'P': 0,
    'H': 0, 'E': 0, 'Q': 0, 'D': 0, 'N': 0, 'K': 0, 'R': 0,
}

# Core gate: Ala excluded (packing defect dopant, ε_Ala ≲ 0)
AA_GATE_CORE = {
    'I': 1, 'V': 1, 'L': 1, 'F': 1, 'C': 1, 'M': 1, 'W': 1, 'Y': 1,
    'A': 0,
    'G': 0, 'T': 0, 'S': 0, 'P': 0,
    'H': 0, 'E': 0, 'Q': 0, 'D': 0, 'N': 0, 'K': 0, 'R': 0,
}

# Formal charges at neutral pH
AA_CHARGE = {'D': -1, 'E': -1, 'K': +1, 'R': +1}

# Modified Kyte-Doolittle hydropathy scale
AA_HKD = {
    'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5, 'M': 1.9, 'A': 1.8,
    'G': -0.4, 'T': -0.7, 'S': -0.8, 'W': -0.9, 'Y': -1.3, 'P': -1.6,
    'H': -3.2, 'E': -3.5, 'Q': -3.5, 'D': -3.5, 'N': -3.5, 'K': -3.9, 'R': -4.5,
}

# Hydrophobicity threshold for cluster detection (W included at -0.9)
HYDRO_THRESHOLD = -1.0


# ==============================================================================
# Composition Analysis (= "Atom Counting" in alloy analogy)
# ==============================================================================

def max_window_ncpr(q: np.ndarray, w: int = 10) -> float:
    """Peak local net charge per residue in sliding window."""
    L = len(q)
    if L < w:
        return abs(float(q.sum())) / max(L, 1)
    best = 0.0
    for i in range(L - w + 1):
        best = max(best, abs(float(q[i:i+w].sum())) / w)
    return best


def count_atoms(seq: str) -> dict:
    """
    Analyze sequence composition for phase and energy calculations.
    
    Returns dict with all composition features needed by both
    ssoc_phase (solvus) and ssoc_energy (depth).
    
    Alloy analogy:
      N_H      = total hydrophobic "solute" count (gate_full, Ala included)
      N_H_core = core hydrophobic count (Ala excluded)
      clusters = precipitate nuclei (hydrophobic patches)
      Cq1      = Warren-Cowley SRO parameter for charge
    """
    residues = [c for c in seq.upper() if c in AA_GATE]
    L = len(residues)
    if L == 0:
        return {'L': 0}

    # --- Composition counts ---
    N_H = sum(1 for c in residues if AA_GATE.get(c, 0) == 1)
    N_H_core = sum(1 for c in residues if AA_GATE_CORE.get(c, 0) == 1)
    N_Ala = sum(1 for c in residues if c == 'A')
    N_pos = sum(1 for c in residues if AA_CHARGE.get(c, 0) > 0)
    N_neg = sum(1 for c in residues if AA_CHARGE.get(c, 0) < 0)
    N_Pro = sum(1 for c in residues if c == 'P')
    N_Gly = sum(1 for c in residues if c == 'G')
    N_W = sum(1 for c in residues if c == 'W')
    N_Y = sum(1 for c in residues if c == 'Y')

    net_charge = N_pos - N_neg
    ncpr = abs(net_charge) / L
    f_H = N_H / L
    f_Pro = N_Pro / L
    f_Gly = N_Gly / L

    hkd_values = [AA_HKD.get(c, 0) for c in residues]
    mean_H = np.mean(hkd_values)
    H_norm = (mean_H + 4.5) / 9.0

    counts = Counter(residues)
    freqs = np.array(list(counts.values()), dtype=float) / L
    shannon = -np.sum(freqs * np.log(freqs + 1e-10))

    # --- SRO-1: Warren-Cowley charge correlation C_q(1) ---
    q = np.array([AA_CHARGE.get(c, 0) for c in residues], dtype=float)
    Cq1 = float(np.mean(q[:-1] * q[1:])) if L >= 2 else 0.0

    # --- SRO-2: Hydrophobic cluster geometry ---
    h = np.array(hkd_values)
    clusters = []
    in_c, start = False, 0
    for i in range(L):
        if h[i] >= HYDRO_THRESHOLD:
            if not in_c:
                start = i
                in_c = True
        else:
            if in_c:
                clen = i - start
                if clen >= 2:
                    clusters.append(clen)
            in_c = False
    if in_c:
        clen = L - start
        if clen >= 2:
            clusters.append(clen)

    n_clusters = len(clusters)
    largest_cluster = max(clusters) if clusters else 0
    largest_cluster_frac = largest_cluster / L
    cluster_coverage = sum(clusters) / L

    # --- Rigidity (Pro constraint vs Gly freedom) ---
    stiffness = f_Pro - 0.7 * f_Gly

    # --- Charge segregation (grain boundary detection) ---
    n_regions = 3
    region_size = L // n_regions
    regional_ncprs = []
    for r in range(n_regions):
        s = r * region_size
        e = s + region_size if r < n_regions - 1 else L
        r_ncpr = abs(float(q[s:e].sum())) / (e - s)
        regional_ncprs.append(r_ncpr)
    charge_asym = max(regional_ncprs) - ncpr

    # --- Charge patch (surface patch detection) ---
    local_peak = max_window_ncpr(q, w=10)
    charge_patch = max(local_peak - ncpr, 0.0)

    return {
        'L': L, 'N_H': N_H, 'N_H_core': N_H_core, 'N_Ala': N_Ala,
        'N_pos': N_pos, 'N_neg': N_neg,
        'N_Pro': N_Pro, 'N_Gly': N_Gly, 'N_W': N_W, 'N_Y': N_Y,
        'net_charge': net_charge, 'ncpr': ncpr,
        'f_H': f_H, 'f_Pro': f_Pro, 'f_Gly': f_Gly,
        'mean_H': mean_H, 'H_norm': H_norm, 'shannon': shannon,
        'Cq1': float(Cq1),
        'n_clusters': n_clusters, 'largest_cluster': largest_cluster,
        'largest_cluster_frac': float(largest_cluster_frac),
        'cluster_coverage': float(cluster_coverage),
        'stiffness': float(stiffness),
        'charge_asym': float(charge_asym),
        'regional_ncprs': regional_ncprs,
        'local_peak_ncpr': float(local_peak),
        'charge_patch': float(charge_patch),
    }
