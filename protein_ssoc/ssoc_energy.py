#!/usr/bin/env python3
"""
================================================================================
Protein-SSOC Energy — Thermodynamic Stability Depth
================================================================================

"How stable is it?" — Free energy depth from hydrophobic core count.

Model:
  ΔG = −ε × N_H(core)

  Primary (Ala excluded):  ε_core = 0.274 kcal/mol  (MAE=0.68, r=0.980)
  Reference (Ala included): ε_gate = 0.213 kcal/mol  (MAE=0.91, r=0.975)

Physical decomposition:
  ε = γ_eff × A_eff × η
    γ_eff ≈ 16 cal/mol/Å²   (Eisenberg-McLachlan solvation parameter)
    A_eff ≈ 148 Å²           (mean buried nonpolar SASA, core residues)
    η     ≈ 0.12             (effective packing efficiency)

Key insight — Alanine as packing defect:
  ε_Ala = −0.021 kcal/mol (negative = destabilizing)
  Equivalent to undersized solute creating lattice vacancies in alloys.
  When Ala included: ε_gate ≈ kT/(2√2) at T=300K (0.9% match)

Independence principle:
  Phase diagram (solvus S) ≠ Phase energy (ΔG)
  "Will it fold?" and "How stable?" are separate questions.

v1.5 bridge insight:
  ΔG ≈ ε × N_H(core) + measurement_noise (~1 kcal/mol)
  Core packing (GG branch) is condition-independent.
  Surface-involving mutations (GN, NN) couple to T/pH/method.

Author: Masamichi Iizumi + Tamaki
License: MIT
================================================================================
"""

from __future__ import annotations


# ==============================================================================
# Energy Model Constants
# ==============================================================================

# Primary: ΔG = −ε_core × N_H(core) [Ala excluded]
EPSILON_CORE = 0.274    # kcal/mol (LOO CV=1.7%, MAE=0.68, r=0.980)

# Reference: ΔG = −ε_gate × N_H(gate_full) [Ala included]
EPSILON_GATE = 0.213    # kcal/mol (LOO CV=1.4%, MAE=0.91, r=0.975)
# Remark: ε_gate × 2√2 / kT(300K) = 1.009 ≈ 1.0

# v1.6 sequence-calibrated epsilon
EPSILON_SEQ = 0.210     # kcal/mol (calibrated against larger dataset)

# Alanine contribution (implied from core vs gate difference)
EPSILON_ALA = -0.021    # kcal/mol (negative = destabilizing = packing defect)


# ==============================================================================
# Energy Calculation
# ==============================================================================

def calc_dG(comp: dict, model: str = 'core') -> dict:
    """
    Calculate thermodynamic stability depth.
    
    Parameters
    ----------
    comp : dict
        Output from ssoc_core.count_atoms()
    model : str
        'core'     — Primary model, Ala excluded (default)
        'gate'     — Reference model, Ala included
        'seq'      — v1.6 sequence-calibrated
        'all'      — Return all three
    
    Returns
    -------
    dict with:
        dG_primary  — ΔG from selected model (kcal/mol, negative = stable)
        N_H_core    — Core hydrophobic count
        N_H_gate    — Full gate hydrophobic count
        N_Ala       — Alanine count
        epsilon     — ε value used
        model       — Model name
        + additional fields for model='all'
    """
    L = comp['L']
    if L == 0:
        return {'dG_primary': 0.0, 'N_H_core': 0, 'N_H_gate': 0,
                'N_Ala': 0, 'epsilon': 0.0, 'model': model}

    N_H_core = comp['N_H_core']
    N_H_gate = comp['N_H']
    N_Ala = comp['N_Ala']

    # Calculate all models
    dG_core = -EPSILON_CORE * N_H_core
    dG_gate = -EPSILON_GATE * N_H_gate
    dG_seq = -EPSILON_SEQ * N_H_gate

    # Select primary
    if model == 'core':
        dG_primary = dG_core
        epsilon = EPSILON_CORE
    elif model == 'gate':
        dG_primary = dG_gate
        epsilon = EPSILON_GATE
    elif model == 'seq':
        dG_primary = dG_seq
        epsilon = EPSILON_SEQ
    elif model == 'all':
        return {
            'dG_core': float(dG_core),
            'dG_gate': float(dG_gate),
            'dG_seq': float(dG_seq),
            'N_H_core': N_H_core,
            'N_H_gate': N_H_gate,
            'N_Ala': N_Ala,
            'epsilon_core': EPSILON_CORE,
            'epsilon_gate': EPSILON_GATE,
            'epsilon_seq': EPSILON_SEQ,
            'model': 'all',
        }
    else:
        raise ValueError(f"Unknown model: {model}. Use 'core', 'gate', 'seq', or 'all'")

    return {
        'dG_primary': float(dG_primary),
        'N_H_core': N_H_core,
        'N_H_gate': N_H_gate,
        'N_Ala': N_Ala,
        'epsilon': epsilon,
        'model': model,
    }


def dG_per_residue(comp: dict, model: str = 'core') -> float:
    """
    Normalized stability: ΔG / L (kcal/mol per residue).
    
    Useful for comparing proteins of different sizes.
    Typical range: −0.05 to −0.15 kcal/mol/residue for folded proteins.
    """
    result = calc_dG(comp, model=model)
    L = comp['L']
    if L == 0:
        return 0.0
    return result['dG_primary'] / L


def decompose_ala_contribution(comp: dict) -> dict:
    """
    Decompose Alanine's destabilizing contribution.
    
    Shows how much stability is "lost" by Ala acting as packing defect.
    ΔΔG_Ala = N_Ala × (ε_gate − ε_core_effective)
    """
    N_Ala = comp.get('N_Ala', 0)
    N_H_core = comp.get('N_H_core', 0)
    N_H_gate = comp.get('N_H', 0)

    dG_with_ala = -EPSILON_GATE * N_H_gate
    dG_without_ala = -EPSILON_CORE * N_H_core

    # Implied Ala contribution
    ddG_ala = dG_with_ala - dG_without_ala
    per_ala = ddG_ala / N_Ala if N_Ala > 0 else 0.0

    return {
        'N_Ala': N_Ala,
        'ddG_ala_total': float(ddG_ala),
        'ddG_per_ala': float(per_ala),
        'dG_with_ala': float(dG_with_ala),
        'dG_without_ala': float(dG_without_ala),
    }
