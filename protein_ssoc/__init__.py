"""
Protein-SSOC: Sequence-Space Occupation Classification
=======================================================

A physics-based protein classification framework using alloy metallurgy analogies.

Modules:
  ssoc_core   — Amino acid tables & composition analysis
  ssoc_phase  — Phase boundary classification (Solvus)
  ssoc_energy — Thermodynamic stability depth (ΔG)

Quick start:
  >>> from protein_ssoc import count_atoms, calc_solvus, calc_dG
  >>> comp = count_atoms("MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG")
  >>> sol = calc_solvus(comp)
  >>> eng = calc_dG(comp)
  >>> print(f"Phase: {sol['pred_cls']}, S={sol['S']:.3f}, ΔG={eng['dG_primary']:.1f} kcal/mol")
"""

from .ssoc_core import count_atoms, AA_GATE, AA_GATE_CORE, AA_CHARGE, AA_HKD
from .ssoc_phase import calc_solvus, assess_metastability
from .ssoc_energy import calc_dG, dG_per_residue, decompose_ala_contribution
from .ssoc_energy import EPSILON_CORE, EPSILON_GATE, EPSILON_SEQ

__version__ = "1.4.0"
__author__ = "Masamichi Iizumi + Tamaki
