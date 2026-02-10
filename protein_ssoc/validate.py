#!/usr/bin/env python3
"""
================================================================================
Protein-SSOC Validation Script
================================================================================

Validates Phase (Solvus) and Energy (ΔG) models against external data.
Data is loaded from JSON files — nothing is hardcoded.

Usage:
  python validate.py                         # Use bundled data/proteins.json
  python validate.py --data my_proteins.json  # Use custom dataset

Data format (JSON):
  {
    "proteins": {
      "name": {
        "seq": "AMINOACIDSEQUENCE...",
        "dG": -7.2,          // kcal/mol (0.0 for IDP)
        "cls": "folded"       // "folded" or "idp"
      }
    }
  }

Author: Masamichi Iizumi + Tamaki + Gemini + GPT
License: MIT
================================================================================
"""

from __future__ import annotations
import json
import argparse
import sys
from pathlib import Path

import numpy as np

# Import from package
from ssoc_core import count_atoms
from ssoc_phase import calc_solvus, assess_metastability
from ssoc_energy import calc_dG, decompose_ala_contribution, EPSILON_CORE


def load_data(path: str | Path) -> dict:
    """Load protein dataset from JSON file."""
    path = Path(path)
    if not path.exists():
        print(f"Error: {path} not found")
        sys.exit(1)
    with open(path) as f:
        data = json.load(f)
    return data['proteins']


def validate(data: dict, verbose: bool = True) -> dict:
    """
    Run full Phase + Energy validation.
    
    Returns dict with summary statistics.
    """
    all_comp = {}
    all_sol = {}
    all_eng = {}

    # ============================================================
    # Phase 1: Composition analysis
    # ============================================================
    if verbose:
        print("=" * 90)
        print("  PHASE 1: Composition Analysis")
        print("=" * 90)
        print(f"  {'Protein':<15} {'L':>4} {'N_H':>4} {'N_Hc':>5} {'f_H':>6} "
              f"{'NCPR':>6} {'Cq1':>7} {'LCF':>6} {'Stiff':>7}")
        print("-" * 90)

    for key, prot in data.items():
        comp = count_atoms(prot['seq'])
        all_comp[key] = comp
        if verbose and comp['L'] > 0:
            print(f"  {key:<15} {comp['L']:>4d} {comp['N_H']:>4d} "
                  f"{comp['N_H_core']:>5d} {comp['f_H']:>6.3f} "
                  f"{comp['ncpr']:>6.3f} {comp['Cq1']:>7.4f} "
                  f"{comp['largest_cluster_frac']:>6.3f} "
                  f"{comp['stiffness']:>7.3f}")

    # ============================================================
    # Phase 2: Solvus classification
    # ============================================================
    if verbose:
        print()
        print("=" * 90)
        print("  PHASE 2: Solvus Classification (S > 0 → folded)")
        print("=" * 90)
        print(f"  {'Protein':<15} {'S':>7} {'S_base':>7} {'Phase':<12} "
              f"{'True':<8} {'Match':>5}")
        print("-" * 90)

    correct = 0
    total = 0
    for key, prot in data.items():
        comp = all_comp[key]
        sol = calc_solvus(comp)
        meta = assess_metastability(comp, sol)
        all_sol[key] = {**sol, **meta}
        
        true_cls = prot['cls']
        pred_cls = sol['pred_cls']
        match = '✓' if pred_cls == true_cls else '✗'
        if pred_cls == true_cls:
            correct += 1
        total += 1

        if verbose:
            print(f"  {key:<15} {sol['S']:>+7.3f} {sol['S_base']:>+7.3f} "
                  f"{sol['phase']:<12} {true_cls:<8} {match:>5}")

    phase_acc = correct / total * 100 if total > 0 else 0
    if verbose:
        print("-" * 90)
        print(f"  Classification accuracy: {correct}/{total} = {phase_acc:.1f}%")

    # ============================================================
    # Phase 3: Energy depth
    # ============================================================
    folded_keys = [k for k, p in data.items() if p.get('dG', 0) != 0]
    
    if folded_keys:
        if verbose:
            print()
            print("=" * 90)
            print("  PHASE 3: Energy Depth  ΔG = −ε × N_H(core)")
            print("=" * 90)
            print(f"  {'Protein':<15} {'N_Hc':>5} {'ΔG_pred':>8} {'ΔG_exp':>8} "
                  f"{'Resid':>8} {'N_Ala':>5}")
            print("-" * 90)

        dG_exp_arr = []
        dG_pred_arr = []
        for key in folded_keys:
            eng = calc_dG(all_comp[key], model='core')
            all_eng[key] = eng
            dG_exp = data[key]['dG']
            dG_pred = eng['dG_primary']
            resid = dG_exp - dG_pred
            dG_exp_arr.append(dG_exp)
            dG_pred_arr.append(dG_pred)

            if verbose:
                print(f"  {key:<15} {eng['N_H_core']:>5d} {dG_pred:>+8.2f} "
                      f"{dG_exp:>+8.2f} {resid:>+8.2f} {eng['N_Ala']:>5d}")

        dG_exp_arr = np.array(dG_exp_arr)
        dG_pred_arr = np.array(dG_pred_arr)
        residuals = dG_exp_arr - dG_pred_arr
        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals ** 2))
        r = np.corrcoef(dG_exp_arr, dG_pred_arr)[0, 1]

        # LOO cross-validation
        N_fold = len(folded_keys)
        NH_arr = np.array([all_comp[k]['N_H_core'] for k in folded_keys], dtype=float)
        loo_preds = []
        for i in range(N_fold):
            mask = np.ones(N_fold, dtype=bool)
            mask[i] = False
            eps_loo = -np.mean(dG_exp_arr[mask]) / np.mean(NH_arr[mask])
            loo_preds.append(-eps_loo * NH_arr[i])
        loo_preds = np.array(loo_preds)
        loo_residuals = dG_exp_arr - loo_preds
        loo_mae = np.mean(np.abs(loo_residuals))
        eps_loo_spread = np.std([-np.mean(dG_exp_arr[np.arange(N_fold) != i])
                                  / np.mean(NH_arr[np.arange(N_fold) != i])
                                  for i in range(N_fold)])
        loo_cv = eps_loo_spread / EPSILON_CORE * 100

        if verbose:
            print("-" * 90)
            print(f"  ε_core = {EPSILON_CORE:.3f} kcal/mol")
            print(f"  MAE = {mae:.2f} kcal/mol | RMSE = {rmse:.2f} | r = {r:.3f}")
            print(f"  LOO CV: MAE = {loo_mae:.2f} | ε spread = {loo_cv:.1f}%")
    else:
        mae, rmse, r, loo_mae, loo_cv = 0, 0, 0, 0, 0

    # ============================================================
    # Summary
    # ============================================================
    summary = {
        'n_proteins': total,
        'phase_accuracy': phase_acc,
        'phase_correct': correct,
        'energy_mae': float(mae),
        'energy_rmse': float(rmse),
        'energy_r': float(r),
        'energy_loo_mae': float(loo_mae),
        'energy_loo_cv': float(loo_cv),
        'epsilon_core': EPSILON_CORE,
    }

    if verbose:
        print()
        print("=" * 90)
        print("  SUMMARY")
        print("=" * 90)
        print(f"  Phase classification:  {correct}/{total} ({phase_acc:.0f}%)")
        print(f"  Energy MAE:            {mae:.2f} kcal/mol (r = {r:.3f})")
        print(f"  LOO cross-validation:  {loo_cv:.1f}% ε stability")
        print(f"  Parameters:            Phase=11, Energy=1, Total=12")
        print("=" * 90)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Protein-SSOC Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validate.py                              # Bundled test set
  python validate.py --data my_proteins.json      # Custom dataset
  python validate.py --data data/proteins.json -q # Quiet mode
        """)
    parser.add_argument('--data', '-d', type=str, default=None,
                        help='Path to protein dataset JSON')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress detailed output')
    args = parser.parse_args()

    # Default: bundled dataset
    if args.data is None:
        default_path = Path(__file__).parent / 'data' / 'proteins.json'
        if default_path.exists():
            data_path = default_path
        else:
            print("Error: No data file specified and default data/proteins.json not found")
            print("Usage: python validate.py --data <path_to_json>")
            sys.exit(1)
    else:
        data_path = Path(args.data)

    print(f"\n  Loading data from: {data_path}\n")
    proteins = load_data(data_path)
    print(f"  Loaded {len(proteins)} proteins\n")

    summary = validate(proteins, verbose=not args.quiet)
    return summary


if __name__ == '__main__':
    main()
