# Protein-SSOC: Sequence-Space Occupation Classification

**A physics-based protein folding classification framework using alloy metallurgy analogies.**

> 13/13 perfect classification + ΔG MAE = 0.68 kcal/mol with only 12 parameters  

## Overview

Protein-SSOC treats amino acid sequences as binary alloy compositions and applies materials science phase diagram theory to predict:

1. **Phase** — "Will it fold?" (Solvus boundary classification)
2. **Energy** — "How stable?" (Thermodynamic depth from hydrophobic core count)

### Key Insight

These are independent questions, just as in metallurgy:
- Phase diagram (solvus) ≠ Phase energy (mixing enthalpy)
- A protein can be in the "folded" phase region but have varying stability depths

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/protein-ssoc.git
cd protein-ssoc
pip install numpy  # only dependency
```

## Quick Start

```python
from protein_ssoc import count_atoms, calc_solvus, calc_dG

# Ubiquitin
seq = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"

comp = count_atoms(seq)
sol  = calc_solvus(comp)
eng  = calc_dG(comp)

print(f"Phase: {sol['pred_cls']}")       # → folded
print(f"Solvus S = {sol['S']:.3f}")      # → S > 0 (inside two-phase region)
print(f"ΔG = {eng['dG_primary']:.1f} kcal/mol")  # → −7.1 kcal/mol
```

## Validation

```bash
# Run with bundled test set (13 proteins)
cd protein_ssoc
python validate.py

# Run with your own data
python validate.py --data path/to/your_proteins.json
```

### Data Format

```json
{
  "proteins": {
    "my_protein": {
      "seq": "AMINOACIDSEQUENCE...",
      "dG": -7.2,
      "cls": "folded"
    }
  }
}
```

- `seq`: Amino acid sequence (single-letter code)
- `dG`: Experimental ΔG in kcal/mol (use 0.0 for IDPs)
- `cls`: `"folded"` or `"idp"`

## Architecture

```
protein_ssoc/
├── __init__.py       # Public API
├── ssoc_core.py      # AA tables + composition analysis (count_atoms)
├── ssoc_phase.py     # Phase classification (calc_solvus)
├── ssoc_energy.py    # Energy depth (calc_dG)
├── validate.py       # Validation script (external data only)
└── data/
    └── proteins.json # Bundled benchmark dataset
```

### Module Independence

| Module | Question | Parameters | Output |
|--------|----------|-----------|--------|
| `ssoc_phase` | Will it fold? | 11 | S (solvus), phase class |
| `ssoc_energy` | How stable? | 1 (ε) | ΔG (kcal/mol) |

Both modules depend on `ssoc_core.count_atoms()` but are independent of each other.

## The Alloy Analogy

| Metallurgy | Protein-SSOC |
|-----------|-------------|
| Binary alloy composition | Hydrophobic fraction (f_H) |
| Solvus boundary | Uversky boundary (extended) |
| Two-phase region | Folded state |
| Solid solution | Intrinsically disordered |
| SRO (Warren-Cowley) | Charge correlation C_q(1) |
| Precipitate nuclei | Hydrophobic clusters |
| Lattice strain | Proline rigidity |
| Grain boundary segregation | Charge asymmetry |
| Packing defect dopant | Alanine (ε_Ala < 0) |

## Phase Model (11 parameters)

Extended Solvus:
```
S = S_base + Σ(corrections)
S_base = H_norm − (1.151 × |NCPR_eff| + 0.413)
```

Corrections include Gibbs-Thomson finite-size, charge patch relaxation, Warren-Cowley SRO, hydrophobic clusters, Pro/Gly rigidity, Shannon entropy, and charge segregation.

## Energy Model (1 parameter)

```
ΔG = −ε × N_H(core)
ε = 0.274 kcal/mol
```

Where N_H(core) counts hydrophobic residues excluding Alanine (a packing defect dopant).

Physical basis: ε = γ_eff × A_eff × η
- γ_eff ≈ 16 cal/mol/Å² (solvation parameter)
- A_eff ≈ 148 Å² (mean buried SASA)
- η ≈ 0.12 (packing efficiency)

## Performance

| Metric | Value |
|--------|-------|
| Phase classification | 13/13 (100%) |
| Energy MAE | 0.68 kcal/mol |
| Energy correlation | r = 0.980 |
| LOO CV ε stability | 1.7% |
| Total parameters | 12 |

## Version History

- **v1.4** — Phase + Energy unified (this release)
- **v1.5** — Temperature validation (ε as compensation residual)
- **v1.6** — Sequence-calibrated ε, disulfide extension, Megascale validation

## Citation

If you use Protein-SSOC in your research, please cite:

```
Iizumi, M. et al. (2026). Protein-SSOC: Sequence-Space Occupation Classification
using Alloy Phase Diagram Analogies. [preprint]
```

## License

MIT License

## Author

Masamichi Iizumi (Miosync, Inc.) + Tamaki + Gemini + GPT
