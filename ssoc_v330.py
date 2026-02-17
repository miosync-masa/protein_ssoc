"""
SSOC v3.18 "Gates 1+2+3a+3b" — Module
Reusable functions for ΔΔG prediction

v3.20: Gate 1r — MT Rigidity Penalty for D/R
  D,R charge intro → SCC cannot adapt → additional destabilization
  penalty = K_RIGID[ma] * bur (subtracted from pred)
  Data: N=747 X→charged unified, D excess +0.70, R excess +1.30

v3.19: Gate 4 — Charge Loss Cavity Restoration
  p_void=1.0 暫定値を ΔVol sigmoid に置換
  ΔVol > ~45Å³ で cavity physics が復活（D→A は charge dominant のまま）

v3.18 changes from v3.17:
  - SCC Gate 3b (PROVISIONAL): Aromatic-gain π Bonus
    When a charged residue (D/E/K/R) is replaced by an aromatic (F/W/Y)
    in a buried environment (bur ≥ 0.50):
      → π-network formation bonus not captured by PCC
      → Model under-stabilizes (OVER-predicts ΔΔG, i.e. too positive)
      → pred -= K0 * aa_weight (base correction)
      → pred -= K1 * aa_weight (boost if cat≥2 and bur≥0.70)
    Phase separation: Only Charged→Aro fires (from_hydro/ala/polar excluded
      as PCC already handles these correctly, mean_err ≈ 0).
    Design: base + boost two-tier structure
      Base (K0=1.0): broad correction for all buried charged→aro
      Boost (K1=1.0): additional correction when cation-π network is dense
    Subset KPI (from_charged, b≥0.50, N=50):
      MAE: 1.90 → 1.35, OVER: 17 → 9, win: 80%, UNDER: 0 → 1
      Global r: +0.0029 (partially recovers Gate 3a r-cost)

v3.17: Gate 3a phase separation refined (charged excluded from 3a)
v3.16: SCC Gate 3a (PROVISIONAL) — Aromatic-loss Network Buffering
v3.15: PCC Gate 2 (PROVISIONAL) — Backbone Entropy (Gly freedom)
v3.14: SCC Gate 1 — Charge Intro Network Resilience (cap×demand)
v3.13: P_void lattice plasticity (continuous cavity/strain material constants)

v3.30: Charge sub-gate — charge_helix / charge_sheet / charge_coil
  Phase='charge' split by mutation site SS for SS-specific corrections:
    charge_helix: dq damped by helix dipole shielding (dq×h_nb = -0.322)
    charge_sheet: dq only propagates via H-bond network (dq×e_nb = +0.859)
    charge_coil: long-range Coulomb dominates (lr_coulomb = +0.731)
  Cofactor guard: heme/metal PDBs → CSG disabled (charge env dominated by cofactor)
  GdnHCl validation: β-rich×sheet -0.101→+0.173, α-rich×helix +0.017→+0.102
v3.10: WT polar discount on charge_intro desolvation (C_HB_DISCOUNT=0.15)
v3.9: env-gated charge (3-type), data hygiene (2GNQ, 1ITM)
v3.8: p90 sigmoid burial (x0=0.60, C_CAV=0.050, C_HYD=0.050)
v3.7: backbone strain, helix/beta propensity
v3.6: bur² cavity, hydro transfer
"""
import math
import numpy as np

# ── Amino acid properties ──
AA_HYDRO = {
    'I':4.5, 'V':4.2, 'L':3.8, 'F':2.8, 'C':2.5, 'M':1.9, 'A':1.8,
    'G':-0.4, 'T':-0.7, 'S':-0.8, 'W':-0.9, 'Y':-1.3, 'P':-1.6,
    'H':-3.2, 'D':-3.5, 'E':-3.5, 'N':-3.5, 'Q':-3.5, 'K':-3.9, 'R':-4.5
}
AA_VOLUME = {
    'G':60.1, 'A':88.6, 'V':140.0, 'L':166.7, 'I':166.7, 'P':112.7,
    'F':189.9, 'W':227.8, 'Y':193.6, 'M':162.9, 'C':108.5, 'S':89.0,
    'T':116.1, 'N':114.1, 'D':111.1, 'Q':143.8, 'E':138.4, 'H':153.2,
    'K':168.6, 'R':173.4
}
AA_CHARGE = {'D':-1.0, 'E':-1.0, 'K':1.0, 'R':1.0, 'H':0.1}
AA_AROMATIC = {'F':1.0, 'W':1.0, 'Y':1.0, 'H':0.5}
AA_CATION = {'K':1.0, 'R':1.0, 'H':0.1}
AA_SULFUR = {'C':1.0, 'M':1.0}
AA_NROT = {'G':0,'A':0,'V':1,'L':2,'I':2,'P':0,'F':2,'W':2,'Y':2,
           'M':3,'C':1,'S':1,'T':1,'N':2,'D':2,'Q':3,'E':3,'H':2,'K':4,'R':4}
AA_HBDON = {'S':1,'T':1,'N':1,'Q':1,'Y':1,'W':1,'H':1,'K':1,'R':2,'C':0.5}
AA_HBACC = {'S':1,'T':1,'N':1,'Q':1,'Y':1,'D':2,'E':2,'H':1}
AA_BBRANCH = set('VIT')
POLAR_SET = set('STNQDEHKR')
THREE_TO_ONE = {
    'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G',
    'HIS':'H','ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N',
    'PRO':'P','GLN':'Q','ARG':'R','SER':'S','THR':'T','VAL':'V',
    'TRP':'W','TYR':'Y'
}

# Energy constants
C_HH=+0.36; C_HV=+0.26; C_VV=-0.12; C_VC=-0.09; C_CC=+1.04; C_PP=+0.58
C_NROT=-0.13; C_BRANCH=+0.37; C_HB=-0.18
C_CAVITY=0.075; C_HYDRO_TRANSFER=0.055
E_CATPI=+2.5; E_SPI_SINGLE=+1.5; E_SPI_3BR=+6.0
E_DESOLV=-5.5; E_GLY=0.7; E_PRO=0.65; E_NH=1.0
C_HB_DISCOUNT=0.15  # WT polar discount on charge_intro desolvation

# ── SCC Gate 1: Charge Intro Network Resilience (v3.14) ──
# Physics: β-branch network absorbs charge intro catastrophe
K_SCC = 5.0          # SCC compensation scale (kcal/mol per unit cap·demand)
B0_SCC = 0.15        # branch_frac threshold (minimum network redundancy)
CAP_MAX_SCC = 0.25   # maximum effective redundancy (β-core structural limit)
S0_SCC = 1.0         # demand saturation scale (most charge penalties exceed this)
OPP_THRESH_SCC = 4.0 # salt bridge distance cutoff (Å) — PCC/SCC boundary
ASYM_MAX_SCC = 0.20  # burial asymmetry phase boundary (closed core vs open surface)
HYDROPHOBIC_CORE_SCC = {'V','I','L','F','W','Y','M','A'}  # non-polar WT residues

# ── Gate 1r: MT Rigidity Penalty for D/R charge introduction (v3.20) ──
# D,R have rigid side chains that cannot adapt to buried environments
# even when SCC network provides resilience (Gate 1 over-corrects for D,R)
# Physics: D(short carboxylate, Nrot=2), R(planar guanidinium, Nrot=4 but terminal rigid)
# Data: D excess vs E = +0.70, R excess vs K = +1.30 (N=747, all X→charged)
K_RIGID = {'D': 1.0, 'R': 2.0}       # rigidity penalty scale per residue
BUR_MIN_RIGID = 0.30                   # surface mutations unaffected
RIGID_CAP = 2.5                        # safety cap
CHARGE_AA = {'D', 'E', 'K', 'R'}

# ── PCC Gate 2: Backbone Entropy — Gly Freedom (v3.15 PROVISIONAL) ──
# Gate 2a: G→X in helix — penalty (model too lenient)
# Gate 2b: X→G in beta — bonus (model too harsh)
# Physics: Gly has uniquely broad φ/ψ access; loss/gain of this freedom
#   is amplified by secondary structure constraints (helix=rigid, beta=accommodating)
K_G2A = 1.5          # G→X helix penalty scale (conservative; PROVISIONAL)
H0_G2A = 0.40        # helix_nf hard threshold (helix-dominated environment)
K_G2B = 1.5          # X→G beta bonus scale (conservative; PROVISIONAL)
B0_G2B = 0.50        # beta_nf hard threshold (beta-dominated environment)
BUR_MIN_G2 = 0.50    # minimum burial for Gate 2 (surface Gly less constrained)

# ── SCC Gate 3a: Aromatic-loss Network Buffering (v3.16 PROVISIONAL) ──
# When buried aromatic (F/W/Y) → non-hydrophobic residue:
#   π-network loss absorbed by SCC re-arrangement → model over-penalizes
# Phase separation: Aro→Hydro (V/L/I/M/C) excluded (PCC handles correctly)
# demand = P_void × aa_weight: "residual void × aromatic importance"
K_3A = 3.0            # SCC relaxation scale (PROVISIONAL)
BUR_MIN_3A = 0.70     # burial threshold (surface aromatics less network-dependent)
COMP_MAX_3A = 3.0     # safety cap on compensation (ops insurance)
AA_WEIGHT_3A = {'W': 1.0, 'F': 0.7, 'Y': 0.7}  # Trp > Phe ≈ Tyr
HYDRO_LARGE_3A = {'V', 'L', 'I', 'M', 'C'}  # excluded: packing→packing OK
CHARGED_3A = {'D', 'E', 'K', 'R'}  # excluded: handled by Gate 1 (charge SCC)

# ── SCC Gate 3b: Aromatic-gain π Bonus (v3.18 PROVISIONAL) ──
# When buried charged (D/E/K/R) → aromatic (F/W/Y):
#   π-network formation bonus not captured by PCC → model OVER-predicts
# Phase separation: Only Charged→Aro (other sources have mean_err ≈ 0)
# Two-tier: base (all buried) + boost (cation-π dense environment)
K0_3B = 1.0           # base correction scale
K1_3B = 1.0           # boost correction for cation-π dense sites
BUR_MIN_3B = 0.50     # burial threshold for base
BUR_MIN_3B_BOOST = 0.70  # burial threshold for boost tier
CAT_MIN_3B = 2        # minimum cation-π neighbors for boost
COMP_MAX_3B = 2.0     # safety cap (K0+K1 = 2.0 max per aa_w unit)
AA_WEIGHT_3B = {'W': 1.0, 'F': 0.7, 'Y': 0.7}  # same as 3a
CHARGED_3B = {'D', 'E', 'K', 'R'}  # source residues for this gate

# ── Gate 4: Charge Loss Cavity Restoration (v3.19) ──
# When charged residue → small/non-charged with large ΔVol:
#   p_void=1.0 assumption breaks — significant cavity is created
# Restore cavity physics via sigmoid transition on ΔVolume
# D→A (ΔVol=22.5): charge dominant, no cavity needed
# K→A (ΔVol=80.0): cavity dominant, full restoration
# R→G (ΔVol=113.3): extreme cavity
DVOL_X0_G4 = 45.0    # sigmoid midpoint (Å³) — between D→A(22) and E→A(50)
DVOL_K_G4 = 0.08     # sigmoid steepness

# ── Phase gate (v3.12) ──
# Phase-specific P_void material constants (lattice plasticity)
# P_void = P0 + a_pack·large_frac + a_comp·polar_frac + a_backbone·ss_nf
# cavity (FCC-like): sidechain reorientation dominant
# strain (BCC-like): backbone constraint dominant, compensation important
PVOID_PARAMS = {
    'cavity': {'P0': 0.60, 'a_pack': -0.3, 'a_comp': -0.2, 'a_backbone': +0.2, 'ss_key': 'helix_nf'},
    'strain': {'P0': 0.70, 'a_pack': -0.2, 'a_comp': -0.3, 'a_backbone': +0.1, 'ss_key': 'beta_nf'},
    'charge': {'P0': 1.00, 'a_pack': 0.0,  'a_comp': 0.0,  'a_backbone': 0.0,  'ss_key': None},  # neutral: SCC territory
}
PVOID_CLAMP = (0.30, 1.50)  # physical bounds

CHARGE_RECLASSIFY = {'D','E','K','R'}  # from_charged → force to charge phase

def _softmax3(a, b, c):
    """Softmax of 3 scores."""
    mx = max(a, b, c)
    ea = math.exp(a - mx); eb = math.exp(b - mx); ec = math.exp(c - mx)
    s = ea + eb + ec
    return ea/s, eb/s, ec/s

def compute_pdb_gate(frac_E, frac_H, core_frac, ACO, charge_density, N_res):
    """PDB-level phase scores."""
    s_cavity = (frac_H - 0.25) * 2.0 + (0.55 - core_frac) * 1.0 - (frac_E - 0.35) * 1.0
    s_strain = (frac_E - 0.35) * 2.0 + (core_frac - 0.55) * 2.0 + (ACO - 15) * 0.05
    s_charge = (charge_density - 0.25) * 3.0 + (100 / max(50, N_res) - 1.0) * 0.5
    return s_cavity, s_strain, s_charge

def compute_micro_gate(ss_local, beta_neigh_frac, helix_neigh_frac, bur):
    """Mutation-level phase scores."""
    m_cavity = (bur - 0.5) * 1.0 - (beta_neigh_frac - 0.5) * 0.5
    m_strain = (0.5 if ss_local == 'E' else 0) + (beta_neigh_frac - 0.5) * 1.0 - helix_neigh_frac * 2.0
    m_charge = (1.0 - bur) * 1.0 + helix_neigh_frac * 0.5
    return m_cavity, m_strain, m_charge

def get_phase(pdb_gate, micro_gate):
    """Combine PDB + micro gate, return phase name."""
    combined = (pdb_gate[0]+micro_gate[0], pdb_gate[1]+micro_gate[1], pdb_gate[2]+micro_gate[2])
    w_cav, w_str, w_chg = _softmax3(*combined)
    if w_cav >= w_str and w_cav >= w_chg: return 'cavity'
    if w_str >= w_chg: return 'strain'
    return 'charge'

def compute_pvoid(phase, large_frac, polar_frac, helix_nf, beta_nf):
    """Compute P_void from local lattice state.
    P_void = effective void fraction = probability that cavity persists as ΔΔG."""
    params = PVOID_PARAMS[phase]
    ss_nf = helix_nf if params['ss_key'] == 'helix_nf' else (beta_nf if params['ss_key'] == 'beta_nf' else 0.0)
    p_void = params['P0'] + params['a_pack'] * large_frac + params['a_comp'] * polar_frac + params['a_backbone'] * ss_nf
    return max(PVOID_CLAMP[0], min(PVOID_CLAMP[1], p_void))

def detect_cofactors(pdb_path):
    """Detect heme/metal cofactors from HETATM lines.
    Returns dict with has_heme, has_metal, has_cofactor."""
    has_heme = False; has_metal = False
    HEME_NAMES = {'HEM', 'HEC', 'HEA', 'HEB'}
    METAL_ATOMS = {'FE', 'ZN', 'CU', 'MN', 'CO', 'NI', 'MG'}
    try:
        with open(pdb_path) as f:
            for line in f:
                if line.startswith('HETATM'):
                    resn = line[17:20].strip()
                    if resn in HEME_NAMES: has_heme = True
                    atom_name = line[12:16].strip()
                    if atom_name in METAL_ATOMS: has_metal = True
    except: pass
    return {'has_heme': has_heme, 'has_metal': has_metal,
            'has_cofactor': has_heme or has_metal}

def compute_pdb_features(atoms, ss_map, backbone, nc_p90=None, pdb_path=None):
    """Compute PDB-level features for phase gate.
    Returns dict with frac_E, frac_H, core_frac, ACO, charge_density, N_res, nc_p90."""
    srnums = sorted(backbone.keys())
    from collections import Counter
    ss_counts = Counter(ss_map.get(rn, 'C') for rn in srnums)
    total_ss = max(1, sum(ss_counts.values()))
    frac_E = ss_counts.get('E', 0) / total_ss
    frac_H = ss_counts.get('H', 0) / total_ss
    # Compute nc_p90 for sigmoid burial
    ncs_all = [sum(1 for rn2, aa2, xyz2 in atoms if rn2 != rnum and np.linalg.norm(xyz2 - xyz) < 8.0)
               for rnum, aa, xyz in atoms]
    if nc_p90 is None:
        nc_p90 = max(8.0, np.percentile(ncs_all, 90))
    # Core fraction using sigmoid burial (x0=0.60, k=10)
    n_core = 0
    for nc_i in ncs_all:
        bur_i = sigmoid(nc_i / nc_p90, 0.60, 10.0)
        if bur_i > 0.7:
            n_core += 1
    core_frac = n_core / max(1, len(atoms))
    # ACO
    seq_seps = []
    for i, (rn1, aa1, xyz1) in enumerate(atoms):
        for j, (rn2, aa2, xyz2) in enumerate(atoms):
            if j <= i: continue
            if np.linalg.norm(xyz1 - xyz2) < 8.0:
                seq_seps.append(abs(rn2 - rn1))
    ACO = np.mean(seq_seps) if seq_seps else 0
    # Charge density
    n_charged = sum(1 for _, aa, _ in atoms if aa in ('D', 'E', 'K', 'R'))
    charge_density = n_charged / max(1, len(atoms))
    # Cofactor detection (v3.30)
    cofactor = detect_cofactors(pdb_path) if pdb_path else {'has_heme': False, 'has_metal': False, 'has_cofactor': False}
    return {
        'frac_E': frac_E, 'frac_H': frac_H, 'core_frac': core_frac,
        'ACO': ACO, 'charge_density': charge_density, 'N_res': len(atoms), 'nc_p90': nc_p90,
        'has_heme': cofactor['has_heme'], 'has_metal': cofactor['has_metal'],
        'has_cofactor': cofactor['has_cofactor']
    }


# Backbone strain constants (v3.7)
HELIX_PROP = {
    'A':1.0, 'L':0.79, 'M':0.74, 'E':0.71, 'K':0.65, 'Q':0.61,
    'R':0.56, 'I':0.52, 'F':0.47, 'D':0.40, 'W':0.37, 'V':0.35,
    'T':0.31, 'S':0.29, 'N':0.28, 'H':0.27, 'Y':0.25, 'C':0.24,
    'G':0.00, 'P':-0.50,
}
BETA_PROP = {
    'V':1.0, 'I':0.95, 'T':0.75, 'F':0.70, 'Y':0.70, 'W':0.65,
    'L':0.55, 'C':0.50, 'M':0.45, 'A':0.30, 'R':0.25, 'K':0.20,
    'Q':0.15, 'E':0.10, 'S':0.10, 'H':0.10, 'D':0.05, 'N':0.05,
    'G':0.00, 'P':-0.30,
}
CHARGE_AA = {'D','E','K','R','H'}
POLAR_AA_STRAIN = {'S','T','N','Q','D','E','K','R','H','Y','W','C'}
W_STRAIN_H = 1.1; W_STRAIN_E = 1.1
PRO_CAP = +0.7; PRO_CORE = -1.5; GLY_H = +0.4
BETA_CHARGE = +0.2; BETA_POLAR = -0.7
SAT_H = 1.2; SAT_E = 0.8; B_E = +0.20

_hvals=list(AA_HYDRO.values()); H_MEAN=np.mean(_hvals); H_STD=np.std(_hvals)
_vvals=list(AA_VOLUME.values()); V_MEAN=np.mean(_vvals); V_STD=np.std(_vvals)

def nh(aa): return (AA_HYDRO.get(aa,0)-H_MEAN)/H_STD
def nv(aa): return (AA_VOLUME.get(aa,130)-V_MEAN)/V_STD
def sigmoid(x,x0,k):
    z=np.clip(-k*(x-x0),-30,30); return 1.0/(1.0+np.exp(z))

def dihedral_angle(p1,p2,p3,p4):
    b1=p2-p1;b2=p3-p2;b3=p4-p3
    n1=np.cross(b1,b2);n2=np.cross(b2,b3)
    m1=np.cross(n1,b2/max(np.linalg.norm(b2),1e-10))
    return -math.degrees(math.atan2(np.dot(m1,n2),np.dot(n1,n2)))

def place_virtual_CB(N,CA,C):
    v1=(CA-N)/max(np.linalg.norm(CA-N),1e-10)
    v2=(CA-C)/max(np.linalg.norm(CA-C),1e-10)
    bisect=v1+v2; bisect/=max(np.linalg.norm(bisect),1e-10)
    n_perp=np.cross(v1,v2); n_perp/=max(np.linalg.norm(n_perp),1e-10)
    theta=math.radians(54.75)
    cb_dir=math.cos(theta)*bisect+math.sin(theta)*n_perp
    cb_dir/=max(np.linalg.norm(cb_dir),1e-10)
    return CA+1.52*cb_dir

# ── Secondary structure assignment (φ/ψ based) ──
def assign_ss(backbone):
    """Assign secondary structure from backbone φ/ψ angles."""
    srnums = sorted(backbone.keys())
    ss = {}
    for i, rnum in enumerate(srnums):
        bb = backbone[rnum]
        if not all(a in bb for a in ('N','CA','C')):
            ss[rnum] = 'C'; continue
        phi = psi = None
        if i > 0:
            prev = srnums[i-1]
            bbp = backbone.get(prev, {})
            if 'C' in bbp and prev == rnum - 1:
                try: phi = dihedral_angle(bbp['C'], bb['N'], bb['CA'], bb['C'])
                except: pass
        if i < len(srnums) - 1:
            nxt = srnums[i+1]
            bbn = backbone.get(nxt, {})
            if 'N' in bbn and nxt == rnum + 1:
                try: psi = dihedral_angle(bb['N'], bb['CA'], bb['C'], bbn['N'])
                except: pass
        if phi is None or psi is None:
            ss[rnum] = 'C'; continue
        if -100 < phi < -30 and -80 < psi < -10: ss[rnum] = 'H'
        elif -170 < phi < -50 and (80 < psi < 180 or -180 < psi < -120): ss[rnum] = 'E'
        elif 30 < phi < 100 and -20 < psi < 60: ss[rnum] = 'T'
        else: ss[rnum] = 'C'
    # Smoothing: isolated H/E → C
    for i, rnum in enumerate(srnums):
        if ss[rnum] in ('H','E'):
            nbr = []
            if i > 0: nbr.append(ss.get(srnums[i-1],'C'))
            if i < len(srnums)-1: nbr.append(ss.get(srnums[i+1],'C'))
            if ss[rnum] not in nbr: ss[rnum] = 'C'
    return ss

def get_helix_position(ss_map, pos):
    """Classify helix residue as 'cap' (terminal 2) or 'core' (interior)."""
    srnums = sorted(ss_map.keys())
    if ss_map.get(pos) != 'H': return 'none'
    try: idx = srnums.index(pos)
    except ValueError: return 'none'
    left = idx
    while left > 0 and ss_map.get(srnums[left-1]) == 'H': left -= 1
    right = idx
    while right < len(srnums)-1 and ss_map.get(srnums[right+1]) == 'H': right += 1
    pos_in_helix = idx - left
    helix_len = right - left + 1
    if pos_in_helix <= 1 or pos_in_helix >= helix_len - 2: return 'cap'
    return 'core'

def calc_backbone_strain(wa, ma, ss, hpos='none',
                         is_charge_intro=False, is_polar_intro=False):
    """Backbone strain from SS-dependent propensity + special terms.
    Returns strain energy (negative = destabilizing)."""
    if ss == 'H':
        dp = HELIX_PROP.get(ma, 0.3) - HELIX_PROP.get(wa, 0.3)
        raw = W_STRAIN_H * dp
        if ma == 'P':
            raw += PRO_CAP if hpos == 'cap' else PRO_CORE
        if ma == 'G' and wa != 'G':
            raw += GLY_H
        return SAT_H * math.tanh(raw / SAT_H)
    elif ss == 'E':
        dp = BETA_PROP.get(ma, 0.3) - BETA_PROP.get(wa, 0.3)
        raw = W_STRAIN_E * dp
        if is_charge_intro: raw += BETA_CHARGE
        if is_polar_intro: raw += BETA_POLAR
        return SAT_E * math.tanh(raw / SAT_E) + B_E
    return 0.0

# ── PDB parser ──
def parse_pdb(pdb_path, chain='A'):
    lines=open(pdb_path).read().split('\n')
    has_models=any(l.startswith('MODEL') for l in lines)
    filtered=[]
    in_model1=not has_models
    for l in lines:
        if l.startswith('MODEL'): in_model1=(int(l.split()[1])==1)
        if l.startswith('ENDMDL') and in_model1: filtered.append(l); break
        if in_model1: filtered.append(l)

    for ch_try in [chain,'A','B',' ']:
        backbone={};atoms=[];all_heavy=[]
        for l in filtered:
            if not l.startswith('ATOM'): continue
            an=l[12:16].strip();rn=l[17:20].strip();c=l[21]
            if c!=ch_try: continue
            if an.startswith('H'): continue
            try:
                rnum=int(l[22:26])
                xyz=np.array([float(l[30:38]),float(l[38:46]),float(l[46:54])])
            except: continue
            if an=='CA': atoms.append((rnum,THREE_TO_ONE.get(rn,'X'),xyz))
            if an in ('N','CA','C'):
                if rnum not in backbone: backbone[rnum]={'rn':rn}
                backbone[rnum][an]=xyz
            all_heavy.append((rnum,an,xyz))
        if len(atoms)>10: return backbone,atoms,all_heavy

    backbone={};atoms=[];all_heavy=[];seen=set()
    for l in filtered:
        if not l.startswith('ATOM'): continue
        an=l[12:16].strip();rn=l[17:20].strip()
        if an.startswith('H'): continue
        try:
            rnum=int(l[22:26])
            xyz=np.array([float(l[30:38]),float(l[38:46]),float(l[46:54])])
        except: continue
        if an=='CA' and rnum not in seen:
            atoms.append((rnum,THREE_TO_ONE.get(rn,'X'),xyz))
            seen.add(rnum)
        if an in ('N','CA','C'):
            if rnum not in backbone: backbone[rnum]={'rn':rn}
            if an not in backbone[rnum]: backbone[rnum][an]=xyz
        all_heavy.append((rnum,an,xyz))
    return backbone,atoms,all_heavy

# ── Correction functions ──
def correction_gly(wa,ma,d_CB_min=999.0,kappa=90.0,n_branch=0):
    if wa!='G' and ma!='G': return 0.0
    if wa=='G':
        s1=1/(1+math.exp(-5*(d_CB_min-2.6)))
        s2=1/(1+math.exp(0.05*(kappa-80)))
        s3=1/(1+math.exp(1.5*(n_branch-3)))
        return -1.0*(1-s1*s2*s3)+E_GLY
    else: return -E_GLY

def correction_pro(wa,ma):
    if wa!='P' and ma!='P': return 0.0
    return -E_PRO if wa=='P' else E_PRO-E_NH

def correction_charge(wa,ma,bur,neighbors,polar_frac=0.0):
    q_wt=AA_CHARGE.get(wa,0); q_mt=AA_CHARGE.get(ma,0)
    has_wt=abs(q_wt)>0.3; has_mt=abs(q_mt)>0.3
    if not has_wt and not has_mt: return 0.0
    neigh_charge=sum(AA_CHARGE.get(n,0) for n in neighbors)
    # TYPE 1: charge_loss (charged → neutral)
    if has_wt and not has_mt:
        if neigh_charge!=0 and q_wt*neigh_charge>0:
            return 1.5*abs(q_wt)*bur  # repulsion removal → stabilizing
        return 0.0  # salt bridge/neutral → let contact terms handle
    # TYPE 2: charge_intro (neutral → charged)
    if not has_wt and has_mt:
        dry_bur=bur*(1.0-polar_frac)
        desolv_factor=sigmoid(dry_bur,0.40,8.0)
        E_desolv=E_DESOLV*desolv_factor
        salt_bridge=0
        if q_mt!=0 and neigh_charge!=0:
            if q_mt*neigh_charge<0: salt_bridge=-0.5*min(abs(q_mt),abs(neigh_charge))
            else: salt_bridge=+0.15*min(abs(q_mt),abs(neigh_charge))
        E_total=E_desolv+salt_bridge
        # WT polar discount: polar WT already partially solvated
        hb_wt=AA_HBDON.get(wa,0)+AA_HBACC.get(wa,0)
        if hb_wt>0:
            E_total*=max(0.1,1.0-C_HB_DISCOUNT*hb_wt)
        return E_total
    # TYPE 3: charge_flip (charged → differently charged)
    if has_wt and has_mt and abs(q_wt-q_mt)>0.3:
        loss_part=0
        if neigh_charge!=0 and q_wt*neigh_charge>0:
            loss_part=1.5*abs(q_wt)*bur
        intro_part=0
        if q_mt*neigh_charge<0:
            intro_part=+0.3*min(abs(q_mt),abs(neigh_charge))*bur
        elif q_mt*neigh_charge>0:
            intro_part=-0.3*min(abs(q_mt),abs(neigh_charge))*bur
        return loss_part+intro_part
    return 0.0  # charge_conserved (D→E, K→R)

def correction_special(wa,ma,bur,neighbors):
    nc=max(1,len(neighbors))
    neigh_arom=sum(AA_AROMATIC.get(n,0) for n in neighbors)
    neigh_cat=sum(AA_CATION.get(n,0) for n in neighbors)
    neigh_sul=sum(1 for n in neighbors if n in AA_SULFUR)
    E=0.0
    catpi_net=(max(0,AA_CATION.get(ma,0)-AA_CATION.get(wa,0))*neigh_arom/nc+
              max(0,AA_AROMATIC.get(ma,0)-AA_AROMATIC.get(wa,0))*neigh_cat/nc-
              max(0,AA_CATION.get(wa,0)-AA_CATION.get(ma,0))*neigh_arom/nc-
              max(0,AA_AROMATIC.get(wa,0)-AA_AROMATIC.get(ma,0))*neigh_cat/nc)
    if abs(catpi_net)>0.01: E+=E_CATPI*bur*catpi_net
    spi_net=(max(0,(1 if ma in AA_SULFUR else 0)-(1 if wa in AA_SULFUR else 0))*neigh_arom/nc+
             max(0,AA_AROMATIC.get(ma,0)-AA_AROMATIC.get(wa,0))*neigh_sul/nc-
             max(0,(1 if wa in AA_SULFUR else 0)-(1 if ma in AA_SULFUR else 0))*neigh_arom/nc-
             max(0,AA_AROMATIC.get(wa,0)-AA_AROMATIC.get(ma,0))*neigh_sul/nc)
    if abs(spi_net)>0.01:
        E_spi=E_SPI_3BR/3 if neigh_arom>=3 else E_SPI_SINGLE
        E+=E_spi*bur*spi_net
    return E

def compute_ddg(wa,ma,bur,neighbors,d_CB_min=999.0,kappa=90.0,n_branch=0,polar_frac=0.0,cav_scale=1.0):
    nc=max(1,len(neighbors))
    du_hh=du_hv=du_vv=du_vc=du_cc=du_pp=0
    for naa in neighbors:
        dh=nh(ma)-nh(wa);dv=nv(ma)-nv(wa)
        hn=nh(naa);vn=nv(naa)
        qn=AA_CHARGE.get(naa,0);pn=AA_AROMATIC.get(naa,0)
        dq=AA_CHARGE.get(ma,0)-AA_CHARGE.get(wa,0)
        dp=AA_AROMATIC.get(ma,0)-AA_AROMATIC.get(wa,0)
        du_hh+=dh*hn;du_hv+=dh*vn+dv*hn;du_vv+=dv*vn
        du_vc+=dv*qn;du_cc+=dq*qn;du_pp+=dp*pn
    e_gly=correction_gly(wa,ma,d_CB_min,kappa,n_branch)
    e_pro=correction_pro(wa,ma)
    e_charge=correction_charge(wa,ma,bur,neighbors,polar_frac)
    e_special=correction_special(wa,ma,bur,neighbors)
    has_charge=abs(e_charge)>0.01
    has_special=abs(e_special)>0.01
    use_cc=0 if has_charge else du_cc
    E_contact=(C_HH*du_hh+C_HV*du_hv+C_VV*du_vv+C_VC*du_vc+C_CC*use_cc+C_PP*du_pp)/nc
    dnrot=AA_NROT.get(ma,0)-AA_NROT.get(wa,0)
    dbranch=(1 if ma in AA_BBRANCH else 0)-(1 if wa in AA_BBRANCH else 0)
    dhb=(AA_HBDON.get(ma,0)+AA_HBACC.get(ma,0))-(AA_HBDON.get(wa,0)+AA_HBACC.get(wa,0))
    E_entropy=C_NROT*dnrot+C_BRANCH*dbranch+C_HB*bur*dhb
    vol_loss=max(0,AA_VOLUME.get(wa,130)-AA_VOLUME.get(ma,130))
    hydro_loss=max(0,AA_HYDRO.get(wa,0)-AA_HYDRO.get(ma,0))
    E_cavity=-(C_CAVITY*cav_scale*vol_loss*bur**2+C_HYDRO_TRANSFER*hydro_loss*bur)
    ddg=E_contact+E_entropy+E_cavity
    tags=[]
    if abs(e_gly)>0.01: ddg+=e_gly; tags.append('G')
    if abs(e_pro)>0.01: ddg+=e_pro; tags.append('P')
    if has_charge: ddg+=e_charge; tags.append('C')
    if has_special: ddg+=e_special; tags.append('S')
    if not tags: tags.append('F')
    return ddg, '+'.join(tags)

def compute_scc_gate3a(wa, ma, bur, p_void):
    """SCC Gate 3a (PROVISIONAL): Aromatic-loss Network Buffering.
    
    When buried aromatic → non-hydrophobic: SCC absorbs π-network loss.
    Phase separation: Aro→Hydro excluded (PCC handles packing replacement).
    
    Returns: aro_comp (float, ≥0), gate3a_info (dict)
    """
    AROMATIC = {'F', 'W', 'Y'}
    
    # Must be aromatic loss
    if wa not in AROMATIC or ma in AROMATIC:
        return 0.0, {'gate': 'not_aro_loss'}
    
    # Phase separation: exclude Aro→Hydro (PCC正常)
    if ma in HYDRO_LARGE_3A:
        return 0.0, {'gate': 'g3a_to_hydro_excluded', 'wa': wa, 'ma': ma}
    
    # Phase separation: exclude Aro→Charged (Gate 1 handles charge physics)
    if ma in CHARGED_3A:
        return 0.0, {'gate': 'g3a_to_charged_excluded', 'wa': wa, 'ma': ma}
    
    # Burial gate
    if bur < BUR_MIN_3A:
        return 0.0, {'gate': 'g3a_below_bur', 'bur': bur}
    
    # demand = P_void × aa_weight
    aa_w = AA_WEIGHT_3A.get(wa, 0.7)
    demand = p_void * aa_w
    comp = min(K_3A * demand, COMP_MAX_3A)
    
    return comp, {
        'gate': 'g3a_aro_loss_scc',
        'wa': wa, 'ma': ma, 'bur': bur,
        'p_void': p_void, 'aa_w': aa_w, 'demand': demand, 'comp': comp,
    }


def compute_scc_gate3b(wa, ma, bur, n_cat):
    """SCC Gate 3b (PROVISIONAL): Aromatic-gain π Bonus.
    
    When buried charged → aromatic: π-network formation bonus missed by PCC.
    Phase separation: Only Charged→Aro fires.
    Two-tier: base (all buried) + boost (cation-π dense).
    
    Returns: aro_bonus (float, ≥0, to be SUBTRACTED from pred), gate3b_info (dict)
    """
    AROMATIC = {'F', 'W', 'Y'}
    
    # Must be aromatic gain
    if ma not in AROMATIC or wa in AROMATIC:
        return 0.0, {'gate': 'not_aro_gain'}
    
    # Phase separation: only Charged→Aro
    if wa not in CHARGED_3B:
        return 0.0, {'gate': 'g3b_not_from_charged', 'wa': wa, 'ma': ma}
    
    # Burial gate (base tier)
    if bur < BUR_MIN_3B:
        return 0.0, {'gate': 'g3b_below_bur', 'bur': bur}
    
    aa_w = AA_WEIGHT_3B.get(ma, 0.7)
    bonus = K0_3B * aa_w
    
    # Boost tier: cation-π dense + deeply buried
    if n_cat >= CAT_MIN_3B and bur >= BUR_MIN_3B_BOOST:
        bonus += K1_3B * aa_w
    
    bonus = min(bonus, COMP_MAX_3B)
    
    return bonus, {
        'gate': 'g3b_aro_gain_pi',
        'wa': wa, 'ma': ma, 'bur': bur,
        'n_cat': n_cat, 'aa_w': aa_w,
        'base': K0_3B * aa_w,
        'boost': K1_3B * aa_w if (n_cat >= CAT_MIN_3B and bur >= BUR_MIN_3B_BOOST) else 0.0,
        'bonus': bonus,
    }


def compute_pcc_gate2(wa, ma, bur, helix_nf, beta_nf):
    """PCC Gate 2 (PROVISIONAL): Backbone Entropy — Gly conformational freedom.
    
    Gate 2a: G→X in helix → additional destabilization (pred more negative)
    Gate 2b: X→G in beta → stabilization bonus (pred more positive)
    
    Returns: bb_comp (float, signed), gate2_info (dict)
    """
    # Gate 2a: Gly-loss in helix
    if wa == 'G' and ma != 'P':  # G→P has its own backbone physics
        if bur >= BUR_MIN_G2 and helix_nf >= H0_G2A:
            return -K_G2A, {'gate':'g2a_gly_helix', 'helix_nf':helix_nf, 'bur':bur}
        return 0.0, {'gate':'g2a_below_thresh', 'helix_nf':helix_nf, 'bur':bur}
    
    # Gate 2b: Gly-gain in beta
    if ma == 'G' and wa != 'P':  # P→G has its own backbone physics
        if bur >= BUR_MIN_G2 and beta_nf >= B0_G2B:
            return +K_G2B, {'gate':'g2b_gly_beta', 'beta_nf':beta_nf, 'bur':bur}
        return 0.0, {'gate':'g2b_below_thresh', 'beta_nf':beta_nf, 'bur':bur}
    
    return 0.0, {'gate':'not_gly_trigger'}

def compute_scc_gate1(wa, ma, bur, atoms, pos, target, neighbors_aa, polar_frac, com):
    """SCC Gate 1: Charge Intro Network Resilience.
    
    Computes SCC compensation for hydrophobic→charged mutations in buried core
    where PCC local rescue (salt bridge, snorkel) is unavailable.
    
    Returns: scc_comp (float), scc_info (dict with gate details)
    """
    CHARGED = {'D','E','K','R'}
    POSITIVE = {'K','R'}
    NEGATIVE = {'D','E'}
    
    # Gate 0: Is this a charge introduction into hydrophobic core?
    is_charge_intro = (wa in HYDROPHOBIC_CORE_SCC and ma in CHARGED and bur > 0.7)
    if not is_charge_intro:
        return 0.0, {'gate':'not_charge_intro'}
    
    # Get coordinate-based neighbor info
    neighbors_full = [(rnum, aa, xyz) for rnum, aa, xyz in atoms
                      if rnum != pos and np.linalg.norm(xyz - target) < 8.0]
    nc = max(1, len(neighbors_full))
    
    # Branch fraction (V/I/T network redundancy)
    branch_set = {'V','I','T'}
    branch_frac = sum(1 for _, aa, _ in neighbors_full if aa in branch_set) / nc
    if branch_frac <= B0_SCC:
        return 0.0, {'gate':'low_branch', 'branch_frac':branch_frac}
    
    # Nearest opposite charge distance (salt bridge check)
    if ma in POSITIVE: opp_set = NEGATIVE
    elif ma in NEGATIVE: opp_set = POSITIVE
    else: opp_set = set()
    opp_dists = [np.linalg.norm(xyz - target) for _, aa, xyz in neighbors_full if aa in opp_set]
    nearest_opp = min(opp_dists) if opp_dists else 99.0
    if nearest_opp < OPP_THRESH_SCC:
        return 0.0, {'gate':'salt_bridge', 'nearest_opp':nearest_opp, 'branch_frac':branch_frac}
    
    # Burial asymmetry (closed core vs open surface)
    vec_to_surface = target - com
    norm = np.linalg.norm(vec_to_surface)
    if norm > 0.01:
        vec_to_surface = vec_to_surface / norm
    outward_burs = []; inward_burs = []
    for rn, aa, xyz in neighbors_full:
        vec_nb = xyz - target
        dist = np.linalg.norm(vec_nb)
        if dist < 0.1: continue
        dot = np.dot(vec_nb / dist, vec_to_surface)
        nb_nc = sum(1 for r2, _, x2 in atoms if r2 != rn and np.linalg.norm(x2 - xyz) < 8.0)
        nb_bur = min(1.0, nb_nc / 16.0)
        if dot > 0.3:
            outward_burs.append(nb_bur)
        else:
            inward_burs.append(nb_bur)
    burial_asym = (np.mean(inward_burs) if inward_burs else 1.0) - \
                  (np.mean(outward_burs) if outward_burs else 1.0)
    if burial_asym >= ASYM_MAX_SCC:
        return 0.0, {'gate':'open_surface', 'burial_asym':burial_asym, 'branch_frac':branch_frac}
    
    # All gates passed — compute SCC compensation
    # cap: network redundancy (clamped)
    cap = min(branch_frac - B0_SCC, CAP_MAX_SCC)
    
    # demand: local contradiction magnitude (saturating)
    charge_penalty = abs(correction_charge(wa, ma, bur, neighbors_aa, polar_frac))
    demand = 1.0 - math.exp(-charge_penalty / S0_SCC)
    
    # SCC compensation (positive = stabilizing correction to overly negative PCC prediction)
    scc_comp = K_SCC * cap * demand
    
    return scc_comp, {
        'gate':'scc_active', 'branch_frac':branch_frac, 'nearest_opp':nearest_opp,
        'burial_asym':burial_asym, 'cap':cap, 'demand':demand, 'scc_comp':scc_comp,
        'charge_penalty':charge_penalty
    }

def compute_rigid_penalty(wa, ma, bur):
    """Gate 1r: MT Rigidity Penalty for D/R charge introduction.
    
    Rigid charged side chains (D: short carboxylate, R: planar guanidinium)
    cannot adapt to SCC network → additional destabilization not captured by Gate 1.
    Returns penalty >= 0 (to be SUBTRACTED from ddg_pred).
    """
    if ma not in K_RIGID:
        return 0.0, {'gate': 'not_rigid_mt'}
    if wa in CHARGE_AA:
        return 0.0, {'gate': 'charged_to_charged'}
    if bur < BUR_MIN_RIGID:
        return 0.0, {'gate': 'too_surface', 'bur': bur}
    
    pen = K_RIGID[ma] * bur
    pen = min(pen, RIGID_CAP)
    
    return pen, {'gate': 'rigid_active', 'ma': ma, 'bur': bur, 'penalty': pen}

# ── Gate 5: Charge Sub-Gate (v3.30) ──────────────────────────────────────
# charge_helix: helix dipole shields charge changes
CSG_HELIX_DQ     = +0.2740   # dq coefficient
CSG_HELIX_DQ_HNB = -0.3220   # dq × helix_neighbor_frac (dipole shielding)
CSG_HELIX_DQ2    = +0.0098   # dq² (negligible, kept for completeness)

# charge_sheet: H-bond network is sole propagation channel
CSG_SHEET_DQ     = -0.8568   # dq alone is reversed!
CSG_SHEET_DQ_ENB = +0.8588   # dq × sheet_neighbor_frac (H-bond propagation)
CSG_SHEET_DQ_NCH = -0.1154   # dq × n_charged_neighbors (charge screening)

# charge_coil: long-range Coulomb dominates
CSG_COIL_DQ      = -0.0120   # dq direct (negligible)
CSG_COIL_LRC     = +0.7305   # long-range Coulomb (8-20Å)
CSG_COIL_DQ_BUR  = -0.5988   # dq × burial (buried coil shielded)

def compute_charge_subgate(mut_ss, dq, helix_nf, beta_nf, n_chg_nb, lr_coulomb, bur):
    """Gate 5: Charge Sub-Gate correction for phase='charge' mutations.

    Splits charge phase by mutation site SS to apply SS-specific physics:
      helix: helix dipole shielding dampens charge effect
      sheet: charge propagates only via H-bond network (sheet neighbors)
      coil: long-range Coulomb (8-20Å) dominates over local dq

    Returns: correction (float), info (dict)
    """
    if abs(dq) < 0.01:
        return 0.0, {'gate': 'no_charge_change'}

    if mut_ss == 'H':
        corr = CSG_HELIX_DQ * dq + CSG_HELIX_DQ_HNB * dq * helix_nf + CSG_HELIX_DQ2 * dq * dq
        sub = 'charge_helix'
    elif mut_ss == 'E':
        corr = CSG_SHEET_DQ * dq + CSG_SHEET_DQ_ENB * dq * beta_nf + CSG_SHEET_DQ_NCH * dq * n_chg_nb
        sub = 'charge_sheet'
    else:
        corr = CSG_COIL_DQ * dq + CSG_COIL_LRC * lr_coulomb + CSG_COIL_DQ_BUR * dq * bur
        sub = 'charge_coil'

    return corr, {'gate': sub, 'dq': dq, 'correction': corr}

def predict_mutation(backbone, atoms, all_heavy, pos, wa, ma, ss_map=None, pdb_features=None, pdb_path=None):
    """Predict ΔΔG for a single mutation. Returns dict or None.
    If ss_map is None, assigns SS internally.
    If pdb_features is None, computes them (cache externally for speed).
    pdb_path: path to PDB file for cofactor detection (v3.30)."""
    target=None
    for rnum,aa,xyz in atoms:
        if rnum==pos: target=xyz; break
    if target is None: return None
    neighbors=[aa for rnum,aa,xyz in atoms if rnum!=pos and np.linalg.norm(xyz-target)<8.0]
    if not neighbors: return None
    nc=len(neighbors)
    # Phase gate needs pdb_features first for nc_p90
    if ss_map is None:
        ss_map = assign_ss(backbone)
    if pdb_features is None:
        pdb_features = compute_pdb_features(atoms, ss_map, backbone, pdb_path=pdb_path)
    nc_p90 = pdb_features['nc_p90']
    bur=sigmoid(nc/nc_p90, 0.60, 10.0)
    n_polar=sum(1 for n in neighbors if n in POLAR_SET)
    polar_frac=n_polar/nc
    srnums=sorted(backbone.keys())
    d_CB_min=999.0;kappa=90.0;n_branch=0
    bb=backbone.get(pos,{})
    if wa=='G' and all(a in bb for a in ('N','CA','C')):
        vCB=place_virtual_CB(bb['N'],bb['CA'],bb['C'])
        d_CB_min=min((np.linalg.norm(vCB-xyz) for r,a,xyz in all_heavy if r!=pos),default=999)
        n_branch=sum(1 for r,a,xyz in atoms if r!=pos and a in ('V','I','T') and np.linalg.norm(target-xyz)<8.0)
        angles=[]
        try: idx=srnums.index(pos)
        except ValueError: idx=-1
        if idx>=0:
            for di in range(-1,2):
                ni=idx+di
                if 0<=ni<len(srnums):
                    rn=srnums[ni];bb2=backbone.get(rn,{})
                    if all(a in bb2 for a in ('N','CA','C')):
                        if ni>0:
                            rp=srnums[ni-1];bbp=backbone.get(rp,{})
                            if 'C' in bbp and rn-rp<=2:
                                try: angles.append(abs(dihedral_angle(bbp['C'],bb2['N'],bb2['CA'],bb2['C'])))
                                except: pass
                        if ni<len(srnums)-1:
                            rn2=srnums[ni+1];bbn=backbone.get(rn2,{})
                            if 'N' in bbn and rn2-rn<=2:
                                try: angles.append(abs(dihedral_angle(bb2['N'],bb2['CA'],bb2['C'],bbn['N'])))
                                except: pass
        if angles: kappa=np.mean(angles)
    # Phase gate (v3.12)
    ss = ss_map.get(pos, 'C')
    pdb_g = compute_pdb_gate(pdb_features['frac_E'], pdb_features['frac_H'],
                              pdb_features['core_frac'], pdb_features['ACO'],
                              pdb_features['charge_density'], pdb_features['N_res'])
    neigh_info_ss = [(rnum, aa, xyz, ss_map.get(rnum, 'C')) for rnum, aa, xyz in atoms
                     if rnum != pos and np.linalg.norm(xyz - target) < 8.0]
    n_E_neigh = sum(1 for _, _, _, s in neigh_info_ss if s == 'E')
    n_H_neigh = sum(1 for _, _, _, s in neigh_info_ss if s == 'H')
    beta_nf = n_E_neigh / max(1, nc)
    helix_nf = n_H_neigh / max(1, nc)
    micro_g = compute_micro_gate(ss, beta_nf, helix_nf, bur)
    phase = get_phase(pdb_g, micro_g)
    phase_original = phase  # preserve for Gate 4 cavity restoration
    # from_charged reclassification
    is_from_charged = wa in CHARGE_RECLASSIFY and ma not in CHARGE_RECLASSIFY
    if is_from_charged and phase in ('cavity', 'strain'):
        phase = 'charge'
    # Compute P_void from local lattice state
    large_aa = {'W','F','Y','I','L'}
    neigh_aa_list = [aa for rnum, aa, xyz in atoms if rnum != pos and np.linalg.norm(xyz - target) < 8.0]
    large_frac = sum(1 for aa in neigh_aa_list if aa in large_aa) / max(1, nc)
    if is_from_charged:
        dvol = AA_VOLUME.get(wa, 130) - AA_VOLUME.get(ma, 130)
        alpha = sigmoid(dvol, DVOL_X0_G4, DVOL_K_G4)
        p_void_cavity = compute_pvoid(phase_original, large_frac, polar_frac, helix_nf, beta_nf)
        p_void = (1.0 - alpha) * 1.0 + alpha * p_void_cavity
    else:
        p_void = compute_pvoid(phase, large_frac, polar_frac, helix_nf, beta_nf)
    ddg_pred,species=compute_ddg(wa,ma,bur,neighbors,d_CB_min=d_CB_min,kappa=kappa,n_branch=n_branch,polar_frac=polar_frac,cav_scale=p_void)
    # Backbone strain (v3.7)
    hpos = get_helix_position(ss_map, pos) if ss == 'H' else 'none'
    is_charge_intro = (ma in CHARGE_AA and wa not in CHARGE_AA)
    is_polar_intro = (ma in POLAR_AA_STRAIN and wa not in POLAR_AA_STRAIN)
    strain = calc_backbone_strain(wa, ma, ss, hpos, is_charge_intro, is_polar_intro)
    ddg_pred += strain
    if abs(strain) > 0.01:
        species += '+ST'
    # SCC Gate 1: Charge Intro Network Resilience (v3.14)
    com = np.mean([xyz for _, _, xyz in atoms], axis=0)
    scc_comp, scc_info = compute_scc_gate1(wa, ma, bur, atoms, pos, target,
                                            neighbors, polar_frac, com)
    ddg_pred += scc_comp
    if abs(scc_comp) > 0.01:
        species += '+SCC1'
    # Gate 1r: MT Rigidity Penalty (v3.20)              ← ★ 追加
    rigid_pen, rigid_info = compute_rigid_penalty(wa, ma, bur)
    ddg_pred -= rigid_pen  # SUBTRACT: more destabilizing
    if rigid_pen > 0.01:
        species += '+RIG'       
    # PCC Gate 2: Backbone Entropy — Gly Freedom (v3.15 PROVISIONAL)
    bb_comp, g2_info = compute_pcc_gate2(wa, ma, bur, helix_nf, beta_nf)
    ddg_pred += bb_comp
    if abs(bb_comp) > 0.01:
        species += '+G2'
    # SCC Gate 3a: Aromatic-loss Network Buffering (v3.16 PROVISIONAL)
    aro_comp, g3a_info = compute_scc_gate3a(wa, ma, bur, p_void)
    ddg_pred += aro_comp
    if abs(aro_comp) > 0.01:
        species += '+G3a'
    # SCC Gate 3b: Aromatic-gain π Bonus (v3.18 PROVISIONAL)
    n_cat = sum(1 for aa in neigh_aa_list if aa in {'K', 'R'})
    aro_bonus, g3b_info = compute_scc_gate3b(wa, ma, bur, n_cat)
    ddg_pred -= aro_bonus  # SUBTRACT: bonus makes pred more negative (more stable)
    if abs(aro_bonus) > 0.01:
        species += '+G3b'
    # Gate 5: Charge Sub-Gate (v3.30) — SS-specific charge correction
    # Heme/metal cofactors dominate charge environment → disable CSG (v3.30)
    has_cofactor = pdb_features.get('has_cofactor', False)
    csg_comp = 0.0
    csg_info = {'gate': 'not_charge_phase'}
    if phase == 'charge' and not has_cofactor:
        dq_csg = AA_CHARGE.get(ma, 0) - AA_CHARGE.get(wa, 0)
        # Count charged neighbors within 8Å
        charged_set = {'D', 'E', 'K', 'R', 'H'}
        n_chg_nb = sum(1 for _, aa2, _, _ in neigh_info_ss if aa2 in charged_set)
        # Long-range Coulomb (8-20Å)
        lr_coulomb = 0.0
        if abs(dq_csg) > 0.01:
            for rnum2, aa2, xyz2 in atoms:
                if rnum2 == pos: continue
                d = np.linalg.norm(xyz2 - target)
                if 8.0 <= d <= 20.0:
                    q2 = AA_CHARGE.get(aa2, 0)
                    lr_coulomb += dq_csg * q2 / max(d, 1.0)
        csg_comp, csg_info = compute_charge_subgate(
            ss, dq_csg, helix_nf, beta_nf, n_chg_nb, lr_coulomb, bur)
        ddg_pred += csg_comp
        if abs(csg_comp) > 0.01:
            species += '+CSG'
    elif phase == 'charge' and has_cofactor:
        csg_info = {'gate': 'cofactor_guard'}
    return {'ddg':ddg_pred,'species':species,'bur':bur,'nc':nc,'pf':polar_frac,
            'dry_bur':bur*(1-polar_frac),'ss':ss,'strain':strain,'phase':phase,
            'p_void':p_void,'scc_comp':scc_comp,'scc_gate':scc_info.get('gate','none'),
            'rigid_pen':rigid_pen,'rigid_gate':rigid_info.get('gate','none'),  # ★追加
            'bb_comp':bb_comp,'g2_gate':g2_info.get('gate','none'),
            'aro_comp':aro_comp,'g3a_gate':g3a_info.get('gate','none'),
            'aro_bonus':aro_bonus,'g3b_gate':g3b_info.get('gate','none'),
            'csg_comp':csg_comp,'csg_gate':csg_info.get('gate','none'),
            'features': {'vol_loss': max(0, AA_VOLUME.get(wa,130)-AA_VOLUME.get(ma,130))}}
