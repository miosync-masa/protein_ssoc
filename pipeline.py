"""
SSOC Gate Explorer Pipeline — Automated Gate Discovery Engine

Usage:
  python pipeline.py --gate <gate_name> --mode sweep|diagnose|phase_split|report
  python pipeline.py --discover   # scan all candidates

Architecture:
  1. load_data()       → PDB cache + mutation records with all features
  2. filter_trigger()  → select mutations matching gate trigger
  3. sweep()           → hard threshold × K sweep with dual KPI
  4. diagnose()        → K=0 problem detection, phase mixing analysis
  5. phase_separate()  → split by mut_class/wt_class, re-sweep
  6. evaluate()        → global + subset KPI with conflict check
  7. report()          → extract gems, causal summary
"""
import sys, os, re, json, math
import numpy as np
from collections import defaultdict

sys.path.insert(0, '/home/claude')

# ── Residue Classes ──
CLASSES = {
    'aromatic': set('FWY'),
    'charged': set('DEKR'),
    'hydrophobic_large': set('VLIMC'),
    'small': set('AG'),
    'polar': set('STNQH'),
    'small_polar_pro': set('AGSTNQHP'),
    'all_aa': set('ACDEFGHIKLMNPQRSTVWY'),
}

def classify_aa(aa):
    """Return all classes an amino acid belongs to."""
    return [cls for cls, members in CLASSES.items() if aa in members]

# ── Data Loading ──
def load_dataset(model_module, dataset_path, pdb_dir, suspicious=None):
    """Load dataset and compute predictions + features for all mutations."""
    if suspicious is None:
        suspicious = {'2GNQ', '1ITM'}
    
    with open(dataset_path) as f:
        data = json.load(f)
    
    by_pdb = defaultdict(list)
    for e in data:
        if e['pdb'].upper() in suspicious: continue
        _m = e.get('method','').strip().lower()
        if not ('gdn' in _m or 'thermal' in _m): continue
        by_pdb[e['pdb'].upper()].append(e)
    
    pdb_cache = {}
    for pid, ents in by_pdb.items():
        pf = os.path.join(pdb_dir, f"{pid.lower()}.pdb")
        if not os.path.exists(pf):
            pf = os.path.join(pdb_dir, f"{pid.upper()}.pdb")
        if not os.path.exists(pf): continue
        ch = ents[0].get('chain','A')
        try:
            backbone, atoms, all_heavy = model_module.parse_pdb(pf, ch)
        except: continue
        if len(atoms) < 5: continue
        ss_map = model_module.assign_ss(backbone)
        pdb_features = model_module.compute_pdb_features(atoms, ss_map, backbone)
        pdb_cache[pid] = {
            'backbone':backbone, 'atoms':atoms, 'all_heavy':all_heavy,
            'ents':ents, 'ss_map':ss_map, 'pdb_features':pdb_features,
        }
    
    records = []
    for pid, info in pdb_cache.items():
        for e in info['ents']:
            mc = e['mutation_code']
            mt_ = re.match(r'([A-Z])(\d+)([A-Z])', mc)
            if not mt_: continue
            wa, pos, ma = mt_.group(1), int(mt_.group(2)), mt_.group(3)
            if wa == 'X' or ma == 'X': continue
            
            result = model_module.predict_mutation(
                info['backbone'], info['atoms'], info['all_heavy'],
                pos, wa, ma, ss_map=info['ss_map'],
                pdb_features=info['pdb_features'])
            if result is None: continue
            
            # Compute neighbor features
            target = None
            for rnum, aa, xyz in info['atoms']:
                if rnum == pos: target = xyz; break
            if target is None: continue
            
            neighbors = [(rnum, aa, xyz) for rnum, aa, xyz in info['atoms']
                        if rnum != pos and np.linalg.norm(xyz - target) < 8.0]
            neigh_aa = [aa for _, aa, _ in neighbors]
            
            records.append({
                'exp': float(e['ddg']),
                'pred': result['ddg'],
                'error': result['ddg'] - float(e['ddg']),
                'wa': wa, 'ma': ma, 'mc': mc, 'pid': pid, 'pos': pos,
                'phase': result['phase'],
                'bur': result['bur'],
                'p_void': result['p_void'],
                'ss': result['ss'],
                'n_aro': sum(1 for aa in neigh_aa if aa in CLASSES['aromatic']),
                'n_cat': sum(1 for aa in neigh_aa if aa in {'K','R'}),
                'n_charged': sum(1 for aa in neigh_aa if aa in CLASSES['charged']),
                'nc': len(neighbors),
                'wa_classes': classify_aa(wa),
                'ma_classes': classify_aa(ma),
            })
    
    return records

# ── Trigger Filter ──
def filter_trigger(records, wt_classes=None, mut_classes=None, 
                   env_filters=None, exclusions=None):
    """Filter records by gate trigger conditions."""
    filtered = []
    for r in records:
        # WT class check
        if wt_classes:
            wt_ok = any(cls in r['wa_classes'] for cls in wt_classes)
            if not wt_ok: continue
        
        # Mutant class check
        if mut_classes:
            ma_ok = any(cls in r['ma_classes'] for cls in mut_classes)
            if not ma_ok: continue
        
        # Exclusions (phase separation)
        if exclusions:
            excluded = False
            for excl in exclusions:
                if 'mut_class' in excl:
                    if any(cls in r['ma_classes'] for cls in excl['mut_class']):
                        excluded = True; break
                if 'wt_class' in excl:
                    if any(cls in r['wa_classes'] for cls in excl['wt_class']):
                        excluded = True; break
            if excluded: continue
        
        # Environment filters
        if env_filters:
            env_ok = True
            for feat, (op, val) in env_filters.items():
                rv = r.get(feat, 0)
                if op == '>=' and rv < val: env_ok = False; break
                if op == '<=' and rv > val: env_ok = False; break
                if op == '>' and rv <= val: env_ok = False; break
                if op == '<' and rv >= val: env_ok = False; break
            if not env_ok: continue
        
        filtered.append(r)
    return filtered

# ── Sweep Engine ──
def sweep(records, comp_fn, K_range, label=""):
    """Sweep K values and compute dual KPI (global + subset)."""
    if len(records) < 3:
        return []
    
    results = []
    for K in K_range:
        err_before = [r['error'] for r in records]
        err_after = [r['error'] + comp_fn(r, K) for r in records]
        
        me_b = np.mean(err_before); me_a = np.mean(err_after)
        mae_b = np.mean(np.abs(err_before)); mae_a = np.mean(np.abs(err_after))
        p95_b = np.percentile(np.abs(err_before), 95)
        p95_a = np.percentile(np.abs(err_after), 95)
        under_b = sum(1 for e in err_before if e < -2.0)
        under_a = sum(1 for e in err_after if e < -2.0)
        over_b = sum(1 for e in err_before if e > 2.0)
        over_a = sum(1 for e in err_after if e > 2.0)
        win = sum(1 for b, a in zip(err_before, err_after) if abs(a) < abs(b))
        
        results.append({
            'K': K, 'N': len(records), 'label': label,
            'me_before': me_b, 'me_after': me_a,
            'mae_before': mae_b, 'mae_after': mae_a,
            'p95_before': p95_b, 'p95_after': p95_a,
            'under_before': under_b, 'under_after': under_a,
            'over_before': over_b, 'over_after': over_a,
            'outlier_before': under_b + over_b,
            'outlier_after': under_a + over_a,
            'win': win, 'win_rate': win / len(records),
        })
    return results

# ── Diagnostics ──
def diagnose_k_zero(records, sweep_results):
    """If K=0 is optimal, diagnose why (phase mixing, N too small, etc.)."""
    best = min(sweep_results, key=lambda x: x['mae_after'])
    
    report = {
        'best_K': best['K'],
        'best_mae': best['mae_after'],
        'N': len(records),
        'mean_err': np.mean([r['error'] for r in records]),
    }
    
    if best['K'] > 0.01:
        report['diagnosis'] = 'K_nonzero_optimal'
        return report
    
    # K=0 optimal — diagnose
    report['diagnosis'] = 'K_zero_problem'
    
    # Check 1: Phase mixing by mutant class
    by_ma_class = defaultdict(list)
    for r in records:
        for cls in r['ma_classes']:
            by_ma_class[cls].append(r)
    
    phase_mix = {}
    for cls, sub in by_ma_class.items():
        if len(sub) < 3: continue
        me = np.mean([r['error'] for r in sub])
        phase_mix[cls] = {'N': len(sub), 'mean_err': me}
    
    # Detect mixing: some classes UNDER, others OVER
    has_under = any(v['mean_err'] < -1.0 for v in phase_mix.values())
    has_over = any(v['mean_err'] > 0.5 for v in phase_mix.values())
    
    if has_under and has_over:
        report['phase_mixing'] = True
        report['phase_details'] = phase_mix
        report['recommendation'] = 'Phase separation needed: split by mutant class'
    else:
        # Check 2: N too small
        if len(records) < 100:
            report['phase_mixing'] = False
            report['low_N'] = True
            report['recommendation'] = f'N={len(records)} too small for global r. Use subset KPI.'
        else:
            report['recommendation'] = 'Unknown cause. Check environment feature distributions.'
    
    return report

# ── Phase Separation ──
def auto_phase_separate(records, split_by='ma_classes'):
    """Automatically propose phase separation based on error distribution."""
    groups = defaultdict(list)
    for r in records:
        if split_by == 'ma_classes':
            key = tuple(sorted(r['ma_classes']))
        elif split_by == 'wa_classes':
            key = tuple(sorted(r['wa_classes']))
        elif split_by == 'phase':
            key = (r['phase'],)
        else:
            key = (r.get(split_by, 'unknown'),)
        groups[key].append(r)
    
    phases = {}
    for key, sub in groups.items():
        if len(sub) < 3: continue
        me = np.mean([r['error'] for r in sub])
        phases[key] = {
            'N': len(sub), 'mean_err': me,
            'under': sum(1 for r in sub if r['error'] < -2.0),
            'over': sum(1 for r in sub if r['error'] > 2.0),
            'direction': 'UNDER' if me < -0.5 else ('OVER' if me > 0.5 else 'NORMAL'),
        }
    return phases

# ── Evaluation ──
def evaluate_global(all_records, comp_fn, K, gate_filter):
    """Evaluate global r impact of a gate configuration."""
    e_all = np.array([r['exp'] for r in all_records])
    p_before = np.array([r['pred'] for r in all_records])
    p_after = np.array([
        r['pred'] + (comp_fn(r, K) if gate_filter(r) else 0.0)
        for r in all_records
    ])
    
    r_before = np.corrcoef(e_all, p_before)[0, 1]
    r_after = np.corrcoef(e_all, p_after)[0, 1]
    
    return {
        'r_before': r_before, 'r_after': r_after,
        'delta_r': r_after - r_before,
        'N_fired': sum(1 for r in all_records if gate_filter(r)),
    }

# ── Report Generator ──
def generate_report(gate_name, sweep_results, diagnosis, phase_info, global_eval):
    """Generate human-readable gate exploration report."""
    lines = []
    lines.append(f"{'='*80}")
    lines.append(f"GATE EXPLORATION REPORT: {gate_name}")
    lines.append(f"{'='*80}")
    
    # Best configuration
    best_subset = max(sweep_results, key=lambda x: x['win_rate'])
    best_mae = min(sweep_results, key=lambda x: x['mae_after'])
    
    lines.append(f"\n  Best by win rate: K={best_subset['K']:.1f}  win={best_subset['win_rate']:.0%}  MAE {best_subset['mae_before']:.2f}→{best_subset['mae_after']:.2f}")
    lines.append(f"  Best by MAE:      K={best_mae['K']:.1f}  MAE {best_mae['mae_before']:.2f}→{best_mae['mae_after']:.2f}")
    
    # Diagnosis
    if diagnosis:
        lines.append(f"\n  Diagnosis: {diagnosis.get('diagnosis', 'N/A')}")
        if diagnosis.get('phase_mixing'):
            lines.append(f"  → Phase mixing detected. Recommendation: {diagnosis['recommendation']}")
        if diagnosis.get('low_N'):
            lines.append(f"  → Low N ({diagnosis['N']}). Use subset KPI.")
    
    # Global impact
    if global_eval:
        lines.append(f"\n  Global r impact: {global_eval['r_before']:.4f}→{global_eval['r_after']:.4f} (Δ={global_eval['delta_r']:+.4f})")
        lines.append(f"  N_fired: {global_eval['N_fired']}")
    
    # Decision
    if global_eval and global_eval['delta_r'] > 0.0005:
        lines.append(f"\n  ★ RECOMMENDATION: CONFIRMED — r improvement detected")
    elif best_subset['win_rate'] >= 0.65:
        lines.append(f"\n  ★ RECOMMENDATION: PROVISIONAL — strong subset KPI, r too small to move")
    elif best_subset['win_rate'] >= 0.55:
        lines.append(f"\n  △ RECOMMENDATION: CANDIDATE — marginal signal, needs more data")
    else:
        lines.append(f"\n  ✗ RECOMMENDATION: REJECT — insufficient signal")
    
    return '\n'.join(lines)

# ── Main Pipeline ──
def run_gate_exploration(gate_config, all_records, K_range=None):
    """Run full exploration pipeline for a gate configuration."""
    if K_range is None:
        K_range = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    
    # Step 1: Filter by trigger
    triggered = filter_trigger(
        all_records,
        wt_classes=gate_config.get('wt_classes'),
        mut_classes=gate_config.get('mut_classes'),
        env_filters=gate_config.get('env_filters'),
        exclusions=gate_config.get('exclusions'),
    )
    
    if len(triggered) < 5:
        return {'status': 'insufficient_data', 'N': len(triggered)}
    
    # Step 2: Sweep
    direction = gate_config.get('direction', 'positive')
    sign = 1.0 if direction == 'positive' else -1.0
    
    def comp_fn(r, K):
        return sign * K * gate_config.get('aa_weight', {}).get(
            r['wa'] if direction == 'positive' else r['ma'], 0.7)
    
    results = sweep(triggered, comp_fn, K_range)
    
    # Step 3: Diagnose
    diagnosis = diagnose_k_zero(triggered, results)
    
    # Step 4: Phase separation (if needed)
    phase_info = None
    if diagnosis.get('phase_mixing'):
        phase_info = auto_phase_separate(triggered)
    
    # Step 5: Global evaluation (best K by win rate)
    best = max(results, key=lambda x: x['win_rate'])
    gate_filter = lambda r: any(cls in r.get('wa_classes',[]) for cls in gate_config.get('wt_classes',[])) and \
                            any(cls in r.get('ma_classes',[]) for cls in gate_config.get('mut_classes',[]))
    global_eval = evaluate_global(all_records, comp_fn, best['K'], gate_filter)
    
    # Step 6: Report
    report = generate_report(
        gate_config.get('name', 'unnamed'),
        results, diagnosis, phase_info, global_eval)
    
    return {
        'status': 'completed',
        'triggered_N': len(triggered),
        'sweep_results': results,
        'diagnosis': diagnosis,
        'phase_info': phase_info,
        'global_eval': global_eval,
        'report': report,
    }


# ── Quick Test ──
if __name__ == '__main__':
    print("Gate Explorer Pipeline — loaded successfully")
    print(f"  Residue classes: {list(CLASSES.keys())}")
    print(f"  Functions: load_dataset, filter_trigger, sweep, diagnose_k_zero,")
    print(f"             auto_phase_separate, evaluate_global, run_gate_exploration")
    
    # Demo: run on existing data with a candidate gate
    import ssoc_v318 as model
    records = load_dataset(
        model,
        '/home/claude/thermomut_verified_v3.json',
        '/home/claude/pdb_cache')
    print(f"\n  Loaded {len(records)} records")
    
    # Test Gate 4 candidate: Pro introduction
    gate4 = {
        'name': 'Gate 4: Proline Introduction',
        'wt_classes': ['all_aa'],
        'mut_classes': None,  # any → P
        'env_filters': {},
        'exclusions': [],
        'direction': 'positive',
        'aa_weight': {},
    }
    # Quick filter: X→P only
    pro_gain = [r for r in records if r['ma'] == 'P']
    pro_loss = [r for r in records if r['wa'] == 'P']
    
    print(f"\n  X→P (Pro gain): N={len(pro_gain)}  mean_err={np.mean([r['error'] for r in pro_gain]):+.3f}")
    print(f"  P→X (Pro loss): N={len(pro_loss)}  mean_err={np.mean([r['error'] for r in pro_loss]):+.3f}")
    
    if len(pro_gain) >= 5:
        phases = auto_phase_separate(pro_gain, 'phase')
        print(f"\n  X→P by phase:")
        for k, v in sorted(phases.items()):
            print(f"    {str(k):>20}: N={v['N']:>3}  mean_err={v['mean_err']:+.3f}  dir={v['direction']}")
    
    # Test Gate 6 candidate: Salt bridge loss
    salt_loss = [r for r in records if r['wa'] in CLASSES['charged'] and r['ma'] not in CLASSES['charged']]
    salt_loss_buried = [r for r in salt_loss if r['bur'] >= 0.70]
    
    print(f"\n  Charged→NonCharged (salt bridge loss): N={len(salt_loss)}  mean_err={np.mean([r['error'] for r in salt_loss]):+.3f}")
    if len(salt_loss_buried) >= 5:
        print(f"  ...buried (≥0.70): N={len(salt_loss_buried)}  mean_err={np.mean([r['error'] for r in salt_loss_buried]):+.3f}")
        phases = auto_phase_separate(salt_loss_buried, 'ma_classes')
        print(f"  By mutant class:")
        for k, v in sorted(phases.items()):
            print(f"    {str(k):>40}: N={v['N']:>3}  mean_err={v['mean_err']:+.3f}  dir={v['direction']}")
    
    # Test Gate 7: Large→Small hydrophobic
    large_to_small = [r for r in records 
                      if r['wa'] in CLASSES['hydrophobic_large'] and r['ma'] in CLASSES['small']]
    if len(large_to_small) >= 5:
        print(f"\n  LargeHydro→Small: N={len(large_to_small)}  mean_err={np.mean([r['error'] for r in large_to_small]):+.3f}")
        phases = auto_phase_separate(large_to_small, 'phase')
        for k, v in sorted(phases.items()):
            print(f"    {str(k):>20}: N={v['N']:>3}  mean_err={v['mean_err']:+.3f}  dir={v['direction']}")

