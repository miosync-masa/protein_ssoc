"""
Gate Explorer Orchestrator — 全候補を自動スキャンするメインエントリポイント

Usage:
  python orchestrator.py                    # scan all candidates
  python orchestrator.py --gate gate5       # explore specific gate
  python orchestrator.py --mirror-scan      # find all mirror structures
  python orchestrator.py --conflict-check   # run conflict matrix on current gates
"""
import sys, os
sys.path.insert(0, '/home/claude/gate_explorer')
sys.path.insert(0, '/home/claude')

import numpy as np
from collections import defaultdict
from pipeline import (load_dataset, filter_trigger, sweep, diagnose_k_zero,
                      auto_phase_separate, evaluate_global, CLASSES)
from reporter import (detect_sweet_spot, detect_collateral, detect_mirrors,
                      detect_double_count, auto_kpi_decision, generate_full_report)
from conflict_resolver import GateRegistry


def full_scan(records, verbose=True):
    """
    Scan ALL candidate mutation categories for gate potential.
    This is the "what should we explore next?" tool.
    """
    # Define all interesting mutation category pairs
    categories = {
        'aromatic': set('FWY'),
        'charged': set('DEKR'),
        'hydro_large': set('VLIMC'),
        'small': set('AG'),
        'polar': set('STNQH'),
        'proline': set('P'),
    }
    
    results = []
    
    for src_name, src_set in categories.items():
        for dst_name, dst_set in categories.items():
            if src_name == dst_name: continue
            
            sub = [r for r in records if r['wa'] in src_set and r['ma'] in dst_set]
            if len(sub) < 5: continue
            
            me = np.mean([r['error'] for r in sub])
            mae = np.mean(np.abs([r['error'] for r in sub]))
            under = sum(1 for r in sub if r['error'] < -2.0)
            over = sum(1 for r in sub if r['error'] > 2.0)
            outlier_pct = (under + over) / len(sub)
            
            # Signal strength = |mean_err| × sqrt(N) — like a t-statistic
            signal = abs(me) * np.sqrt(len(sub))
            
            results.append({
                'src': src_name, 'dst': dst_name,
                'N': len(sub), 'mean_err': me, 'mae': mae,
                'under': under, 'over': over,
                'outlier_pct': outlier_pct, 'signal': signal,
                'direction': 'UNDER' if me < -0.5 else ('OVER' if me > 0.5 else 'NORMAL'),
            })
    
    # Sort by signal strength
    results.sort(key=lambda x: -x['signal'])
    
    if verbose:
        print(f"{'='*90}")
        print(f"FULL CANDIDATE SCAN — {len(results)} mutation categories")
        print(f"{'='*90}")
        print(f"  {'src→dst':>25} {'N':>5} {'me':>7} {'MAE':>6} {'U':>4} {'O':>4} {'out%':>5} {'signal':>7} {'dir':>7}")
        for r in results:
            marker = ' ★' if r['signal'] > 5.0 and r['direction'] != 'NORMAL' else ''
            if r['signal'] > 3.0 and r['direction'] != 'NORMAL':
                marker = ' ★' if not marker else marker
            print(f"  {r['src']+'→'+r['dst']:>25} {r['N']:>5} {r['mean_err']:>+7.3f} "
                  f"{r['mae']:>6.3f} {r['under']:>4} {r['over']:>4} "
                  f"{r['outlier_pct']:>5.0%} {r['signal']:>7.1f} {r['direction']:>7}{marker}")
    
    return results


def mirror_scan(records, verbose=True):
    """Find all mirror structures in the data."""
    categories = {
        'aromatic': set('FWY'),
        'charged': set('DEKR'),
        'hydro_large': set('VLIMC'),
        'small': set('AG'),
        'polar': set('STNQH'),
    }
    
    mirrors = []
    checked = set()
    
    for name_a, set_a in categories.items():
        for name_b, set_b in categories.items():
            if name_a == name_b: continue
            pair = tuple(sorted([name_a, name_b]))
            if pair in checked: continue
            checked.add(pair)
            
            result = detect_mirrors(records, set_a, set_b)
            if result and result['is_mirror']:
                result['name_a'] = name_a
                result['name_b'] = name_b
                mirrors.append(result)
    
    if verbose and mirrors:
        print(f"\n{'='*90}")
        print(f"MIRROR STRUCTURES DETECTED: {len(mirrors)} pairs")
        print(f"{'='*90}")
        for m in sorted(mirrors, key=lambda x: -x['asymmetry']):
            a = m['a_to_b']; b = m['b_to_a']
            print(f"  🪞 {m['name_a']}↔{m['name_b']}  (asymmetry={m['asymmetry']:.2f})")
            print(f"     {m['name_a']}→{m['name_b']}: N={a['N']:>3}  me={a['mean_err']:+.3f}  {a['direction']}")
            print(f"     {m['name_b']}→{m['name_a']}: N={b['N']:>3}  me={b['mean_err']:+.3f}  {b['direction']}")
    
    return mirrors


def explore_candidate(records, src_set, dst_set, gate_name, 
                      env_filters=None, K_range=None, verbose=True):
    """
    Run full exploration pipeline on a candidate gate.
    """
    if K_range is None:
        K_range = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    
    # Filter
    triggered = [r for r in records if r['wa'] in src_set and r['ma'] in dst_set]
    if env_filters:
        for feat, (op, val) in env_filters.items():
            if op == '>=': triggered = [r for r in triggered if r.get(feat, 0) >= val]
            if op == '<=': triggered = [r for r in triggered if r.get(feat, 0) <= val]
    
    if len(triggered) < 5:
        if verbose: print(f"  ✗ {gate_name}: N={len(triggered)} too small")
        return None
    
    # Direction detection
    me = np.mean([r['error'] for r in triggered])
    direction = 'positive' if me < -0.5 else 'negative' if me > 0.5 else 'ambiguous'
    sign = 1.0 if direction == 'positive' else -1.0
    
    # Sweep
    def comp_fn(r, K):
        return sign * K
    
    sweep_results = sweep(triggered, comp_fn, K_range)
    
    # Diagnosis
    diagnosis = diagnose_k_zero(triggered, sweep_results)
    
    # Phase separation
    phase_by_class = auto_phase_separate(triggered, 'ma_classes')
    phase_by_phase = auto_phase_separate(triggered, 'phase')
    
    # Global eval (best K)
    best = max(sweep_results, key=lambda x: x['win_rate'])
    gate_filter = lambda r: r['wa'] in src_set and r['ma'] in dst_set
    if env_filters:
        orig_filter = gate_filter
        def gate_filter(r):
            if not orig_filter(r): return False
            for feat, (op, val) in env_filters.items():
                v = r.get(feat, 0)
                if op == '>=' and v < val: return False
                if op == '<=' and v > val: return False
            return True
    
    global_eval = evaluate_global(records, comp_fn, best['K'], gate_filter)
    
    # Collateral
    collateral = detect_collateral(records, comp_fn, best['K'], gate_filter)
    
    # KPI decision
    kpi_decision = auto_kpi_decision(sweep_results, global_eval)
    
    if verbose:
        # Burial sweep (sub-sweep)
        print(f"\n  ─── Burial Sub-sweep ───")
        for bur_min in [0.50, 0.60, 0.70, 0.80]:
            sub = [r for r in triggered if r['bur'] >= bur_min]
            if len(sub) < 3: continue
            sub_sweep = sweep(sub, comp_fn, [0.0, 1.0, 2.0, 3.0])
            best_sub = max(sub_sweep, key=lambda x: x['win_rate'])
            print(f"  bur≥{bur_min:.2f}: N={len(sub):>4}  best K={best_sub['K']:.1f}  "
                  f"MAE {best_sub['mae_before']:.2f}→{best_sub['mae_after']:.2f}  "
                  f"win={best_sub['win_rate']:.0%}")
    
    # Report
    report = generate_full_report(
        gate_name, sweep_results, diagnosis,
        phase_by_class, global_eval, collateral,
        kpi_decision=kpi_decision,
    )
    
    if verbose:
        print(report)
    
    return {
        'gate_name': gate_name,
        'N': len(triggered),
        'sweep_results': sweep_results,
        'diagnosis': diagnosis,
        'phase_by_class': phase_by_class,
        'phase_by_phase': phase_by_phase,
        'global_eval': global_eval,
        'collateral': collateral,
        'kpi_decision': kpi_decision,
        'report': report,
    }


# ── Main ──
if __name__ == '__main__':
    import ssoc_v318 as model
    
    print("Loading dataset...")
    records = load_dataset(
        model,
        '/home/claude/thermomut_verified_v3.json',
        '/home/claude/pdb_cache')
    print(f"Loaded {len(records)} records\n")
    
    # ── Full Scan ──
    scan_results = full_scan(records)
    
    # ── Mirror Scan ──
    mirrors = mirror_scan(records)
    
    # ── Top 3 Candidates: Auto-explore ──
    top_candidates = [r for r in scan_results if r['direction'] != 'NORMAL' and r['signal'] > 3.0][:5]
    
    print(f"\n{'='*90}")
    print(f"AUTO-EXPLORING TOP CANDIDATES")
    print(f"{'='*90}")
    
    categories = {
        'aromatic': set('FWY'),
        'charged': set('DEKR'),
        'hydro_large': set('VLIMC'),
        'small': set('AG'),
        'polar': set('STNQH'),
        'proline': set('P'),
    }
    
    for cand in top_candidates:
        src_set = categories[cand['src']]
        dst_set = categories[cand['dst']]
        name = f"Candidate: {cand['src']}→{cand['dst']}"
        print(f"\n{'─'*80}")
        explore_candidate(records, src_set, dst_set, name)

