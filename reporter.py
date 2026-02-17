"""
Reporter — 宝石抽出エンジン

Gem Detection Rules (human knowledge → automated):
  1. Sweet Spot Detection: hard threshold sweep の段差最大点
  2. Monotonicity Check: soft gate の単調性・飽和性
  3. Hard Boundary Discovery: "soft dies → hard is the boundary"
  4. Auto Subset-KPI Switch: global r flat → subset KPI priority
  5. Collateral Detection: phase別r悪化、二重計上疑い
  6. Mirror Structure: 鏡像ゲートの自動検出 (loss ↔ gain)
"""
import numpy as np
from collections import defaultdict


# ── Gem 1: Sweet Spot Detection ──
def detect_sweet_spot(sweep_results, metric='win_rate'):
    """Find the K value with largest improvement gradient."""
    if len(sweep_results) < 3:
        return None
    
    sorted_by_k = sorted(sweep_results, key=lambda x: x['K'])
    
    # Find maximum gradient in metric
    best_gradient = 0
    sweet_spot = None
    
    for i in range(1, len(sorted_by_k)):
        prev = sorted_by_k[i-1]
        curr = sorted_by_k[i]
        
        dk = curr['K'] - prev['K']
        if dk < 0.01: continue
        
        dm = curr[metric] - prev[metric]
        gradient = dm / dk
        
        if gradient > best_gradient:
            best_gradient = gradient
            sweet_spot = {
                'K_from': prev['K'], 'K_to': curr['K'],
                'metric_from': prev[metric], 'metric_to': curr[metric],
                'gradient': gradient,
            }
    
    # Also find the plateau (where adding more K stops helping)
    best_k = max(sorted_by_k, key=lambda x: x[metric])
    plateau = None
    for i in range(len(sorted_by_k)-1):
        curr = sorted_by_k[i]
        next_ = sorted_by_k[i+1]
        if abs(next_[metric] - curr[metric]) < 0.01 and curr[metric] > best_k[metric] * 0.95:
            plateau = curr['K']
            break
    
    return {
        'sweet_spot': sweet_spot,
        'optimal_K': best_k['K'],
        'optimal_value': best_k[metric],
        'plateau_K': plateau,
    }


# ── Gem 2: Soft Gate Death Detector ──
def detect_soft_death(sweep_results_by_threshold):
    """
    If continuous (soft) gating produces K=0 optimal but hard threshold
    produces K>0 optimal, the boundary IS the physics.
    
    Input: dict of {threshold: sweep_results}
    """
    boundaries = []
    
    for thresh, results in sorted(sweep_results_by_threshold.items()):
        best = max(results, key=lambda x: x['win_rate'])
        if best['K'] > 0.01 and best['win_rate'] > 0.55:
            boundaries.append({
                'threshold': thresh,
                'optimal_K': best['K'],
                'win_rate': best['win_rate'],
                'N': best['N'],
            })
    
    if not boundaries:
        return {'diagnosis': 'no_hard_boundary_found'}
    
    # Find sharpest boundary (highest win rate with enough N)
    best_boundary = max(boundaries, key=lambda x: x['win_rate'] * min(1.0, x['N']/20))
    
    return {
        'diagnosis': 'hard_boundary_detected',
        'boundary': best_boundary,
        'all_boundaries': boundaries,
    }


# ── Gem 3: Collateral Damage Detector ──
def detect_collateral(records, comp_fn, K, gate_filter):
    """
    Check for phase-specific r degradation when gate fires.
    Returns list of damaged phases.
    """
    damages = []
    
    for phase in ['cavity', 'strain', 'charge']:
        sub = [r for r in records if r['phase'] == phase]
        if len(sub) < 10: continue
        
        e = np.array([r['exp'] for r in sub])
        p_before = np.array([r['pred'] for r in sub])
        p_after = np.array([
            r['pred'] + (comp_fn(r, K) if gate_filter(r) else 0.0)
            for r in sub
        ])
        
        r_before = np.corrcoef(e, p_before)[0,1]
        r_after = np.corrcoef(e, p_after)[0,1]
        delta_r = r_after - r_before
        
        n_fired = sum(1 for r in sub if gate_filter(r))
        
        if delta_r < -0.005 and n_fired > 0:
            damages.append({
                'phase': phase,
                'r_before': r_before, 'r_after': r_after,
                'delta_r': delta_r, 'n_fired': n_fired,
                'severity': 'critical' if delta_r < -0.02 else 'warning',
            })
    
    return damages


# ── Gem 4: Mirror Structure Detector ──
def detect_mirrors(records, class_a_set, class_b_set):
    """
    Detect mirror structure: A→B vs B→A showing opposite error patterns.
    Classic example: Aro→Charged (UNDER) vs Charged→Aro (OVER)
    """
    a_to_b = [r for r in records if r['wa'] in class_a_set and r['ma'] in class_b_set]
    b_to_a = [r for r in records if r['wa'] in class_b_set and r['ma'] in class_a_set]
    
    if len(a_to_b) < 5 or len(b_to_a) < 5:
        return None
    
    me_ab = np.mean([r['error'] for r in a_to_b])
    me_ba = np.mean([r['error'] for r in b_to_a])
    
    is_mirror = (me_ab * me_ba < 0) and (abs(me_ab) > 0.5 or abs(me_ba) > 0.5)
    
    return {
        'class_a': sorted(class_a_set),
        'class_b': sorted(class_b_set),
        'a_to_b': {'N': len(a_to_b), 'mean_err': me_ab,
                   'direction': 'UNDER' if me_ab < -0.5 else ('OVER' if me_ab > 0.5 else 'NORMAL')},
        'b_to_a': {'N': len(b_to_a), 'mean_err': me_ba,
                   'direction': 'UNDER' if me_ba < -0.5 else ('OVER' if me_ba > 0.5 else 'NORMAL')},
        'is_mirror': is_mirror,
        'asymmetry': abs(me_ab - me_ba),
    }


# ── Gem 5: Double-Count Detector ──
def detect_double_count(records, gate_a_filter, gate_b_filter):
    """
    Find mutations where two gates would both fire.
    Potential double-counting if both compensate in same direction.
    """
    both_fire = [r for r in records if gate_a_filter(r) and gate_b_filter(r)]
    only_a = [r for r in records if gate_a_filter(r) and not gate_b_filter(r)]
    only_b = [r for r in records if not gate_a_filter(r) and gate_b_filter(r)]
    
    return {
        'both_fire': len(both_fire),
        'only_a': len(only_a),
        'only_b': len(only_b),
        'overlap_pct': len(both_fire) / max(1, len(both_fire) + len(only_a) + len(only_b)),
        'mutations': [r['mc'] for r in both_fire[:10]],  # sample
    }


# ── Gem 6: Auto Subset-KPI Switch ──
def auto_kpi_decision(sweep_results, global_eval, N_threshold=100):
    """
    Decide whether to use global r or subset KPI as primary metric.
    """
    N = sweep_results[0]['N'] if sweep_results else 0
    delta_r = global_eval.get('delta_r', 0) if global_eval else 0
    best_win = max(r['win_rate'] for r in sweep_results) if sweep_results else 0
    
    if N >= N_threshold and abs(delta_r) > 0.0005:
        return {
            'decision': 'global_r',
            'reason': f'N={N} sufficient, Δr={delta_r:+.4f} significant',
        }
    elif N < N_threshold and best_win > 0.60:
        return {
            'decision': 'subset_kpi',
            'reason': f'N={N} too small for global r, but win_rate={best_win:.0%} strong',
        }
    elif best_win < 0.55:
        return {
            'decision': 'reject',
            'reason': f'Win rate {best_win:.0%} insufficient',
        }
    else:
        return {
            'decision': 'subset_kpi',
            'reason': f'N={N}, Δr={delta_r:+.4f} marginal, subset KPI preferred',
        }


# ── Full Report Generator ──
def generate_full_report(gate_name, sweep_results, diagnosis, phase_info,
                         global_eval, collateral, mirror_info=None,
                         double_counts=None, kpi_decision=None):
    """Generate comprehensive gate exploration report with gem highlights."""
    lines = []
    lines.append(f"{'='*80}")
    lines.append(f"GATE EXPLORATION REPORT: {gate_name}")
    lines.append(f"{'='*80}")
    
    # ── Status Decision ──
    if kpi_decision:
        lines.append(f"\n  📊 KPI Decision: {kpi_decision['decision'].upper()}")
        lines.append(f"     Reason: {kpi_decision['reason']}")
    
    # ── Sweep Summary ──
    if sweep_results:
        best_win = max(sweep_results, key=lambda x: x['win_rate'])
        best_mae = min(sweep_results, key=lambda x: x['mae_after'])
        
        lines.append(f"\n  ─── Sweep Summary ───")
        lines.append(f"  Best win rate: K={best_win['K']:.1f}  {best_win['win_rate']:.0%}  "
                     f"N={best_win['N']}  MAE {best_win['mae_before']:.2f}→{best_win['mae_after']:.2f}")
        lines.append(f"  Best MAE:      K={best_mae['K']:.1f}  MAE {best_mae['mae_before']:.2f}→{best_mae['mae_after']:.2f}")
        
        # Sweet spot
        ss = detect_sweet_spot(sweep_results)
        if ss and ss['sweet_spot']:
            sp = ss['sweet_spot']
            lines.append(f"  💎 Sweet spot: K={sp['K_from']:.1f}→{sp['K_to']:.1f} "
                        f"(gradient={sp['gradient']:.2f})")
            if ss['plateau_K']:
                lines.append(f"     Plateau at K={ss['plateau_K']:.1f} — diminishing returns beyond")
    
    # ── Diagnosis ──
    if diagnosis:
        lines.append(f"\n  ─── Diagnosis ───")
        lines.append(f"  {diagnosis.get('diagnosis', 'N/A')}")
        if diagnosis.get('phase_mixing'):
            lines.append(f"  ⚠️ PHASE MIXING DETECTED")
            for cls, info in diagnosis.get('phase_details', {}).items():
                lines.append(f"     {cls:>25}: N={info['N']:>3}  me={info['mean_err']:+.3f}")
            lines.append(f"  → {diagnosis['recommendation']}")
    
    # ── Phase Separation ──
    if phase_info:
        lines.append(f"\n  ─── Phase Separation ───")
        for key, info in sorted(phase_info.items()):
            marker = ' ★' if info['direction'] != 'NORMAL' else ''
            lines.append(f"  {str(key):>35}: N={info['N']:>3}  me={info['mean_err']:+.3f}  "
                        f"{info['direction']}{marker}")
    
    # ── Global Impact ──
    if global_eval:
        lines.append(f"\n  ─── Global r Impact ───")
        lines.append(f"  r: {global_eval['r_before']:.4f} → {global_eval['r_after']:.4f}  "
                     f"Δ={global_eval['delta_r']:+.4f}  N_fired={global_eval['N_fired']}")
    
    # ── Collateral Damage ──
    if collateral:
        lines.append(f"\n  ─── ⚠️ Collateral Damage ───")
        for d in collateral:
            lines.append(f"  {d['phase']:>8}: r {d['r_before']:.4f}→{d['r_after']:.4f}  "
                        f"Δ={d['delta_r']:+.4f}  [{d['severity']}]  N_fired={d['n_fired']}")
    
    # ── Mirror Structure ──
    if mirror_info and mirror_info.get('is_mirror'):
        lines.append(f"\n  ─── 🪞 Mirror Structure Detected ───")
        a = mirror_info['a_to_b']
        b = mirror_info['b_to_a']
        lines.append(f"  A→B: N={a['N']}  me={a['mean_err']:+.3f}  ({a['direction']})")
        lines.append(f"  B→A: N={b['N']}  me={b['mean_err']:+.3f}  ({b['direction']})")
        lines.append(f"  Asymmetry: {mirror_info['asymmetry']:.2f}")
    
    # ── Double Counting ──
    if double_counts and double_counts['both_fire'] > 0:
        lines.append(f"\n  ─── ⚠️ Double-Count Risk ───")
        lines.append(f"  Overlap: {double_counts['both_fire']} mutations "
                     f"({double_counts['overlap_pct']:.0%})")
        if double_counts['mutations']:
            lines.append(f"  Examples: {', '.join(double_counts['mutations'][:5])}")
    
    # ── Final Recommendation ──
    lines.append(f"\n  {'='*60}")
    if kpi_decision:
        dec = kpi_decision['decision']
        if dec == 'global_r' and global_eval and global_eval['delta_r'] > 0.0005:
            lines.append(f"  ★ CONFIRMED — r improvement Δ={global_eval['delta_r']:+.4f}")
        elif dec == 'subset_kpi' and sweep_results:
            bw = max(sweep_results, key=lambda x: x['win_rate'])
            if bw['win_rate'] >= 0.65:
                lines.append(f"  ★ PROVISIONAL — win rate {bw['win_rate']:.0%}, deploy with subset KPI monitoring")
            else:
                lines.append(f"  △ CANDIDATE — win rate {bw['win_rate']:.0%}, needs more data or refinement")
        else:
            lines.append(f"  ✗ REJECT or HOLD — insufficient evidence")
    
    if collateral:
        lines.append(f"  ⚠️ NOTE: Collateral damage detected in {len(collateral)} phase(s)")
    if double_counts and double_counts['both_fire'] > 0:
        lines.append(f"  ⚠️ NOTE: {double_counts['both_fire']} mutations at risk of double-counting")
    
    return '\n'.join(lines)


# ── Test ──
if __name__ == '__main__':
    # Demonstrate gem detection
    mock_sweep = [
        {'K': 0.0, 'N': 50, 'mae_before': 1.90, 'mae_after': 1.90, 'win_rate': 0.50,
         'me_before': 1.74, 'me_after': 1.74, 'over_before': 17, 'over_after': 17,
         'under_before': 0, 'under_after': 0},
        {'K': 0.5, 'N': 50, 'mae_before': 1.90, 'mae_after': 1.67, 'win_rate': 0.80,
         'me_before': 1.74, 'me_after': 1.37, 'over_before': 17, 'over_after': 14,
         'under_before': 0, 'under_after': 0},
        {'K': 1.0, 'N': 50, 'mae_before': 1.90, 'mae_after': 1.47, 'win_rate': 0.80,
         'me_before': 1.74, 'me_after': 1.00, 'over_before': 17, 'over_after': 11,
         'under_before': 0, 'under_after': 1},
        {'K': 1.5, 'N': 50, 'mae_before': 1.90, 'mae_after': 1.30, 'win_rate': 0.78,
         'me_before': 1.74, 'me_after': 0.63, 'over_before': 17, 'over_after': 7,
         'under_before': 0, 'under_after': 2},
    ]
    
    ss = detect_sweet_spot(mock_sweep)
    print("Sweet Spot Detection:")
    print(f"  Optimal K: {ss['optimal_K']}")
    print(f"  Sweet spot: {ss['sweet_spot']}")
    print(f"  Plateau: {ss['plateau_K']}")
    
    kpi = auto_kpi_decision(mock_sweep, {'delta_r': 0.0029}, N_threshold=100)
    print(f"\nKPI Decision: {kpi['decision']}")
    print(f"  Reason: {kpi['reason']}")
    
    report = generate_full_report(
        'Gate 3b: Aro Gain (mock)', mock_sweep,
        diagnosis={'diagnosis': 'K_nonzero_optimal'},
        phase_info=None,
        global_eval={'r_before': 0.5305, 'r_after': 0.5334, 'delta_r': 0.0029, 'N_fired': 50},
        collateral=[],
        kpi_decision=kpi,
    )
    print(f"\n{report}")

