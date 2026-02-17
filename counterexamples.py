"""
Counterexample Engine — 反例抽出モジュール

Three classes of counterexamples:
  1. gate_expected_but_failed: 発火条件を満たすのに効かない（悪化した）
     → 相境界が間違っている、または未知の物理が干渉している
  2. unexpected_improvement: Gate外なのに残差が改善方向にある
     → Gateの適用範囲が狭すぎる、または別の物理が同方向に効いている
  3. phase_contamination: 相境界を跨いでるデータ
     → UNDER/OVERが混在する群の中に、反対方向の外れ値がある

Each counterexample is a seed for the next gate.
"""
import numpy as np
from collections import defaultdict


# ── Class 1: Gate Expected But Failed ──
def find_gate_failures(records, gate_filter, comp_fn, K, 
                       failure_threshold=0.5, min_worsening=0.3):
    """
    Find mutations where the gate fires but makes things WORSE.
    
    These are the most valuable counterexamples:
    the gate's physics doesn't apply here → WHY?
    
    Args:
        records: all mutation records
        gate_filter: function(record) → bool (does gate fire?)
        comp_fn: function(record, K) → compensation value
        K: gate constant
        failure_threshold: minimum |error_after| to count as failure
        min_worsening: minimum increase in |error| to count as worsening
    
    Returns:
        list of failure records with analysis
    """
    failures = []
    
    for r in records:
        if not gate_filter(r):
            continue
        
        err_before = r['error']
        comp = comp_fn(r, K)
        err_after = err_before + comp
        
        # Worsened: |error| increased
        delta_abs = abs(err_after) - abs(err_before)
        
        if delta_abs > min_worsening:
            # Classify failure mode
            if err_before > 0 and comp > 0:
                mode = 'overshoot_positive'  # was OVER, pushed more OVER
            elif err_before < 0 and comp < 0:
                mode = 'overshoot_negative'  # was UNDER, pushed more UNDER
            elif err_before > 0 and comp < 0 and err_after < -failure_threshold:
                mode = 'overcorrection'  # was OVER, corrected past zero into UNDER
            elif err_before < 0 and comp > 0 and err_after > failure_threshold:
                mode = 'overcorrection'  # was UNDER, corrected past zero into OVER
            else:
                mode = 'sign_flip'  # correction flipped the sign
            
            failures.append({
                **r,
                'err_before': err_before,
                'err_after': err_after,
                'comp': comp,
                'delta_abs': delta_abs,
                'failure_mode': mode,
            })
    
    # Sort by severity (worst first)
    failures.sort(key=lambda x: -x['delta_abs'])
    
    # Analyze failure patterns
    analysis = _analyze_failure_patterns(failures)
    
    return failures, analysis


def _analyze_failure_patterns(failures):
    """Find common features among failures → hints for phase separation."""
    if len(failures) < 3:
        return {'n_failures': len(failures), 'patterns': []}
    
    patterns = []
    
    # Check: do failures cluster by phase?
    by_phase = defaultdict(list)
    for f in failures:
        by_phase[f.get('phase', 'unknown')].append(f)
    
    for phase, group in by_phase.items():
        if len(group) >= 2:
            patterns.append({
                'type': 'phase_cluster',
                'phase': phase,
                'N': len(group),
                'pct': len(group) / len(failures),
                'mean_delta': np.mean([f['delta_abs'] for f in group]),
            })
    
    # Check: do failures cluster by mutant class?
    by_ma = defaultdict(list)
    for f in failures:
        by_ma[f['ma']].append(f)
    
    for ma, group in by_ma.items():
        if len(group) >= 2:
            patterns.append({
                'type': 'mutant_cluster',
                'mutant': ma,
                'N': len(group),
                'pct': len(group) / len(failures),
            })
    
    # Check: do failures cluster by wt class?
    by_wa = defaultdict(list)
    for f in failures:
        by_wa[f['wa']].append(f)
    
    for wa, group in by_wa.items():
        if len(group) >= 2:
            patterns.append({
                'type': 'wildtype_cluster',
                'wildtype': wa,
                'N': len(group),
                'pct': len(group) / len(failures),
            })
    
    # Check: burial distribution of failures vs successes
    fail_bur = [f['bur'] for f in failures]
    mean_fail_bur = np.mean(fail_bur)
    
    # Check: do failures have different burial from successes?
    patterns.append({
        'type': 'burial_profile',
        'mean_burial': mean_fail_bur,
        'std_burial': np.std(fail_bur),
        'hint': 'failures may need different burial threshold' if mean_fail_bur < 0.6 else 'burial OK',
    })
    
    # Check: do failures cluster by SS type?
    by_ss = defaultdict(list)
    for f in failures:
        by_ss[f.get('ss', 'unknown')].append(f)
    
    for ss, group in by_ss.items():
        if len(group) >= 2 and len(group) / len(failures) > 0.4:
            patterns.append({
                'type': 'ss_cluster',
                'ss_type': ss,
                'N': len(group),
                'pct': len(group) / len(failures),
            })
    
    return {
        'n_failures': len(failures),
        'failure_modes': dict((m, sum(1 for f in failures if f['failure_mode'] == m))
                             for m in set(f['failure_mode'] for f in failures)),
        'patterns': sorted(patterns, key=lambda p: -p.get('N', 0)),
    }


# ── Class 2: Unexpected Improvement ──
def find_unexpected_improvements(records, gate_filter, 
                                  improvement_threshold=1.0):
    """
    Find mutations where the gate DOESN'T fire, but the residual error
    is suspiciously aligned with what the gate corrects.
    
    These mutations suggest the gate's scope should be expanded,
    or a sibling gate is needed.
    
    Args:
        records: all mutation records
        gate_filter: function(record) → bool
        improvement_threshold: minimum |error| for non-gated mutations to flag
    """
    # First, determine the gate's correction direction
    gated = [r for r in records if gate_filter(r)]
    non_gated = [r for r in records if not gate_filter(r)]
    
    if len(gated) < 3:
        return [], {}
    
    gate_mean_err = np.mean([r['error'] for r in gated])
    gate_direction = 'UNDER' if gate_mean_err < -0.5 else 'OVER' if gate_mean_err > 0.5 else 'NEUTRAL'
    
    # Find non-gated mutations with same error direction
    candidates = []
    for r in non_gated:
        same_direction = False
        if gate_direction == 'UNDER' and r['error'] < -improvement_threshold:
            same_direction = True
        elif gate_direction == 'OVER' and r['error'] > improvement_threshold:
            same_direction = True
        
        if same_direction:
            candidates.append({
                **r,
                'gate_direction': gate_direction,
                'abs_error': abs(r['error']),
            })
    
    candidates.sort(key=lambda x: -x['abs_error'])
    
    # Analyze: why weren't these gated?
    analysis = _analyze_missed_opportunities(candidates, gate_filter, gated)
    
    return candidates, analysis


def _analyze_missed_opportunities(candidates, gate_filter, gated):
    """Why did the gate miss these? What feature separates them from gated mutations?"""
    if len(candidates) < 3 or len(gated) < 3:
        return {'n_missed': len(candidates)}
    
    insights = []
    
    # Compare feature distributions
    for feat in ['bur', 'p_void', 'n_cat', 'n_charged', 'nc']:
        gated_vals = [r.get(feat, 0) for r in gated]
        missed_vals = [r.get(feat, 0) for r in candidates]
        
        if not gated_vals or not missed_vals:
            continue
        
        g_mean = np.mean(gated_vals)
        m_mean = np.mean(missed_vals)
        
        if abs(g_mean - m_mean) > 0.1 * max(abs(g_mean), 0.1):
            insights.append({
                'feature': feat,
                'gated_mean': g_mean,
                'missed_mean': m_mean,
                'delta': m_mean - g_mean,
                'hint': f'Missed mutations have {"higher" if m_mean > g_mean else "lower"} {feat}',
            })
    
    # Check mutant class distribution of missed
    by_ma = defaultdict(int)
    for c in candidates:
        by_ma[c['ma']] += 1
    
    top_ma = sorted(by_ma.items(), key=lambda x: -x[1])[:3]
    
    # Check wt class distribution
    by_wa = defaultdict(int)
    for c in candidates:
        by_wa[c['wa']] += 1
    
    top_wa = sorted(by_wa.items(), key=lambda x: -x[1])[:3]
    
    return {
        'n_missed': len(candidates),
        'top_mutant_classes': top_ma,
        'top_wildtype_classes': top_wa,
        'feature_insights': sorted(insights, key=lambda x: -abs(x['delta'])),
    }


# ── Class 3: Phase Contamination ──
def find_phase_contamination(records, group_filter, contamination_std=1.5):
    """
    Within a group that should be homogeneous (same gate, same phase),
    find data points that pull in the opposite direction.
    
    These are either:
    - Measurement errors (noise)
    - Sub-phases that need further separation
    - Edge cases at phase boundaries
    """
    group = [r for r in records if group_filter(r)]
    
    if len(group) < 5:
        return [], {}
    
    errors = np.array([r['error'] for r in group])
    mean_err = np.mean(errors)
    std_err = np.std(errors)
    
    # Direction of the group
    group_direction = 'UNDER' if mean_err < -0.3 else 'OVER' if mean_err > 0.3 else 'NEUTRAL'
    
    contaminants = []
    for r, err in zip(group, errors):
        # Opposite direction from group mean
        is_opposite = (mean_err < 0 and err > 0) or (mean_err > 0 and err < 0)
        # Or extreme outlier in same direction
        is_extreme = abs(err - mean_err) > contamination_std * std_err
        
        if is_opposite and abs(err) > 1.0:
            contaminants.append({
                **r,
                'group_mean': mean_err,
                'group_direction': group_direction,
                'contamination_type': 'opposite_direction',
                'z_score': (err - mean_err) / max(std_err, 0.01),
            })
        elif is_extreme:
            contaminants.append({
                **r,
                'group_mean': mean_err,
                'group_direction': group_direction,
                'contamination_type': 'extreme_outlier',
                'z_score': (err - mean_err) / max(std_err, 0.01),
            })
    
    contaminants.sort(key=lambda x: -abs(x['z_score']))
    
    # Analyze: what makes contaminants different?
    analysis = _analyze_contamination(contaminants, group)
    
    return contaminants, analysis


def _analyze_contamination(contaminants, group):
    """What separates contaminants from the clean group?"""
    if len(contaminants) < 2:
        return {'n_contaminants': len(contaminants)}
    
    clean = [r for r in group if r not in contaminants]
    if len(clean) < 3:
        return {'n_contaminants': len(contaminants)}
    
    separators = []
    
    for feat in ['bur', 'p_void', 'n_cat', 'n_charged', 'nc', 'n_aro']:
        clean_vals = [r.get(feat, 0) for r in clean]
        contam_vals = [r.get(feat, 0) for r in contaminants]
        
        if not clean_vals or not contam_vals:
            continue
        
        c_mean = np.mean(clean_vals)
        t_mean = np.mean(contam_vals)
        c_std = np.std(clean_vals) + 0.01
        
        separation = abs(t_mean - c_mean) / c_std
        
        if separation > 0.5:
            separators.append({
                'feature': feat,
                'clean_mean': c_mean,
                'contam_mean': t_mean,
                'separation': separation,
                'hint': f'Contaminants have {"higher" if t_mean > c_mean else "lower"} {feat} '
                       f'(z={separation:.1f}σ) — possible sub-phase boundary',
            })
    
    # Check mutation type clustering
    by_type = defaultdict(int)
    for c in contaminants:
        by_type[f"{c['wa']}→{c['ma']}"] += 1
    
    return {
        'n_contaminants': len(contaminants),
        'contamination_rate': len(contaminants) / len(group),
        'separating_features': sorted(separators, key=lambda x: -x['separation']),
        'mutation_types': sorted(by_type.items(), key=lambda x: -x[1])[:5],
        'types': dict((t, sum(1 for c in contaminants if c['contamination_type'] == t))
                     for t in set(c['contamination_type'] for c in contaminants)),
    }


# ── Full Counterexample Report ──
def generate_counterexample_report(gate_name, failures, failure_analysis,
                                    missed, missed_analysis,
                                    contaminants, contam_analysis):
    """Generate the counterexample section of a gate report."""
    lines = []
    lines.append(f"\n  {'─'*60}")
    lines.append(f"  🔬 COUNTEREXAMPLE ANALYSIS: {gate_name}")
    lines.append(f"  {'─'*60}")
    
    # ── Class 1: Failures ──
    lines.append(f"\n  ① Gate Fired But Failed: {failure_analysis.get('n_failures', 0)} mutations")
    if failure_analysis.get('failure_modes'):
        for mode, count in failure_analysis['failure_modes'].items():
            lines.append(f"     {mode}: {count}")
    
    if failure_analysis.get('patterns'):
        lines.append(f"     Patterns detected:")
        for p in failure_analysis['patterns'][:3]:
            if p['type'] == 'phase_cluster':
                lines.append(f"       ⚡ {p['N']} failures in {p['phase']} phase ({p['pct']:.0%})")
            elif p['type'] == 'mutant_cluster':
                lines.append(f"       ⚡ {p['N']} failures with →{p['mutant']} ({p['pct']:.0%})")
            elif p['type'] == 'wildtype_cluster':
                lines.append(f"       ⚡ {p['N']} failures from {p['wildtype']}→ ({p['pct']:.0%})")
            elif p['type'] == 'ss_cluster':
                lines.append(f"       ⚡ {p['N']} failures in {p['ss_type']} ({p['pct']:.0%})")
    
    if failures:
        lines.append(f"     Top failures:")
        for f in failures[:5]:
            lines.append(f"       {f['pid']} {f['mc']:>7}  err {f['err_before']:+.2f}→{f['err_after']:+.2f}  "
                        f"Δ|e|={f['delta_abs']:+.2f}  [{f['failure_mode']}]  bur={f['bur']:.2f}")
    
    # ── Class 2: Missed Opportunities ──
    lines.append(f"\n  ② Unexpected Same-Direction (Gate missed): {missed_analysis.get('n_missed', 0)} mutations")
    if missed_analysis.get('top_mutant_classes'):
        lines.append(f"     Top mutant classes: {missed_analysis['top_mutant_classes']}")
    if missed_analysis.get('top_wildtype_classes'):
        lines.append(f"     Top wildtype classes: {missed_analysis['top_wildtype_classes']}")
    if missed_analysis.get('feature_insights'):
        lines.append(f"     Feature insights:")
        for ins in missed_analysis['feature_insights'][:3]:
            lines.append(f"       💡 {ins['hint']} ({ins['feature']}: "
                        f"gated={ins['gated_mean']:.2f}, missed={ins['missed_mean']:.2f})")
    
    if missed:
        lines.append(f"     Top missed opportunities:")
        for m in missed[:5]:
            lines.append(f"       {m['pid']} {m['mc']:>7}  err={m['error']:+.2f}  "
                        f"bur={m['bur']:.2f}  phase={m.get('phase','?')}")
    
    # ── Class 3: Contamination ──
    lines.append(f"\n  ③ Phase Contamination: {contam_analysis.get('n_contaminants', 0)} mutations "
                f"({contam_analysis.get('contamination_rate', 0):.0%} of group)")
    if contam_analysis.get('types'):
        for t, count in contam_analysis['types'].items():
            lines.append(f"     {t}: {count}")
    if contam_analysis.get('separating_features'):
        lines.append(f"     Separating features:")
        for s in contam_analysis['separating_features'][:3]:
            lines.append(f"       🔍 {s['hint']}")
    
    if contaminants:
        lines.append(f"     Top contaminants:")
        for c in contaminants[:5]:
            lines.append(f"       {c['pid']} {c['mc']:>7}  err={c['error']:+.2f}  "
                        f"z={c['z_score']:+.1f}σ  [{c['contamination_type']}]")
    
    # ── Seeds for Next Gate ──
    seeds = _extract_seeds(failure_analysis, missed_analysis, contam_analysis)
    if seeds:
        lines.append(f"\n  🌱 SEEDS FOR NEXT GATE:")
        for seed in seeds:
            lines.append(f"     → {seed}")
    
    return '\n'.join(lines)


def _extract_seeds(failure_analysis, missed_analysis, contam_analysis):
    """Convert counterexample patterns into actionable next-gate hypotheses."""
    seeds = []
    
    # From failures: phase clusters suggest phase separation
    for p in failure_analysis.get('patterns', []):
        if p['type'] == 'phase_cluster' and p['pct'] > 0.3:
            seeds.append(f"Failures cluster in {p['phase']} phase ({p['N']} mutations) "
                        f"→ consider phase-specific sub-gate or exclusion")
        if p['type'] == 'mutant_cluster' and p['pct'] > 0.3:
            seeds.append(f"Failures cluster with →{p['mutant']} ({p['N']} mutations) "
                        f"→ consider mutant-class exclusion or separate gate")
        if p['type'] == 'wildtype_cluster' and p['pct'] > 0.3:
            seeds.append(f"Failures cluster from {p['wildtype']}→ ({p['N']} mutations) "
                        f"→ consider wildtype-class refinement")
    
    # From missed: scope expansion
    if missed_analysis.get('n_missed', 0) > 5:
        top_wa = missed_analysis.get('top_wildtype_classes', [])
        if top_wa:
            top = top_wa[0]
            seeds.append(f"{top[1]} mutations from {top[0]}→ show same error pattern "
                        f"→ consider expanding gate wt_set to include {top[0]}")
        
        for ins in missed_analysis.get('feature_insights', [])[:1]:
            seeds.append(f"Missed mutations have {ins['hint']} "
                        f"→ consider relaxing {ins['feature']} threshold")
    
    # From contamination: sub-phase boundaries
    for s in contam_analysis.get('separating_features', [])[:1]:
        if s['separation'] > 1.0:
            seeds.append(f"Contaminants separated by {s['feature']} (z={s['separation']:.1f}σ) "
                        f"→ {s['feature']} may define a sub-phase boundary")
    
    return seeds


# ── Convenience: Run All Three on a Gate ──
def full_counterexample_analysis(records, gate_name, gate_filter, comp_fn, K):
    """Run all three counterexample analyses on a gate."""
    
    # Class 1: Failures
    failures, failure_analysis = find_gate_failures(
        records, gate_filter, comp_fn, K)
    
    # Class 2: Missed opportunities
    missed, missed_analysis = find_unexpected_improvements(
        records, gate_filter)
    
    # Class 3: Phase contamination
    contaminants, contam_analysis = find_phase_contamination(
        records, gate_filter)
    
    report = generate_counterexample_report(
        gate_name, failures, failure_analysis,
        missed, missed_analysis,
        contaminants, contam_analysis)
    
    return {
        'failures': failures,
        'failure_analysis': failure_analysis,
        'missed': missed,
        'missed_analysis': missed_analysis,
        'contaminants': contaminants,
        'contam_analysis': contam_analysis,
        'report': report,
        'seeds': _extract_seeds(failure_analysis, missed_analysis, contam_analysis),
    }


# ── Test ──
if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/home/claude')
    sys.path.insert(0, '/home/claude/gate_explorer')
    import ssoc_v318 as model
    from pipeline import load_dataset, CLASSES
    
    print("Loading dataset...")
    records = load_dataset(model, '/home/claude/thermomut_verified_v3.json', '/home/claude/pdb_cache')
    print(f"Loaded {len(records)} records\n")
    
    # ── Test on Gate 3b (Aro Gain from Charged) ──
    AROMATIC = set('FWY')
    CHARGED = set('DEKR')
    
    def gate3b_filter(r):
        return r['wa'] in CHARGED and r['ma'] in AROMATIC and r['bur'] >= 0.50
    
    def gate3b_comp(r, K):
        aa_w = {'W': 1.0, 'F': 0.7, 'Y': 0.7}.get(r['ma'], 0.7)
        bonus = K * aa_w
        if r.get('n_cat', 0) >= 2 and r['bur'] >= 0.70:
            bonus += K * aa_w
        return -bonus  # subtract (OVER correction)
    
    result = full_counterexample_analysis(records, 'Gate 3b: Aro Gain', gate3b_filter, gate3b_comp, K=1.0)
    print(result['report'])
    
    # ── Test on Gate 5 candidate (Charged→Small) ──
    print(f"\n{'='*80}")
    SMALL = set('AG')
    
    def gate5_filter(r):
        return r['wa'] in CHARGED and r['ma'] in SMALL and r['bur'] >= 0.70
    
    def gate5_comp(r, K):
        return K  # positive comp (UNDER correction)
    
    result5 = full_counterexample_analysis(records, 'Gate 5: Charged→Small', gate5_filter, gate5_comp, K=1.0)
    print(result5['report'])

