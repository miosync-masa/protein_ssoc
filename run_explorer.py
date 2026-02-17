#!/usr/bin/env python3
"""
SSOC Gate Explorer — ワンショット実行スクリプト

Usage:
  python run_explorer.py                          # フルスキャン
  python run_explorer.py --gate gate5             # 特定Gate探索
  python run_explorer.py --mirror                 # 鏡像構造スキャン
  python run_explorer.py --counter gate3b         # 反例分析
  python run_explorer.py --all                    # 全部やる

必要なもの:
  - ssoc_v318.py (同ディレクトリまたは親ディレクトリ)
  - thermomut_verified_v3.json (変異データ)
  - pdb_cache/ (PDBファイル群)
  - gates/*.yaml (Gate定義)

Output:
  results/ ディレクトリに全レポートを出力
"""
import sys, os, argparse, json, datetime

# パス設定（ご主人さまの環境に合わせて変更）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.dirname(BASE_DIR))  # ssoc_v318.py用

# ── Config ──
CONFIG = {
    'model_module': 'ssoc_v318',
    'dataset_path': os.path.join(os.path.dirname(BASE_DIR), 'thermomut_verified_v3.json'),
    'pdb_dir': os.path.join(os.path.dirname(BASE_DIR), 'pdb_cache'),
    'gates_dir': os.path.join(BASE_DIR, 'gates'),
    'output_dir': os.path.join(BASE_DIR, 'results'),
}

def ensure_output_dir():
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

def load_all():
    """Load model + dataset."""
    import importlib
    model = importlib.import_module(CONFIG['model_module'])
    from pipeline import load_dataset
    records = load_dataset(model, CONFIG['dataset_path'], CONFIG['pdb_dir'])
    print(f"Loaded {len(records)} records")
    return model, records

def run_full_scan(records):
    """Full candidate scan — 28 categories ranked by signal."""
    from orchestrator import full_scan
    results = full_scan(records)
    
    ensure_output_dir()
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    outpath = os.path.join(CONFIG['output_dir'], f'full_scan_{timestamp}.txt')
    
    with open(outpath, 'w') as f:
        f.write(f"SSOC Gate Explorer — Full Scan\n")
        f.write(f"Date: {timestamp}\n")
        f.write(f"Records: {len(records)}\n\n")
        f.write(f"{'src→dst':>25} {'N':>5} {'me':>7} {'MAE':>6} {'U':>4} {'O':>4} {'signal':>7} {'dir':>7}\n")
        for r in results:
            f.write(f"{r['src']+'→'+r['dst']:>25} {r['N']:>5} {r['mean_err']:>+7.3f} "
                   f"{r['mae']:>6.3f} {r['under']:>4} {r['over']:>4} "
                   f"{r['signal']:>7.1f} {r['direction']:>7}\n")
    
    print(f"  → Saved to {outpath}")
    return results

def run_mirror_scan(records):
    """Find all mirror structures."""
    from orchestrator import mirror_scan
    mirrors = mirror_scan(records)
    
    ensure_output_dir()
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    outpath = os.path.join(CONFIG['output_dir'], f'mirrors_{timestamp}.txt')
    
    with open(outpath, 'w') as f:
        f.write(f"Mirror Structure Scan — {len(mirrors)} pairs found\n\n")
        for m in mirrors:
            a = m['a_to_b']; b = m['b_to_a']
            f.write(f"🪞 {m['name_a']} ↔ {m['name_b']}  (asymmetry={m['asymmetry']:.2f})\n")
            f.write(f"   {m['name_a']}→{m['name_b']}: N={a['N']}  me={a['mean_err']:+.3f}  {a['direction']}\n")
            f.write(f"   {m['name_b']}→{m['name_a']}: N={b['N']}  me={b['mean_err']:+.3f}  {b['direction']}\n\n")
    
    print(f"  → Saved to {outpath}")
    return mirrors

def run_counterexamples(records, gate_name):
    """Run counterexample analysis on a specific gate."""
    from counterexamples import full_counterexample_analysis
    from pipeline import CLASSES
    
    # Gate definitions (add new gates here)
    GATE_DEFS = {
        'gate1': {
            'name': 'Gate 1: Charge Intro',
            'filter': lambda r: (r['wa'] in set('VILFWYMA') and r['ma'] in set('DEKR') 
                                and r['bur'] >= 0.85),
            'comp': lambda r, K: K,
            'K': 3.0,
        },
        'gate3a': {
            'name': 'Gate 3a: Aro Loss',
            'filter': lambda r: (r['wa'] in set('FWY') and r['ma'] in set('AGSTNQHP')
                                and r['bur'] >= 0.70),
            'comp': lambda r, K: K * {'W':1.0,'F':0.7,'Y':0.7}.get(r['wa'], 0.7),
            'K': 3.0,
        },
        'gate3b': {
            'name': 'Gate 3b: Aro Gain',
            'filter': lambda r: (r['wa'] in set('DEKR') and r['ma'] in set('FWY')
                                and r['bur'] >= 0.50),
            'comp': lambda r, K: -K * {'W':1.0,'F':0.7,'Y':0.7}.get(r['ma'], 0.7) * 
                                 (2.0 if r.get('n_cat',0) >= 2 and r['bur'] >= 0.70 else 1.0),
            'K': 1.0,
        },
        'gate5': {
            'name': 'Gate 5: Charged→Small',
            'filter': lambda r: (r['wa'] in set('DEKR') and r['ma'] in set('AG')
                                and r['bur'] >= 0.70),
            'comp': lambda r, K: K,
            'K': 1.0,
        },
    }
    
    if gate_name not in GATE_DEFS:
        print(f"Unknown gate: {gate_name}. Available: {list(GATE_DEFS.keys())}")
        return None
    
    gdef = GATE_DEFS[gate_name]
    result = full_counterexample_analysis(
        records, gdef['name'], gdef['filter'], gdef['comp'], gdef['K'])
    
    ensure_output_dir()
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    outpath = os.path.join(CONFIG['output_dir'], f'counter_{gate_name}_{timestamp}.txt')
    
    with open(outpath, 'w') as f:
        f.write(result['report'])
        f.write(f"\n\n{'='*60}\nSeeds for next gate:\n")
        for seed in result['seeds']:
            f.write(f"  → {seed}\n")
    
    print(result['report'])
    print(f"\n  → Saved to {outpath}")
    return result

def run_explore_gate(records, gate_name):
    """Full exploration of a specific gate candidate."""
    from orchestrator import explore_candidate
    from pipeline import CLASSES
    
    CANDIDATES = {
        'gate5': {
            'name': 'Gate 5: Charged→Small',
            'src': set('DEKR'), 'dst': set('AG'),
            'env': {'bur': ('>=', 0.70)},
        },
        'gate6_hydro_to_charged': {
            'name': 'Gate 6: HydroLarge→Charged',
            'src': set('VLIMC'), 'dst': set('DEKR'),
            'env': {},
        },
        'gate7_polar_to_charged': {
            'name': 'Gate 7: Polar→Charged',
            'src': set('STNQH'), 'dst': set('DEKR'),
            'env': {},
        },
    }
    
    if gate_name not in CANDIDATES:
        print(f"Unknown candidate: {gate_name}. Available: {list(CANDIDATES.keys())}")
        return None
    
    cand = CANDIDATES[gate_name]
    result = explore_candidate(records, cand['src'], cand['dst'], cand['name'],
                                env_filters=cand.get('env'))
    
    if result:
        ensure_output_dir()
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        outpath = os.path.join(CONFIG['output_dir'], f'explore_{gate_name}_{timestamp}.txt')
        with open(outpath, 'w') as f:
            f.write(result['report'])
        print(f"\n  → Saved to {outpath}")
    
    return result

def run_all(records):
    """Everything: scan + mirrors + top-5 counterexamples."""
    print(f"\n{'='*80}")
    print(f"RUNNING FULL PIPELINE")
    print(f"{'='*80}\n")
    
    scan = run_full_scan(records)
    mirrors = run_mirror_scan(records)
    
    for gate in ['gate1', 'gate3a', 'gate3b', 'gate5']:
        print(f"\n{'─'*80}")
        run_counterexamples(records, gate)
    
    print(f"\n{'='*80}")
    print(f"ALL DONE — results in {CONFIG['output_dir']}/")
    print(f"{'='*80}")


# ── Main ──
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SSOC Gate Explorer')
    parser.add_argument('--all', action='store_true', help='Run everything')
    parser.add_argument('--scan', action='store_true', help='Full candidate scan')
    parser.add_argument('--mirror', action='store_true', help='Mirror structure scan')
    parser.add_argument('--counter', type=str, help='Counterexample analysis (gate name)')
    parser.add_argument('--explore', type=str, help='Explore specific gate candidate')
    parser.add_argument('--dataset', type=str, help='Override dataset path')
    parser.add_argument('--pdb-dir', type=str, help='Override PDB directory')
    args = parser.parse_args()
    
    if args.dataset:
        CONFIG['dataset_path'] = args.dataset
    if args.pdb_dir:
        CONFIG['pdb_dir'] = args.pdb_dir
    
    model, records = load_all()
    
    if args.all:
        run_all(records)
    elif args.scan:
        run_full_scan(records)
    elif args.mirror:
        run_mirror_scan(records)
    elif args.counter:
        run_counterexamples(records, args.counter)
    elif args.explore:
        run_explore_gate(records, args.explore)
    else:
        # Default: full scan + mirrors
        run_full_scan(records)
        run_mirror_scan(records)
        print(f"\nTip: use --all for full pipeline, --counter gate3b for counterexamples")

