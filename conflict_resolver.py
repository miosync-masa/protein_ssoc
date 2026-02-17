"""
Conflict Resolver — Gate間の排他・優先・合成ルールエンジン

Conflict Types:
  1. exclusive    — 同一変異に対して同時発火禁止 (priority で勝者決定)
  2. suppress     — 特定条件で発火を抑制
  3. compose      — 両方発火OK、ただし合成ルール適用 (additive / max / ...)
  4. priority     — 低いpriority番号が先に適用

Architecture:
  GateRegistry に全ゲート定義をロード → 変異ごとに発火候補を列挙
  → conflict check → 最終発火リスト + 補正値を返す
"""
import yaml
import os
from collections import defaultdict


class GateDefinition:
    """Single gate loaded from YAML."""
    def __init__(self, yaml_path):
        with open(yaml_path) as f:
            self.config = yaml.safe_load(f)
        self.gate_id = self.config['gate_id']
        self.status = self.config.get('status', 'candidate')
        self.priority = self.config.get('conflicts', {}).get('priority', 99)
        self.exclusive_with = set(self.config.get('conflicts', {}).get('exclusive_with', []))
        self.suppress_if = self.config.get('conflicts', {}).get('suppress_if', [])
        self.compose_mode = self.config.get('conflicts', {}).get('compose_mode', 'additive')
        
        # Trigger
        trig = self.config.get('trigger', {})
        self.wt_set = set(trig.get('wt_set', []))
        self.mt_set = set(trig.get('mt_set', []))
        
        # Hard filters
        self.hard_filters = self.config.get('hard_filters', [])
        
        # Exclusions
        self.exclusions = self.config.get('exclusions', [])
        
        # Direction
        expected = self.config.get('expected', {})
        self.error_direction = expected.get('error_direction', 'UNKNOWN')
        self.correction_sign = expected.get('correction_sign', '+')
    
    def matches_trigger(self, record):
        """Check if a mutation record matches this gate's trigger."""
        wa, ma = record['wa'], record['ma']
        
        # WT check
        if self.wt_set and 'any' not in self.wt_set:
            if wa not in self.wt_set:
                return False
        
        # MT check
        if self.mt_set and 'any' not in self.mt_set:
            if ma not in self.mt_set:
                return False
        
        # Hard filters
        for filt in self.hard_filters:
            param = filt['param']
            op = filt['op']
            threshold = filt['threshold']
            val = record.get(param, 0)
            
            if op == '>=' and val < threshold: return False
            if op == '>'  and val <= threshold: return False
            if op == '<=' and val > threshold: return False
            if op == '<'  and val >= threshold: return False
            if op == '==' and val != threshold: return False
            if op == '!=' and val == threshold: return False
        
        # Exclusions (phase separation)
        from pipeline import CLASSES
        for excl in self.exclusions:
            wt_excl = excl.get('wt_class', [])
            mt_excl = excl.get('mt_class', [])
            
            wt_excluded = any(wa in CLASSES.get(cls, set()) for cls in wt_excl)
            mt_excluded = any(ma in CLASSES.get(cls, set()) for cls in mt_excl)
            
            if wt_excl and mt_excl:
                if wt_excluded and mt_excluded: return False
            elif wt_excl and wt_excluded: return False
            elif mt_excl and mt_excluded: return False
        
        return True
    
    def __repr__(self):
        return f"Gate({self.gate_id}, p={self.priority}, {self.status})"


class GateRegistry:
    """Registry of all gate definitions with conflict resolution."""
    
    def __init__(self, gates_dir=None):
        self.gates = {}
        if gates_dir:
            self.load_directory(gates_dir)
    
    def load_directory(self, gates_dir):
        """Load all YAML gate definitions from a directory."""
        for fname in os.listdir(gates_dir):
            if fname.endswith('.yaml') or fname.endswith('.yml'):
                path = os.path.join(gates_dir, fname)
                gate = GateDefinition(path)
                self.gates[gate.gate_id] = gate
        return self
    
    def load_gate(self, yaml_path):
        """Load a single gate definition."""
        gate = GateDefinition(yaml_path)
        self.gates[gate.gate_id] = gate
        return gate
    
    def find_matching_gates(self, record):
        """Find all gates that match a given mutation record."""
        matches = []
        for gate in self.gates.values():
            if gate.matches_trigger(record):
                matches.append(gate)
        # Sort by priority (lower = higher priority)
        matches.sort(key=lambda g: g.priority)
        return matches
    
    def resolve_conflicts(self, matches):
        """
        Given a list of matching gates (sorted by priority),
        resolve conflicts and return the final firing list.
        
        Returns: list of (gate, mode) tuples
          mode: 'fire' | 'suppressed' | 'exclusive_blocked'
        """
        results = []
        fired_ids = set()
        
        for gate in matches:
            # Check suppression
            suppressed = False
            for supp in gate.suppress_if:
                # Check if suppressor gate already fired
                cond = supp.get('condition', '')
                if 'already fired' in cond.lower():
                    # Check if any exclusive gate has fired
                    for exc_id in gate.exclusive_with:
                        if exc_id in fired_ids:
                            suppressed = True
                            break
                if suppressed:
                    break
            
            if suppressed:
                results.append((gate, 'suppressed'))
                continue
            
            # Check exclusive conflicts
            blocked = False
            for exc_id in gate.exclusive_with:
                if exc_id in fired_ids:
                    blocked = True
                    break
            
            if blocked:
                results.append((gate, 'exclusive_blocked'))
                continue
            
            # Fire!
            results.append((gate, 'fire'))
            fired_ids.add(gate.gate_id)
        
        return results
    
    def evaluate_record(self, record):
        """
        Full pipeline for one mutation:
        find matches → resolve conflicts → return firing decisions.
        """
        matches = self.find_matching_gates(record)
        if not matches:
            return {'gates_matched': 0, 'gates_fired': [], 'decisions': []}
        
        decisions = self.resolve_conflicts(matches)
        fired = [(g, m) for g, m in decisions if m == 'fire']
        
        return {
            'gates_matched': len(matches),
            'gates_fired': [g.gate_id for g, m in fired],
            'decisions': [(g.gate_id, m, g.priority) for g, m in decisions],
        }
    
    def conflict_matrix(self, records):
        """
        Scan all records and build a conflict frequency matrix.
        Returns: dict of {(gate_a, gate_b): count_of_co_fires}
        """
        co_fire = defaultdict(int)
        suppress_count = defaultdict(int)
        
        for r in records:
            result = self.evaluate_record(r)
            fired = result['gates_fired']
            
            # Co-fire matrix
            for i, ga in enumerate(fired):
                for gb in fired[i+1:]:
                    pair = tuple(sorted([ga, gb]))
                    co_fire[pair] += 1
            
            # Suppression tracking
            for gid, mode, pri in result['decisions']:
                if mode in ('suppressed', 'exclusive_blocked'):
                    suppress_count[gid] += 1
        
        return {
            'co_fires': dict(co_fire),
            'suppressions': dict(suppress_count),
        }
    
    def summary(self):
        """Print registry summary."""
        lines = []
        lines.append(f"Gate Registry: {len(self.gates)} gates loaded")
        for gate in sorted(self.gates.values(), key=lambda g: g.priority):
            excl = f" excl:{gate.exclusive_with}" if gate.exclusive_with else ""
            lines.append(f"  [{gate.priority:>3}] {gate.gate_id:<35} {gate.status:<12} {gate.error_direction}{excl}")
        return '\n'.join(lines)


# ── Test ──
if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/home/claude/gate_explorer')
    
    registry = GateRegistry('/home/claude/gate_explorer/gates')
    print(registry.summary())
    
    # Test with a mock record
    test_record = {
        'wa': 'D', 'ma': 'A', 'bur': 0.85, 'p_void': 0.75,
        'n_cat': 1, 'n_charged': 3, 'polar_frac': 0.15,
        'wa_classes': ['all_aa', 'charged'],
        'ma_classes': ['all_aa', 'small', 'small_polar_pro'],
        'phase': 'charge', 'opposite_charge_count': 2,
        'charge_asymmetry': 1.5,
    }
    print(f"\nTest: D→A (bur=0.85)")
    result = registry.evaluate_record(test_record)
    print(f"  Matched: {result['gates_matched']}")
    for gid, mode, pri in result['decisions']:
        print(f"  [{pri:>3}] {gid:<35} → {mode}")
    
    # Test: K→F (should fire gate3b, NOT gate5)
    test2 = {
        'wa': 'K', 'ma': 'F', 'bur': 0.75, 'p_void': 0.80,
        'n_cat': 3, 'n_charged': 2, 'polar_frac': 0.10,
        'wa_classes': ['all_aa', 'charged'],
        'ma_classes': ['all_aa', 'aromatic'],
        'phase': 'charge',
    }
    print(f"\nTest: K→F (bur=0.75, cat=3)")
    result2 = registry.evaluate_record(test2)
    for gid, mode, pri in result2['decisions']:
        print(f"  [{pri:>3}] {gid:<35} → {mode}")

