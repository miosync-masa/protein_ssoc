# Gate Explorer — セットアップ手順

## ディレクトリ構成（ご主人さまのPCに配置）

```
ssoc_workspace/
├── ssoc_v318.py                    ← モデル本体
├── thermomut_verified_v3.json      ← 変異データ
├── pdb_cache/                      ← PDBファイル群
│   ├── 1stn.pdb
│   ├── 1pga.pdb
│   └── ...
└── gate_explorer/                  ← このディレクトリ
    ├── run_explorer.py             ← エントリポイント
    ├── pipeline.py
    ├── conflict_resolver.py
    ├── reporter.py
    ├── counterexamples.py
    ├── orchestrator.py
    ├── schema.yaml
    ├── gates/                      ← Gate定義YAML
    │   ├── gate1_charge_intro.yaml
    │   ├── gate3b_aro_gain.yaml
    │   └── gate5_charged_to_small.yaml
    └── results/                    ← 自動生成される出力
```

## 実行方法

```bash
# 依存インストール
pip install -r requirements.txt

# フルスキャン（28カテゴリ × 鏡像構造）
python run_explorer.py

# 全部やる（スキャン + 鏡像 + 全Gate反例分析）
python run_explorer.py --all

# 特定Gateの反例分析
python run_explorer.py --counter gate3b
python run_explorer.py --counter gate5

# 候補Gate探索
python run_explorer.py --explore gate5

# パス指定
python run_explorer.py --all --dataset /path/to/data.json --pdb-dir /path/to/pdb/
```

## 新しいGateを追加する

1. `gates/` に YAML ファイルを作る（schema.yaml 参照）
2. `run_explorer.py` の GATE_DEFS / CANDIDATES に定義を追加
3. `python run_explorer.py --explore new_gate` で自動探索
4. `python run_explorer.py --counter new_gate` で反例分析
5. 結果が良ければ ssoc_v3XX.py に搭載

## 出力

results/ ディレクトリにタイムスタンプ付きレポートが生成されます：
- `full_scan_YYYYMMDD_HHMMSS.txt` — 全カテゴリランキング
- `mirrors_YYYYMMDD_HHMMSS.txt` — 鏡像構造一覧
- `counter_gate3b_YYYYMMDD_HHMMSS.txt` — 反例分析レポート
- `explore_gate5_YYYYMMDD_HHMMSS.txt` — Gate探索レポート
