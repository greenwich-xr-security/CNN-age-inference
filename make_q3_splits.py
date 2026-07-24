#!/usr/bin/env python3
"""Create nested, user-disjoint Q3 real fine-tuning and validation subsets."""
from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
from dataset.hand_metadata import load_combined_metadata
from dataset.utils import filter_metadata, load_test_split

FRACTIONS = (0.01, 0.05, 0.10, 0.25, 0.50, 1.00)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', required=True); p.add_argument('--test-users-file', required=True)
    p.add_argument('--output-dir', default='splits/q3'); p.add_argument('--seed', type=int, required=True)
    p.add_argument('--val-fraction', type=float, default=0.2)
    p.add_argument('--datasets', nargs='+', default=['handrgbd','archive','primary','prolific'])
    a = p.parse_args(); out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    meta = filter_metadata(load_combined_metadata(root=a.data_root, sources=a.datasets))
    held = {str(x) for x in load_test_split(a.test_users_file)['test_user_ids']}
    users = meta[~meta.user_id.astype(str).isin(held)].groupby('user_id').agg(age=('age','first'), source=('source','first')).reset_index()
    rng = np.random.default_rng(a.seed)
    users['stratum'] = users.source.astype(str) + '_' + (users.age.astype(float)//10).astype(int).astype(str)
    ordered = []
    for _, group in users.groupby('stratum', sort=True):
        ordered.extend(rng.permutation(group.user_id.astype(str)).tolist())
    val_n = max(1, round(len(ordered)*a.val_fraction)); val_ids = set(ordered[:val_n]); pool = [u for u in ordered if u not in val_ids]
    for fraction in FRACTIONS:
        n = max(1, round(len(pool)*fraction)); train_ids = pool[:n]; tag = f"{int(fraction*100):03d}pct"
        for name, ids in [('train',train_ids),('val',sorted(val_ids))]:
            (out/f"seed{a.seed}_{tag}_{name}_users.json").write_text(json.dumps({'user_ids':ids,'seed':a.seed,'fraction':fraction}, indent=2)+'\n')
    print(f"Wrote Q3 seed {a.seed}: {len(pool)} train-pool users, {len(val_ids)} validation users")
if __name__ == '__main__': main()
