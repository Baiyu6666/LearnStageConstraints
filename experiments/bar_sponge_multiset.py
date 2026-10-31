"""Merge frozen BarSponge scenes and screen reproducible multiset selections."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from envs import load_env
from methods.cores.map import StageWiseMAPConstraintLearningModel


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / 'robot/stage_cons_iiwa14/data/processed'
SAMPLE_KEYS = ('trajectory', 'features', 'timestamps', 'joint_positions',
               'world_z_force_n', 'tcp_speed_mps', 'true_stage_labels')
STATIC_KEYS = ('schema_version', 'trajectory_columns', 'feature_names',
               'downsample_hz', 'frame_id', 'bar_axis_local', 'stage_names', 'boundary_rule')
PARAMETERS = ((.15, 3., 1.), (.2, 3., 1.), (.2, 3., 4.),
              (.2, 3., 8.), (.25, 3., 1.), (.25, 3., 4.))
TARGETS = ((1, 5, 'eq'), (2, 0, 'lb'), (2, 1, 'inactive'))
EXPECTED = (('inactive', 'inactive', 'inactive', 'inactive', 'inactive', 'eq'),
            ('inactive', 'eq', 'eq', 'inactive', 'eq', 'eq'),
            ('lb', 'inactive', 'inactive', 'inactive', 'inactive', 'eq'))


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')


def merge_sets(sources, output):
    """Preserve sample data and each demo's scene; remap only IDs and bounds."""
    archives, provenance, bounds, ids, demo_ids = [], [], [], [], []
    offset = 0
    for set_index, source in enumerate(sources):
        source = Path(source)
        with np.load(source, allow_pickle=False) as archive:
            data = {key: archive[key].copy() for key in archive.files}
        load_env('BarSponge', processed_demo_path=source, n_demos=len(data['source_demo_ids']))
        if archives:
            for key in STATIC_KEYS:
                if not np.array_equal(data[key], archives[0][key]):
                    raise ValueError(f'Incompatible {key}: {source}')
        local_ids = data['source_demo_ids'].astype(int)
        if np.any(local_ids < 1) or np.any(local_ids >= 100):
            raise ValueError('Local demo IDs must be in 1..99 for set*100+demo IDs')
        global_ids = set_index * 100 + local_ids
        source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
        mapped = np.empty(len(data['trajectory']), dtype=int)
        for local_id, global_id, row in zip(local_ids, global_ids, data['true_bounds_indices']):
            mapped[row[0]:row[-1]] = global_id
            provenance.append(dict(merged_demo_id=int(global_id), set_name=f'barsponge_set{set_index}',
                                   original_demo_id=int(local_id), source_path=str(source.relative_to(ROOT))
                                   if source.is_relative_to(ROOT) else str(source),
                                   source_session_id=str(data['source_session_id'].item()),
                                   source_training_sha256=source_digest,
                                   samples=int(row[-1] - row[0])))
        bounds.append(data['true_bounds_indices'] + offset)
        ids.extend(global_ids.tolist())
        demo_ids.append(mapped)
        offset += len(data['trajectory'])
        archives.append(data)
    if not archives:
        raise ValueError('At least one input archive is required')
    merged = {key: archives[0][key] for key in STATIC_KEYS}
    merged.update({key: np.concatenate([a[key] for a in archives]) for key in SAMPLE_KEYS})
    merged.update(true_bounds_indices=np.concatenate(bounds), source_demo_ids=np.array(ids),
                  demo_id=np.concatenate(demo_ids),
                  demo_bar_poses_base=np.concatenate([a['demo_bar_poses_base'] for a in archives]),
                  demo_cup_poses_base=np.concatenate([a['demo_cup_poses_base'] for a in archives]),
                  source_set_ids=np.array([i['merged_demo_id'] // 100 for i in provenance]),
                  source_original_demo_ids=np.array([i['original_demo_id'] for i in provenance]),
                  source_session_ids_per_demo=np.array([i['source_session_id'] for i in provenance]),
                  source_session_id=np.array('barsponge_multiset_' + hashlib.sha256(
                      json.dumps(provenance, sort_keys=True).encode()).hexdigest()[:16]))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **merged)
    load_env('BarSponge', processed_demo_path=output, n_demos=len(ids))
    manifest = dict(schema_version=1, id_rule='merged_demo_id = set_index * 100 + original_demo_id',
                    sample_count=offset, demo_count=len(ids), demos=provenance,
                    training_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    write_json(output.with_suffix('.json'), manifest)
    return manifest


def combinations(groups, count, seed):
    """Sample bounded selections; equal two-per-set selections are included first."""
    rng = np.random.default_rng(seed)
    found = set()
    while len(found) < count:
        sizes = [2] * len(groups) if len(found) < count // 3 else rng.integers(1, 4, len(groups))
        if not 8 <= sum(sizes) <= 12:
            continue
        selection = tuple(sorted(int(i) for group, n in zip(groups, sizes)
                                 for i in rng.choice(group, int(n), replace=False)))
        if selection not in found:
            found.add(selection)
            yield selection


def model_for(dataset, kwargs):
    return StageWiseMAPConstraintLearningModel(
        demos=dataset.demos, env=dataset.env, precomputed_features=dataset.features,
        true_cutpoints=dataset.true_cutpoints, **kwargs)


def mode_score(model, infos, stage, feature, target):
    mode, vector, _, costs = model._pooled_mode_mstep(infos, stage, feature)
    margin = min(value for key, value in costs.items() if key != target) - costs[target]
    column = model.selected_feature_columns[feature]
    value = None if vector is None else float(vector[0] * model.feat_std[column] + model.feat_mean[column])
    return dict(stage=stage + 1, feature=feature, target=target, mode=mode,
                margin=float(margin), value=value, costs=costs)


def screen(archive, output, count=180):
    """Use annotations only for screening; final candidates must learn free boundaries."""
    with np.load(archive, allow_pickle=False) as a:
        ids = a['source_demo_ids'].astype(int).tolist()
    groups = [[i for i in ids if i // 100 == group] for group in range(5)]
    env_cfg = json.loads((ROOT / 'configs/envs/BarSponge.json').read_text())
    kwargs = json.loads((ROOT / 'configs/methods/map_balanced_pooled.json').read_text())
    kwargs.update(env_cfg['method_overrides']['map_balanced_pooled'])
    for key in ('name', 'max_iter'):
        kwargs.pop(key, None)
    kwargs.update(disable_plots=True, verbose=False, map_save_density_plots=False,
                  map_demo_num_workers=1, seed=0)
    if kwargs.get('map_activation_prior') is not None or kwargs.get('map_active_mode_prior') is not None:
        raise ValueError('This search must retain the default priors')
    rows = []
    for index, selection in enumerate(combinations(groups, count, 2026)):
        dataset = load_env('BarSponge', processed_demo_path=archive, source_demo_ids=selection)
        # Archive cutpoints are starts of the next stage; internal ends are inclusive.
        infos = [dict(stage_ends=[int(c[0]) - 1, int(c[1]) - 1, len(d) - 1])
                 for c, d in zip(dataset.true_cutpoints, dataset.demos)]
        for sigma, bg, inequality in PARAMETERS:
            parameters = dict(map_eq_sigma=sigma, map_c_bg=bg, map_c_ineq=inequality)
            model = model_for(dataset, dict(kwargs, **parameters))
            scores = [mode_score(model, infos, *target) for target in TARGETS]
            rows.append(dict(source_demo_ids=list(selection), parameters=parameters, scores=scores,
                             passed=all(s['mode'] == s['target'] for s in scores),
                             worst_margin=min(s['margin'] for s in scores)))
        if (index + 1) % 20 == 0:
            print(f'screened {index + 1} combinations / {len(rows)} fits; '
                  f'passed={sum(row["passed"] for row in rows)}', flush=True)
    rows.sort(key=lambda row: (row['passed'], row['worst_margin']), reverse=True)
    # Check every feature for the best candidates, so extra constraints are visible.
    for row in rows[:30]:
        dataset = load_env('BarSponge', processed_demo_path=archive, source_demo_ids=row['source_demo_ids'])
        model = model_for(dataset, dict(kwargs, **row['parameters']))
        infos = [dict(stage_ends=[int(c[0]) - 1, int(c[1]) - 1, len(d) - 1])
                 for c, d in zip(dataset.true_cutpoints, dataset.demos)]
        row['all_scores'] = [[mode_score(model, infos, s, f, EXPECTED[s][f])
                              for f in range(6)] for s in range(3)]
        row['mode_mismatches'] = sum(score['mode'] != score['target']
                                     for scores in row['all_scores'] for score in scores)
    output = Path(output)
    write_json(output / 'screening.json', dict(
        screening_only=True, boundary_source='recorded annotations, not learned segmentation',
        combination_seed=2026, method_seed=0, combinations=count, candidate_fits=len(rows),
        prior_policy='unchanged defaults', rows=rows))
    shortlist = sorted(rows[:30], key=lambda r: (not r['passed'], r['mode_mismatches'], -r['worst_margin']))
    write_json(output / 'shortlist.json', shortlist)
    for row in shortlist[:8]:
        print('SHORTLIST', row['source_demo_ids'], row['parameters'],
              'mismatches', row['mode_mismatches'], 'margin', round(row['worst_margin'], 3), flush=True)


def train_shortlist(archive, output, count):
    """Confirm distinct selections through the standard benchmark entrypoint."""
    from runners.run_benchmark import run_benchmark

    output = Path(output)
    shortlist = json.loads((output / 'shortlist.json').read_text())
    selections, results = set(), []
    for row in shortlist:
        selection = tuple(row['source_demo_ids'])
        if not row['passed'] or selection in selections:
            continue
        selections.add(selection)
        ordinal = len(results) + 1
        run_root = output / f'candidate_{ordinal:02d}'
        print('TRAIN', ordinal, selection, row['parameters'], flush=True)
        run_benchmark(
            methods=['map_balanced_pooled'], datasets=['BarSponge'], method_seeds=[0],
            outdir=run_root / 'benchmark', run_output_root=run_root,
            dataset_overrides={'BarSponge': dict(processed_demo_path=str(archive.relative_to(ROOT)),
                                               source_demo_ids=list(selection), n_demos=len(selection))},
            method_overrides={'map_balanced_pooled': dict(
                row['parameters'], map_demo_num_workers=2, map_save_density_plots=True)})
        run_dir = run_root / 'map_balanced_pooled/BarSponge/method_seed_000'
        constraints = json.loads((run_dir / 'constraints.json').read_text())
        segmentation = json.loads((run_dir / 'segmentation.json').read_text())
        modes = [[{'': 'inactive', 'target_value': 'eq', 'lower_bound': 'lb', 'upper_bound': 'ub'}[m]
                  for m in stage] for stage in constraints['ConstraintLearnedSemanticsMatrix']]
        targets_pass = all(modes[s][f] == target for s, f, target in TARGETS)
        mismatches = sum(modes[s][f] != EXPECTED[s][f] for s in range(3) for f in range(6))
        mae = float(np.mean(np.abs(np.array(segmentation['predicted_cutpoints'])
                                   - np.array(segmentation['true_cutpoints']))))
        snapshot = json.loads((run_dir / 'config_snapshot.json').read_text())
        dataset = load_env('BarSponge', **snapshot['dataset_kwargs'])
        kwargs = dict(snapshot['method_kwargs'])
        kwargs.pop('max_iter', None)
        kwargs.update(disable_plots=True, verbose=False, map_save_density_plots=False)
        model = model_for(dataset, kwargs)
        infos = [dict(stage_ends=ends) for ends in segmentation['predicted_stage_ends']]
        final_scores = [mode_score(model, infos, *target) for target in TARGETS]
        result = dict(candidate=ordinal, source_demo_ids=list(selection), parameters=row['parameters'],
                      core_targets_pass=targets_pass, mode_mismatches=mismatches,
                      boundary_mae_samples=mae, modes=modes,
                      learned_values=constraints['ConstraintLearnedValueMatrix'],
                      final_scores=final_scores, final_worst_margin=min(s['margin'] for s in final_scores),
                      screening_worst_margin=row['worst_margin'], run_dir=str(run_dir.relative_to(ROOT)))
        # Inactive learned values are NaN in legacy matrices; reports use strict JSON nulls.
        result['learned_values'] = [[v if np.isfinite(v) else None for v in stage]
                                     for stage in result['learned_values']]
        results.append(result)
        write_json(run_dir / 'selection.json', dict(
            source_demo_ids=list(selection), settings={f'barsponge_set{s}': [i % 100 for i in selection
                                                                          if i // 100 == s]
                                                       for s in range(5)},
            merged_manifest=str(archive.with_suffix('.json').relative_to(ROOT))))
        write_json(output / 'confirmed_results.json', results)
        print('CONFIRMED', json.dumps(result), flush=True)
        if len(results) >= count:
            break
    if not results:
        raise RuntimeError('No passing screened candidates to train')
    results.sort(key=lambda r: (not r['core_targets_pass'], r['mode_mismatches'],
                                -r['final_worst_margin'], r['boundary_mae_samples']))
    write_json(output / 'recommended.json', results[0])
    print('RECOMMENDED', json.dumps(results[0]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--screen', action='store_true')
    parser.add_argument('--combinations', type=int, default=180)
    parser.add_argument('--train-top', type=int, default=0)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/barsponge_multiset_search')
    args = parser.parse_args()
    archive = PROCESSED / 'barsponge_multiset/training_5hz.npz'
    if args.prepare:
        manifest = merge_sets([PROCESSED / f'barsponge_set{i}/training_5hz.npz' for i in range(5)], archive)
        print('MERGED', manifest['demo_count'], 'demos', manifest['sample_count'], 'samples', flush=True)
    if args.screen:
        screen(archive, args.output, args.combinations)
    if args.train_top:
        train_shortlist(archive, args.output, args.train_top)


if __name__ == '__main__':
    main()
