"""Deterministic, CSV-only 17-model synthesis. Never imports a model runtime."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from decimal import Decimal, localcontext
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys

MASTER = Path(__file__).resolve().parents[1]
WORKSPACE = MASTER.parent
TIERS = [
    ('largest', 'Large', ['yolo26x-seg.pt', 'yolo11x-seg.pt', 'yolov9e-seg.pt', 'yolov8x-seg.pt']),
    ('second_largest', 'Second_Largest', ['yolo26l-seg.pt', 'yolo11l-seg.pt', 'yolov9c-seg.pt', 'yolov8l-seg.pt']),
    ('medium', 'Medium', ['yolo26m-seg.pt', 'yolo11m-seg.pt', 'yolov8m-seg.pt']),
    ('small', 'Small', ['yolo26s-seg.pt', 'yolo11s-seg.pt', 'yolov8s-seg.pt']),
    ('nano', 'Nano', ['yolo26n-seg.pt', 'yolo11n-seg.pt', 'yolov8n-seg.pt']),
]
FAMILIES = ['YOLO26', 'YOLO11', 'YOLOv8', 'YOLOv9']
SEQUENCES = {'MOTS20-02': (600, 7039), 'MOTS20-05': (837, 6570), 'MOTS20-09': (525, 4774), 'MOTS20-11': (900, 8511)}
WINNERS = [('mask_map50_95', 'max', 'fraction'), ('ap75', 'max', 'fraction'), ('recall', 'max', 'fraction'),
           ('inference_ms_mean', 'min', 'ms/frame'), ('pipeline_ms_mean', 'min', 'ms/frame'),
           ('fps', 'max', 'frames/s'), ('peak_allocated_vram_mib', 'min', 'MiB')]
DELTA_FIELDS = [('mask_map50_95', 'fraction'), ('ap50', 'fraction'), ('ap75', 'fraction'), ('recall', 'fraction'),
                ('f1', 'fraction'), ('inference_ms_mean', 'ms/frame'), ('pipeline_ms_mean', 'ms/frame'),
                ('fps', 'frames/s'), ('peak_allocated_vram_mib', 'MiB'), ('parameters', 'count'), ('checkpoint_mb', 'MB')]
ACCURACY = ['mask_map50_95', 'ap50', 'ap75', 'precision', 'recall', 'f1', 'tp_iou_mean', 'tp_dice_mean']
STAGES = ['preprocessing', 'inference', 'postprocessing', 'pipeline', 'rle_preparation', 'ultralytics_postprocess_inclusive']
D = Decimal


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path):
    return json.loads(path.read_text())


def read_csv(path, header=None):
    with path.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        if header is not None:
            require(reader.fieldnames == header, f'Schema mismatch: {path}')
        result = list(reader)
    require(all(None not in row and all(v is not None for v in row.values()) for row in result), f'Malformed CSV: {path}')
    return result


def csv_bytes(header, rows):
    import io
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=header, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def write_new(path, content):
    """Identical rebuilds are harmless; different artifacts require archival first."""
    if path.exists():
        require(path.read_bytes() == content, f'Refusing to overwrite a different artifact: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def close(a, b, tolerance='0.000000001'):
    return abs(D(str(a)) - D(str(b))) <= D(tolerance)


def fraction_checks(row, label):
    for key in ACCURACY:
        require(D(row[key]).is_finite() and 0 <= D(row[key]) <= 1, f'{label}: invalid {key}')
    tp, fp, fn = [int(row[k]) for k in ['tp', 'fp', 'fn']]
    require(min(tp, fp, fn) >= 0 and tp + fn == int(row['gt_instances']), f'{label}: GT/count mismatch')
    require(close(row['precision'], D(tp) / (tp + fp)), f'{label}: precision arithmetic')
    require(close(row['recall'], D(tp) / (tp + fn)), f'{label}: recall arithmetic')
    require(close(row['f1'], D(2 * tp) / (2 * tp + fp + fn)), f'{label}: F1 arithmetic')


def ensure_identity(row, repo, tier, study, models):
    require(row['study_id'] == study['study_id'] and row['schema_version'] == study['schema_version'], f'{repo}: study/schema identity')
    require(row['experiment'] == repo and row['tier'] == tier and row['model'] in models, f'{repo}: row identity')
    require(row['family'] == models[row['model']]['family'], f'{repo}: family identity')


def validate_inputs(workspace=WORKSPACE):
    master = workspace / MASTER.name
    state = load_json(master / 'STUDY_STATE.json')
    schema = load_json(master / 'schemas.json')
    baseline = load_json(master / 'provenance/BASELINE_REFERENCE.json')
    dataset_evidence = load_json(master / 'provenance/DATASET_VALIDATION.json')
    require(dataset_evidence['status'] == 'PASS' and dataset_evidence['all_image_hashes_and_dimensions_match'] and dataset_evidence['frame_order_matches'], 'Shared dataset validation not PASS')
    reference_repo = workspace / 'YOLO_Large_Seg_MOTS20_Benchmark'
    reference_lists = {name: load_json(reference_repo / f'manifests/{name}.json') for name in ['timing_frames', 'preflight_frames', 'visualization_frames']}
    for name in ['timing_frames', 'preflight_frames']:
        require(len(reference_lists[name]) == 100 and len({(x['sequence'], x['frame']) for x in reference_lists[name]}) == 100, f'{name}: frozen 100-frame subset')
    controls = ['STUDY_STANDARD.md', 'DATA_SCHEMA.md', 'schemas.json', 'METHODOLOGY_REFERENCE.md', 'DOCUMENTATION_LANGUAGE_POLICY.md', 'provenance/BASELINE_REFERENCE.json', 'provenance/DATASET_VALIDATION.json']
    control_hashes = {p: sha(master / p) for p in controls}
    sources = []
    merged, per_sequence, timing, complexity = [], [], [], []
    import yaml  # Existing environment; no package installation or model imports.
    for tier, slug, expected in TIERS:
        repo = f'YOLO_{slug}_Seg_MOTS20_Benchmark'
        directory = workspace / repo
        tier_state = state['tiers'][tier]
        require(tier_state['repository'] == repo and tier_state['completion_status'] == 'COMPLETE' and tier_state['standardized'], f'{tier}: incomplete')
        require(tier_state['result_status'] in ['PASS', 'PASS_WITH_WARNINGS'], f'{tier}: invalid result status')
        require(not subprocess.check_output(['git', '-C', str(directory), 'status', '--porcelain', '--untracked-files=no'], text=True).strip(), f'{tier}: tracked source changes')
        std_path = directory / 'manifests/STANDARDIZATION.json'
        std = load_json(std_path)
        require(std['tier'] == tier and std['study_id'] == schema['study_id'] and std['schema_version'] == schema['schema_version'], f'{tier}: provenance identity')
        verified, not_read = {}, {}
        # Raw prediction artifacts and prediction metadata are not opened.
        for rel, expected_sha in std['source_artifact_sha256'].items():
            if 'predictions' in Path(rel).parts or 'prediction_hashes' in rel:
                not_read[rel] = expected_sha
                continue
            actual = sha(directory / rel)
            require(actual == expected_sha, f'{tier}: changed authoritative source {rel}')
            verified[rel] = actual
        frozen = {}
        reference_hashes = list(baseline['implementation_sha256'].values())
        require(set(std['evaluator_hash'].values()) == {v for k, v in baseline['implementation_sha256'].items() if Path(k).name in ['metrics.py', 'mots.py']}, f'{tier}: incompatible evaluator')
        for rel, expected_sha in std['evaluator_hash'].items():
            require(sha(directory / rel) == expected_sha, f'{tier}: evaluator changed')
            frozen[rel] = expected_sha
            parent = (directory / rel).parent
            for implementation in [parent / 'yolo_adapter.py', parent.parent / 'benchmark_adapter.py']:
                require(implementation.exists() and sha(implementation) in reference_hashes, f'{tier}: incompatible preprocessing {implementation.name}')
                frozen[str(implementation.relative_to(directory))] = sha(implementation)
        config_path = next(p for p in std['source_artifact_sha256'] if p.endswith('benchmark.yaml'))
        config = yaml.safe_load((directory / config_path).read_text())
        for key, value in baseline['config'].items():
            if key != 'models':
                require(config[key] == value, f'{tier}: incompatible frozen config {key}')
        require(set(config['models']) == set(expected), f'{tier}: config membership')
        env = load_json(directory / 'manifests/environment.json')
        for key in ['python', 'packages', 'torch_version', 'torchvision_version', 'cuda_runtime', 'gpu']:
            require(env[key] == baseline['environment'][key], f'{tier}: environment mismatch {key}')
        manifest_path = next(p for p in std['source_artifact_sha256'] if p.endswith('/dataset_manifest.json'))
        dataset = load_json(directory / manifest_path)
        require(dataset['total_frames'] == 2862 and dataset['image_manifest_sha256'] == dataset_evidence['ordered_image_manifest_sha256'], f'{tier}: image ordering/dataset')
        require([x['sequence'] for x in dataset['sequences']] == list(SEQUENCES), f'{tier}: sequence order')
        for sequence, base_seq in zip(dataset['sequences'], dataset_evidence['sequences']):
            for key in ['sequence', 'frames', 'resolution_wh', 'gt_sha256', 'seqinfo_sha256']:
                require(sequence[key] == base_seq[key], f'{tier}: dataset mismatch {key}')
            require(sequence['classes']['2'] == SEQUENCES[sequence['sequence']][1], f'{tier}: GT annotations')
        list_hashes = {}
        for name, reference in reference_lists.items():
            path = directory / f'manifests/{name}.json'
            require(load_json(path) == reference, f'{tier}: frozen subset mismatch {name}')
            list_hashes[str(path.relative_to(directory))] = sha(path)
        table_names = list(schema['schemas']) + ['PREFLIGHT_MAXDET']
        tables = {name: read_csv(directory / f'metrics/{name}.csv', schema['schemas'][name] if name in schema['schemas'] else schema[name]) for name in table_names}
        overall = tables['TIER_RESULTS']
        require([x['checkpoint'] for x in overall] == expected, f'{tier}: model order/set')
        for row in overall:
            stem = row['checkpoint'].removesuffix('-seg.pt')
            prefix = next((a for a in ['yolo26','yolo11','yolov9','yolov8'] if stem.startswith(a)), None)
            family = {'yolo26':'YOLO26','yolo11':'YOLO11','yolov9':'YOLOv9','yolov8':'YOLOv8'}[prefix]
            require(row['family'] == family and row['model'] == family + stem[len(prefix):] + '-Seg', f'{tier}: checkpoint/display/family mapping')
        models = {x['model']: x for x in overall}
        require(len(models) == len(expected), f'{tier}: duplicate display models')
        for name, rows in tables.items():
            for row in rows:
                ensure_identity(row, repo, tier, schema, models)
        original_accuracy_path = next(p for p in verified if p.endswith('/per_model.csv'))
        original_timing_path = next(p for p in verified if p.startswith('timing/') and p.endswith('/clean_repetition/summary.csv'))
        original_accuracy = {x['model']: x for x in read_csv(directory / original_accuracy_path)}
        original_timing = {x['model']: x for x in read_csv(directory / original_timing_path)}
        for row in overall:
            label = row['model']
            require(row['status'] in ['PASS', 'PASS_WITH_WARNINGS'], f'{label}: non-passing row')
            require(row['run_id'] == tier_state['canonical_run_id'] and row['run_id'] in std['source_run_ids'], f'{label}: run identity')
            require(row['frames'] == '2862' and row['gt_instances'] == '26894', f'{label}: coverage')
            for key, value in [('ap_maxdet', '200'), ('fixed_confidence', '0.25'), ('ap_confidence_floor', '0.001'), ('nms_iou', '0.7'), ('evaluation_iou', '0.5'), ('imgsz', '640'), ('precision_mode', 'FP32'), ('device', 'CUDA:0')]:
                require(row[key] == value, f'{label}: protocol column {key}')
            for path in row['source_artifact'].split(';'):
                require(path in std['source_artifact_sha256'], f'{label}: missing source hash')
            fraction_checks(row, label)
            original = original_accuracy[row['checkpoint']]
            mapping = {'mask_map50_95': 'map50_95', 'tp_iou_mean': 'matched_iou_mean', 'tp_dice_mean': 'matched_dice_mean', 'gt_instances': 'gt_persons'}
            for key in ACCURACY + ['frames', 'gt_instances', 'tp', 'fp', 'fn', 'ignored_predictions']:
                require(row[key] == original[mapping.get(key, key)], f'{label}: source precision/value {key}')
            measured = original_timing[row['checkpoint']]
            require(measured['clean_rounds'] == '3' and measured['frames'] == '300', f'{label}: incomplete clean timing')
            for key, source in [('inference_ms_mean', 'inference_ms_mean'), ('pipeline_ms_mean', 'total_ms_mean'), ('fps', 'fps'), ('peak_allocated_vram_mib', 'peak_gpu_allocated_mib')]:
                require(row[key] == measured[source], f'{label}: timing source {key}')
                require(D(row[key]).is_finite() and D(row[key]) > 0, f'{label}: invalid timing/memory')
            require(close(row['fps'], D(1000) / D(row['pipeline_ms_mean'])), f'{label}: FPS arithmetic')
            sequence_rows = [x for x in tables['PER_SEQUENCE_RESULTS'] if x['model'] == label]
            require([x['sequence'] for x in sequence_rows] == list(SEQUENCES), f'{label}: per-sequence coverage/order')
            for seq in sequence_rows:
                require((int(seq['frames']), int(seq['gt_instances'])) == SEQUENCES[seq['sequence']], f'{label}: sequence counts')
                fraction_checks(seq, label + '/' + seq['sequence'])
                require(int(seq['predictions']) == sum(int(seq[k]) for k in ['tp', 'fp', 'ignored_predictions']), f'{label}: prediction count')
            for key in ['frames', 'gt_instances', 'tp', 'fp', 'fn', 'ignored_predictions']:
                require(sum(int(x[key]) for x in sequence_rows) == int(row[key]), f'{label}: pooled count {key}')
            stages = [x for x in tables['TIMING_SUMMARY'] if x['model'] == label]
            require([x['stage'] for x in stages] == STAGES, f'{label}: timing stages')
            require(all(x['repetitions'] == '3' and x['measured_frames'] == '300' and x['contamination_status'] == 'CLEAN' for x in stages), f'{label}: contaminated/incomplete timing')
            for stage, field in [('inference', 'inference_ms_mean'), ('pipeline', 'pipeline_ms_mean')]:
                require(next(x['mean_ms'] for x in stages if x['stage'] == stage) == row[field], f'{label}: canonical timing mismatch')
            require(close(sum(D(x['mean_ms']) for x in stages if x['stage'] in STAGES[:3]), row['pipeline_ms_mean']), f'{label}: pipeline components')
            caps = [x for x in tables['PREFLIGHT_MAXDET'] if x['model'] == label]
            require([x['max_dets'] for x in caps] == ['100', '200', '300', '1000'], f'{label}: maxDet sensitivity coverage')
            reference = caps[-1]
            for cap in caps:
                diffs = []
                for key, difference in [('ap50', 'ap50_abs_difference_from_1000'), ('ap75', 'ap75_abs_difference_from_1000'), ('map50_95', 'map50_95_abs_difference_from_1000')]:
                    delta = abs(D(cap[key]) - D(reference[key]));diffs.append(delta)
                    require(close(delta, cap[difference]), f'{label}: maxDet difference')
                require((cap['converged'] == 'True') == all(x < D('0.0001') for x in diffs), f'{label}: maxDet convergence arithmetic')
            require(caps[1]['converged'] == 'True', f'{label}: maxDet200 invalid; STOP')
            comp = [x for x in tables['MODEL_COMPLEXITY'] if x['model'] == label]
            require(len(comp) == 1, f'{label}: complexity coverage')
            for key, source in [('parameters', 'loaded_parameters'), ('gflops', 'gflops'), ('checkpoint_mb', 'checkpoint_mb')]:
                require(row[key] == comp[0][source], f'{label}: complexity mismatch')
        for name, key in [('PER_SEQUENCE_RESULTS', 'sequence'), ('TIMING_SUMMARY', 'stage'), ('PREFLIGHT_MAXDET', 'max_dets')]:
            require(len({(x['model'], x[key]) for x in tables[name]}) == len(tables[name]), f'{tier}: duplicate {name} key')
        final_path = directory / 'manifests/final_integrity.json'
        if final_path.exists():
            final = load_json(final_path)
            require(final['status'] == 'PASS', f'{tier}: final integrity')
            if 'checks' in final:
                require(all(x['status'] == 'PASS' for x in final['checks']), f'{tier}: failed scientific check')
            verified['manifests/final_integrity.json'] = sha(final_path)
        sources.append({'repository': repo, 'tier': tier, 'revision': subprocess.check_output(['git', '-C', str(directory), 'rev-parse', 'HEAD'], text=True).strip(),
                        'run_id': tier_state['canonical_run_id'], 'completion_status': 'COMPLETE', 'result_status': tier_state['result_status'],
                        'canonical_csv_sha256': {f'metrics/{name}.csv': sha(directory / f'metrics/{name}.csv') for name in table_names},
                        'standardization_sha256': sha(std_path), 'verified_authoritative_sha256': verified, 'verified_implementation_sha256': frozen,
                        'verified_frozen_lists_sha256': list_hashes, 'upstream_prediction_hashes_not_opened': not_read,
                        'compatibility': 'PASS', 'models': expected})
        merged.extend(overall);per_sequence.extend(tables['PER_SEQUENCE_RESULTS']);timing.extend(tables['TIMING_SUMMARY']);complexity.extend(tables['MODEL_COMPLEXITY'])
    require(len(merged) == 17 and len({x['checkpoint'] for x in merged}) == 17 and len({x['model'] for x in merged}) == 17, 'Exactly 17 unique models required')
    require(Counter(x['family'] for x in merged) == Counter({'YOLO26': 5, 'YOLO11': 5, 'YOLOv8': 5, 'YOLOv9': 2}), 'Family counts')
    require(len(per_sequence) == 68 and len(timing) == 102 and len(complexity) == 17, 'Merged auxiliary coverage')
    return {'overall': merged, 'sequence': per_sequence, 'timing': timing, 'complexity': complexity, 'sources': sources, 'controls': control_hashes, 'schema': schema}


def dominates(a, b, cost):
    return (D(a['mask_map50_95']) >= D(b['mask_map50_95']) and D(a[cost]) <= D(b[cost]) and
            (D(a['mask_map50_95']) > D(b['mask_map50_95']) or D(a[cost]) < D(b[cost])))


def pareto(rows, cost):
    return [r for r in rows if not any(dominates(other, r, cost) for other in rows)]


def products(inputs):
    rows = inputs['overall'];schema = inputs['schema'];output = {}
    output['MASTER_17_MODELS.csv'] = (schema['schemas']['TIER_RESULTS'], rows)
    output['PER_SEQUENCE_MASTER.csv'] = (schema['schemas']['PER_SEQUENCE_RESULTS'], inputs['sequence'])
    output['TIMING_MASTER.csv'] = (schema['schemas']['TIMING_SUMMARY'], inputs['timing'])
    output['MODEL_COMPLEXITY_MASTER.csv'] = (schema['schemas']['MODEL_COMPLEXITY'], inputs['complexity'])
    winner_rows = []
    for tier, _, _ in TIERS:
        subset = [x for x in rows if x['tier'] == tier]
        for metric, direction, unit in WINNERS:
            extreme = (max if direction == 'max' else min)(D(x[metric]) for x in subset)
            for row in subset:
                if D(row[metric]) == extreme:
                    winner_rows.append({'tier': tier, 'metric': metric, 'direction': direction, 'model': row['model'], 'checkpoint': row['checkpoint'], 'value': row[metric], 'unit': unit})
    output['TIER_WINNERS.csv'] = (['tier', 'metric', 'direction', 'model', 'checkpoint', 'value', 'unit'], winner_rows)
    deltas = []
    for family in FAMILIES:
        family_rows = [r for r in rows if r['family'] == family]
        for large, small in zip(family_rows, family_rows[1:]):
            for field, unit in DELTA_FIELDS:
                with localcontext() as ctx:
                    ctx.prec = 50
                    absolute = D(small[field]) - D(large[field])
                    relative = absolute / D(large[field]) * 100 if D(large[field]) else None
                    pp = absolute * 100 if unit == 'fraction' else None
                deltas.append({'family': family, 'larger_model': large['model'], 'smaller_model': small['model'], 'larger_tier': large['tier'], 'smaller_tier': small['tier'],
                               'metric': field, 'unit': unit, 'larger_value': large[field], 'smaller_value': small[field], 'delta_absolute': str(absolute),
                               'delta_relative_pct': str(relative) if relative is not None else '', 'delta_percentage_points': str(pp) if pp is not None else ''})
    output['SCALING_DELTAS.csv'] = (list(deltas[0]), deltas)
    front_fields = ['tier', 'family', 'model', 'checkpoint', 'mask_map50_95', 'ap75', 'recall', 'inference_ms_mean', 'pipeline_ms_mean', 'fps', 'peak_allocated_vram_mib', 'parameters', 'checkpoint_mb']
    fronts = {}
    for filename, cost in [('PARETO_FRONTIER.csv', 'inference_ms_mean'), ('PARETO_VRAM.csv', 'peak_allocated_vram_mib')]:
        front = pareto(rows, cost);fronts[cost] = {x['model'] for x in front}
        output[filename] = (front_fields, [{k: x[k] for k in front_fields} for x in front])
    plot_rows = []
    for i, row in enumerate(rows):
        item = {k: row[k] for k in front_fields + ['ap50', 'f1', 'gflops']}
        item.update(display_order=str(i + 1), family_size_index=str(next(j for j, t in enumerate(TIERS) if t[0] == row['tier'])),
                    pareto_accuracy_latency=str(row['model'] in fronts['inference_ms_mean']), pareto_accuracy_vram=str(row['model'] in fronts['peak_allocated_vram_mib']))
        for stage in ['preprocessing', 'postprocessing', 'rle_preparation']:
            item[stage + '_ms_mean'] = next(x['mean_ms'] for x in inputs['timing'] if x['model'] == row['model'] and x['stage'] == stage)
        plot_rows.append(item)
    output['PLOT_READY_RESULTS.csv'] = (list(plot_rows[0]), plot_rows)
    nearby = []
    for a, b in itertools.combinations(rows, 2):
        gap = abs(D(a['mask_map50_95']) - D(b['mask_map50_95']))
        if gap <= D('0.001'):
            nearby.append({'model_a': a['model'], 'model_b': b['model'], 'tier_a': a['tier'], 'tier_b': b['tier'], 'map_a': a['mask_map50_95'], 'map_b': b['mask_map50_95'], 'map_abs_gap': str(gap), 'descriptive_screen_threshold': '0.001', 'statistical_significance_tested': 'False'})
    output['NEAR_TIES.csv'] = (['model_a', 'model_b', 'tier_a', 'tier_b', 'map_a', 'map_b', 'map_abs_gap', 'descriptive_screen_threshold', 'statistical_significance_tested'], nearby)
    return output


def validate_outputs(inputs, directory):
    outputs = products(inputs)
    for name, (header, rows) in outputs.items():
        path = directory / 'metrics' / name
        require(path.read_bytes() == csv_bytes(header, rows), f'Master calculation/source mismatch: {name}')
    # Independent checks use pairwise strict dominance and direct delta formulae.
    rows = read_csv(directory / 'metrics/MASTER_17_MODELS.csv')
    lookup = {x['model']: x for x in rows}
    deltas = read_csv(directory / 'metrics/SCALING_DELTAS.csv')
    require(len(deltas) == 143 and len({(r['larger_model'], r['smaller_model']) for r in deltas}) == 13, 'Scaling coverage')
    for delta in deltas:
        with localcontext() as ctx:
            ctx.prec = 50
            a = D(lookup[delta['larger_model']][delta['metric']]);b = D(lookup[delta['smaller_model']][delta['metric']])
            require(D(delta['delta_absolute']) == b - a and D(delta['delta_relative_pct']) == (b - a) / a * 100, 'Scaling math')
    for name, cost in [('PARETO_FRONTIER.csv', 'inference_ms_mean'), ('PARETO_VRAM.csv', 'peak_allocated_vram_mib')]:
        front = {x['model'] for x in read_csv(directory / 'metrics' / name)}
        for r in rows:
            dominated = False
            for q in rows:
                accuracy_gap = D(q['mask_map50_95']) - D(r['mask_map50_95']);cost_gap = D(q[cost]) - D(r[cost])
                if accuracy_gap >= 0 and cost_gap <= 0 and (accuracy_gap > 0 or cost_gap < 0):
                    dominated = True
            require((r['model'] in front) != dominated, f'{name}: incorrect dominance {r["model"]}')
    return {'status': 'PASS', 'models': 17, 'family_counts': dict(Counter(r['family'] for r in rows)), 'per_sequence_rows': 68, 'scaling_pairs': 13, 'scaling_metric_rows': 143,
            'pareto_latency_models': [r['model'] for r in read_csv(directory / 'metrics/PARETO_FRONTIER.csv')], 'pareto_vram_models': [r['model'] for r in read_csv(directory / 'metrics/PARETO_VRAM.csv')],
            'inference_rerun': False, 'tier_benchmarks_rerun': False, 'measured_values_changed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-inputs', action='store_true')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--workspace', type=Path, default=WORKSPACE)
    parser.add_argument('--output-dir', type=Path, default=MASTER)
    args = parser.parse_args()
    inputs = validate_inputs(args.workspace.resolve())
    if args.check_inputs:
        print('[MASTER] source and protocol precondition: PASS (17/17; five complete tiers)')
        return
    if not args.validate:
        for name, (header, rows) in products(inputs).items():
            write_new(args.output_dir / 'metrics' / name, csv_bytes(header, rows))
        # Reproducible manifest: no wall-clock timestamp or output self-hash cycle.
        manifest = {'study_id': inputs['schema']['study_id'], 'schema_version': inputs['schema']['schema_version'], 'stage': 'MASTER_SYNTHESIS',
                    'source_tiers': inputs['sources'], 'control_sha256': inputs['controls'], 'builder_sha256': sha(Path(__file__)),
                    'synthesis_only': True, 'inference_rerun': False, 'tier_benchmarks_rerun': False, 'measured_values_changed': False, 'benchmark_metrics_recalculated': False,
                    'calculation_policy': {'delta': 'smaller - larger', 'relative_pct': '(smaller - larger) / larger * 100', 'fraction_delta_pp': 'delta_absolute * 100',
                                           'pareto_latency': 'maximize mask_map50_95; minimize inference_ms_mean', 'pareto_vram': 'maximize mask_map50_95; minimize peak_allocated_vram_mib',
                                           'dominance': 'no worse in both objectives and strictly better in at least one', 'near_tie_screen_absolute_map_gap': '0.001', 'near_tie_screen_is_statistical_test': False},
                    'warnings': ['All five source tiers retain PASS_WITH_WARNINGS.', 'Largest superseded contaminated timing excluded; only clean_repetition sources imported.',
                                 'No statistical significance test; correlated video frames.', 'E/X and C/L are available size categories, not equal capacities.', 'VRAM and pipeline need not scale monotonically with parameter count.',
                                 'Raw predictions and prediction metadata were not opened; their declared hashes remain upstream evidence.'],
                    'artifact_sha256': {f'metrics/{name}': sha(args.output_dir / 'metrics' / name) for name in products(inputs)}}
        write_new(args.output_dir / 'provenance/SOURCE_MANIFEST.json', (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode())
    validation = validate_outputs(inputs, args.output_dir)
    manifest = load_json(args.output_dir / 'provenance/SOURCE_MANIFEST.json')
    require(manifest['source_tiers'] == inputs['sources'] and manifest['control_sha256'] == inputs['controls'], 'Source provenance mismatch')
    require(manifest['builder_sha256'] == sha(Path(__file__)), 'Builder provenance mismatch')
    require(manifest['synthesis_only'] is True and manifest['inference_rerun'] is False and manifest['measured_values_changed'] is False, 'Invalid synthesis provenance')
    require(manifest['artifact_sha256'] == {f'metrics/{name}': sha(args.output_dir / 'metrics' / name) for name in products(inputs)}, 'Output provenance mismatch')
    print('[MASTER] ' + ('validation' if args.validate else 'synthesis') + ': PASS (17 models; 13 scaling pairs; 68 sequence rows)')
    print(json.dumps(validation, ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, OSError) as error:
        print(f'[MASTER] FAIL: {error}', file=sys.stderr)
        sys.exit(1)
