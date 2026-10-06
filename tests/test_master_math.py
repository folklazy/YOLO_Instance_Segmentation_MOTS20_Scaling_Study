"""Test scientific calculation edge cases without inference or prediction access."""
import csv
from decimal import Decimal
import importlib.util
from pathlib import Path
import tempfile
import unittest

PATH = Path(__file__).resolve().parents[1] / 'scripts/build_master.py'
spec = importlib.util.spec_from_file_location('build_master', PATH)
build = importlib.util.module_from_spec(spec);spec.loader.exec_module(build)


class MasterMathTests(unittest.TestCase):
    def test_strict_pareto_and_equal_points(self):
        data = [{'model': name, 'mask_map50_95': accuracy, 'inference_ms_mean': latency}
                for name, accuracy, latency in [('A', '.6', '20'), ('B', '.5', '10'), ('C', '.5', '20'), ('D', '.6', '20'), ('E', '.4', '30')]]
        self.assertEqual({x['model'] for x in build.pareto(data, 'inference_ms_mean')}, {'A', 'B', 'D'})
        self.assertFalse(build.dominates(data[0], data[3], 'inference_ms_mean'))

    def test_unrounded_values_determine_frontier(self):
        a = {'model': 'A', 'mask_map50_95': '.50000001', 'inference_ms_mean': '10'}
        b = {'model': 'B', 'mask_map50_95': '.50000000', 'inference_ms_mean': '10'}
        self.assertEqual([x['model'] for x in build.pareto([a, b], 'inference_ms_mean')], ['A'])

    def test_delta_sign_relative_denominator_and_pp(self):
        rows = []
        for model, tier, accuracy, latency in [('YOLO26x-Seg', 'largest', '.6', '20'), ('YOLO26l-Seg', 'second_largest', '.57', '10')]:
            rows.append({'tier': tier, 'family': 'YOLO26', 'model': model, 'checkpoint': model + '.pt', 'mask_map50_95': accuracy,
                         **{key: latency if 'ms' in key else '10' if unit != 'fraction' else '.5' for key, unit in build.DELTA_FIELDS if key != 'mask_map50_95'},
                         'gflops': '20'})
        timing = [{'model': r['model'], 'stage': stage, 'mean_ms': '1'} for r in rows for stage in ['preprocessing', 'postprocessing', 'rle_preparation']]
        inputs = {'overall': rows, 'sequence': [], 'timing': timing, 'complexity': [], 'schema': {'schemas': {'TIER_RESULTS': list(rows[0]), 'PER_SEQUENCE_RESULTS': [], 'TIMING_SUMMARY': [], 'MODEL_COMPLEXITY': []}}}
        # products includes tier winners; use a subset of the configured tier map.
        original = build.TIERS
        try:
            build.TIERS = original[:2]
            result = build.products(inputs)['SCALING_DELTAS.csv'][1]
        finally:
            build.TIERS = original
        acc = next(x for x in result if x['metric'] == 'mask_map50_95')
        speed = next(x for x in result if x['metric'] == 'inference_ms_mean')
        self.assertEqual(Decimal(acc['delta_absolute']), Decimal('-.03'))
        self.assertEqual(Decimal(acc['delta_percentage_points']), Decimal('-3'))
        self.assertEqual(Decimal(acc['delta_relative_pct']), Decimal('-5'))
        self.assertEqual(Decimal(speed['delta_relative_pct']), Decimal('-50'))

    def test_schema_order_is_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'schema.csv';path.write_text('b,a\n2,1\n')
            with self.assertRaisesRegex(ValueError, 'Schema mismatch'):
                build.read_csv(path, ['a', 'b'])

    def test_different_existing_artifact_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'result.csv';path.write_bytes(b'original')
            with self.assertRaisesRegex(ValueError, 'Refusing to overwrite'):
                build.write_new(path, b'changed')
            self.assertEqual(path.read_bytes(), b'original')

    def test_count_and_metric_inconsistency_is_rejected(self):
        row = {k: '.5' for k in build.ACCURACY}
        row.update(tp='10', fp='10', fn='10', gt_instances='20')
        build.fraction_checks(row, 'valid')
        row['recall'] = '.6'
        with self.assertRaisesRegex(ValueError, 'recall arithmetic'):
            build.fraction_checks(row, 'invalid')


if __name__ == '__main__':unittest.main()
