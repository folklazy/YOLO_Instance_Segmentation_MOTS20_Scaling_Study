"""Render purpose-specific Master documents using final synthesis artifacts only."""
import argparse
import hashlib
from collections import Counter
from decimal import Decimal as D
import csv
import json
from pathlib import Path
from build_master import MASTER, TIERS, FAMILIES, WINNERS, write_new

ROWS = list(csv.DictReader((MASTER / 'metrics/MASTER_17_MODELS.csv').open()))
LOOKUP = {r['model']: r for r in ROWS}
DELTAS = list(csv.DictReader((MASTER / 'metrics/SCALING_DELTAS.csv').open()))
TIER_WINNERS = list(csv.DictReader((MASTER / 'metrics/TIER_WINNERS.csv').open()))
NEAR = list(csv.DictReader((MASTER / 'metrics/NEAR_TIES.csv').open()))
LATENCY = list(csv.DictReader((MASTER / 'metrics/PARETO_FRONTIER.csv').open()))
VRAM = list(csv.DictReader((MASTER / 'metrics/PARETO_VRAM.csv').open()))
STAGES = list(csv.DictReader((MASTER / 'metrics/PLOT_READY_RESULTS.csv').open()))
HUMAN = {t: n for t, n in zip([x[0] for x in TIERS], ['Largest (X/E)', 'Second-largest (L/C)', 'Medium (M)', 'Small (S)', 'Nano (N)'])}
CLAIMS = []
DOC = ''
README_ARCHIVE = None


def formatted(value, field, decimals=None, scale=1, signed=False):
    if decimals is None:
        decimals = 0 if field in ['parameters', 'tp', 'fp', 'fn'] else 2 if field in ['peak_allocated_vram_mib', 'checkpoint_mb'] else 3 if field in ['inference_ms_mean', 'pipeline_ms_mean', 'fps', 'gflops'] else 6
    return format(D(value) * D(scale), ('+' if signed else '') + (',' if field == 'parameters' else '') + f'.{decimals}f')


def cell(source, keys, field, decimals=None, scale=1, signed=False):
    row = next(r for r in list(csv.DictReader((MASTER / source).open())) if all(r[k] == v for k, v in keys.items()))
    display = formatted(row[field], field, decimals, scale, signed)
    CLAIMS.append({'document': DOC, 'source': source, 'keys': keys, 'field': field, 'decimals': decimals, 'scale': scale, 'signed': signed, 'display': display})
    return display


def v(model, field, decimals=None, scale=1):
    return cell('metrics/MASTER_17_MODELS.csv', {'model': model}, field, decimals, scale)


def delta(a, b, metric, kind='delta_absolute', decimals=3):
    return cell('metrics/SCALING_DELTAS.csv', {'larger_model': a, 'smaller_model': b, 'metric': metric}, kind, decimals, signed=True)


def near_value(row):
    return cell('metrics/NEAR_TIES.csv', {'model_a': row['model_a'], 'model_b': row['model_b']}, 'map_abs_gap', 9)


def table(headers, data):
    return '| ' + ' | '.join(headers) + ' |\n| ' + ' | '.join(['---'] * len(headers)) + ' |\n' + ''.join('| ' + ' | '.join(str(x) for x in row) + ' |\n' for row in data)


def result_table(data=ROWS, thai=False, compact=False):
    keys = ['mask_map50_95', 'ap75', 'recall', 'inference_ms_mean', 'pipeline_ms_mean', 'fps', 'peak_allocated_vram_mib']
    heads = ['Model', 'Tier', 'Mask mAP50-95', 'AP75', 'Recall', 'Inference ms', 'Pipeline ms', 'FPS', 'VRAM MiB']
    if not compact:
        keys += ['parameters', 'checkpoint_mb'];heads += ['Parameters', 'Checkpoint MB']
    return table(heads, [[r['model'], HUMAN[r['tier']]] + [v(r['model'], k) for k in keys] for r in data])


def tier_winners_table():
    labels = ['Tier', 'mAP', 'AP75', 'Recall', 'Inference', 'Pipeline / FPS', 'VRAM']
    data = []
    for tier, _, _ in TIERS:
        w = {r['metric']: r for r in TIER_WINNERS if r['tier'] == tier}
        cells = [HUMAN[tier]]
        for key in ['mask_map50_95', 'ap75', 'recall', 'inference_ms_mean', 'pipeline_ms_mean', 'peak_allocated_vram_mib']:
            r = w[key];cells.append(r['model'] + ' (' + v(r['model'], key) + ')')
        data.append(cells)
    return table(labels, data)


def delta_table(family=None):
    pairs = list(dict.fromkeys((r['larger_model'], r['smaller_model']) for r in DELTAS if family is None or r['family'] == family))
    data = []
    for a, b in pairs:
        data.append([a + ' → ' + b, delta(a, b, 'mask_map50_95', 'delta_percentage_points'),
                     delta(a, b, 'inference_ms_mean', 'delta_relative_pct', 2), delta(a, b, 'pipeline_ms_mean', 'delta_relative_pct', 2),
                     delta(a, b, 'peak_allocated_vram_mib', 'delta_relative_pct', 2), delta(a, b, 'parameters', 'delta_relative_pct', 2)])
    return table(['Larger → smaller', 'ΔmAP pp', 'ΔInference %', 'ΔPipeline %', 'ΔVRAM %', 'ΔParameters %'], data)


def near_table():
    return table(['Model A', 'Model B', 'Absolute mAP gap'], [[r['model_a'], r['model_b'], near_value(r)] for r in NEAR])


def selection_table(thai=False):
    if thai:
        return table(['Priority', 'Candidate', 'หลักฐานและเงื่อนไข'], [
            ['Accuracy', 'YOLO26x-Seg', f"mAP {v('YOLO26x-Seg','mask_map50_95')}; inference {v('YOLO26x-Seg','inference_ms_mean')} ms"],
            ['Pipeline / Low VRAM', 'YOLO26l-Seg', f"pipeline {v('YOLO26l-Seg','pipeline_ms_mean')} ms; VRAM {v('YOLO26l-Seg','peak_allocated_vram_mib')} MiB; mAP {v('YOLO26l-Seg','mask_map50_95')}"],
            ['ลดเวลาของ forward โดยยังรักษา mAP', 'YOLO26s-Seg', f"inference {v('YOLO26s-Seg','inference_ms_mean')} ms; mAP {v('YOLO26s-Seg','mask_map50_95')}; อยู่บน Pareto ด้าน inference"],
            ['Nano ที่เน้น mask accuracy', 'YOLO26n-Seg', f"mAP {v('YOLO26n-Seg','mask_map50_95')}; Recall {v('YOLO26n-Seg','recall')} ต่ำสุดใน Nano"],
            ['Inference ต่ำสุด', 'YOLOv8n-Seg', f"inference {v('YOLOv8n-Seg','inference_ms_mean')} ms; mAP {v('YOLOv8n-Seg','mask_map50_95')}; pipeline ไม่เร็วสุด"],
        ])
    return table(['Priority', 'Candidate', 'Evidence and condition'], [
        ['Accuracy', 'YOLO26x-Seg', f"mAP {v('YOLO26x-Seg','mask_map50_95')}; inference {v('YOLO26x-Seg','inference_ms_mean')} ms"],
        ['Pipeline / Low VRAM / balanced constraints', 'YOLO26l-Seg', f"pipeline {v('YOLO26l-Seg','pipeline_ms_mean')} ms; VRAM {v('YOLO26l-Seg','peak_allocated_vram_mib')} MiB; mAP {v('YOLO26l-Seg','mask_map50_95')}"],
        ['Shorter forward latency with retained accuracy', 'YOLO26s-Seg', f"inference {v('YOLO26s-Seg','inference_ms_mean')} ms; mAP {v('YOLO26s-Seg','mask_map50_95')}; latency-frontier candidate"],
        ['Nano mask accuracy', 'YOLO26n-Seg', f"mAP {v('YOLO26n-Seg','mask_map50_95')}; Recall {v('YOLO26n-Seg','recall')} is lowest within Nano"],
        ['Minimum inference latency', 'YOLOv8n-Seg', f"inference {v('YOLOv8n-Seg','inference_ms_mean')} ms; mAP {v('YOLOv8n-Seg','mask_map50_95')}; not the pipeline winner"],
    ])


def family_table():
    data = []
    for family in FAMILIES:
        subset = [r for r in ROWS if r['family'] == family];a, b = subset[0], subset[-1]
        data.append([family, len(subset), a['model'] + ' → ' + b['model'],
                     v(a['model'], 'mask_map50_95') + ' → ' + v(b['model'], 'mask_map50_95'),
                     v(a['model'], 'inference_ms_mean') + ' → ' + v(b['model'], 'inference_ms_mean')])
    return table(['Family', 'Models', 'Largest → smallest available', 'mAP endpoints', 'Inference endpoints (ms)'], data)


def publish(name, body):
    path = MASTER / name
    payload = body.encode('utf-8')
    if name == 'README.md' and path.exists() and path.read_bytes() != payload and README_ARCHIVE is not None:
        archive = README_ARCHIVE.resolve()
        if not archive.is_relative_to((MASTER / 'reports/archive').resolve()) or archive.read_bytes() != path.read_bytes():
            raise ValueError('README replacement requires its exact existing archive')
        path.write_bytes(payload)
        return
    write_new(path, payload)


def main():
    global DOC, README_ARCHIVE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archived-readme', type=Path, help='Exact archived current README for an explicitly authorized first migration')
    README_ARCHIVE = parser.parse_args().archived_readme
    DOC = 'MASTER_RESULTS.md'
    body = '''# Complete YOLO Instance Segmentation Scaling Results on MOTS20

## 1. Study Status

**COMPLETE — PASS WITH WARNINGS.** Five completed compatible tiers supply exactly 17 unique official pretrained segmentation checkpoints: YOLO26 = 5, YOLO11 = 5, YOLOv8 = 5, YOLOv9 = 2. Each checkpoint covers the same 2,862 frames and 26,894 frame-level Person GT instances. No training, fine-tuning, inference or tier rerun was performed for synthesis. Source warnings are retained.

The frozen FP32/Tesla T4 protocol and units are in [METHODOLOGY_REFERENCE.md](METHODOLOGY_REFERENCE.md) and [DATA_SCHEMA.md](DATA_SCHEMA.md). [SOURCE_MANIFEST.json](provenance/SOURCE_MANIFEST.json) records repository revisions, canonical CSV hashes, original measurement sources and compatibility checks. Mask AP is pooled across frames, not averaged across sequences.

## 2. Experiment Map

'''
    body += table(['Tier', 'Repository', 'Models', 'Canonical run'], [[HUMAN[t], f'[{slug}](https://github.com/folklazy/YOLO_{slug}_Seg_MOTS20_Benchmark)', len(expected), ROWS[next(i for i,r in enumerate(ROWS) if r['tier']==t)]['run_id']] for t, slug, expected in TIERS])
    body += '\nE/X and C/L are available size categories, not equal-capacity architectures.\n\n## 3. Master 17-Model Results\n\n' + result_table()
    body += '\nAll AP/Recall values are fractions. Inference/pipeline use ms/frame; FPS = 1000 / mean pipeline ms; VRAM is peak allocated MiB. Parameters are loaded checkpoint counts. The [complete CSV](metrics/MASTER_17_MODELS.csv) also retains AP50, Precision, F1, TP-only IoU/Dice, counts, GFLOPs and protocol/source fields.\n\n![17-model accuracy](plots/01_master_map_by_model.png)\n\n## 4. Tier Winners\n\n' + tier_winners_table()
    body += '\nPipeline and FPS have the same winner because FPS is derived from pipeline mean. The [winner CSV](metrics/TIER_WINNERS.csv) retains all seven categories and original values.\n\n## 5. Family Scaling\n\n' + family_table()
    body += '\n**Observation:** All 13 adjacent size reductions decrease mAP, forward latency and parameter count. Pipeline latency increases in 8 of the 13 reductions, and allocated VRAM increases in 7. **Interpretation:** checkpoint size predicts neither whole-pipeline speed nor allocator peak by itself.\n\n![Family accuracy](plots/02_family_accuracy_scaling.png)\n\n![Family inference](plots/03_family_inference_scaling.png)\n\n## 6. Scaling Deltas\n\nDelta = smaller − larger. Negative mAP pp means accuracy loss; negative latency/VRAM percentages mean reductions relative to the larger checkpoint. These columns describe distinct objectives, not a combined score.\n\n' + delta_table()
    body += '\n[SCALING_DELTAS.csv](metrics/SCALING_DELTAS.csv) retains absolute and relative changes for mAP, AP50, AP75, Recall, F1, inference, pipeline, FPS, VRAM, parameters and checkpoint MB.\n\n![Adjacent scaling](plots/10_adjacent_scaling_delta.png)\n\n## 7. Pareto Frontier\n\nA checkpoint is nondominated when no other checkpoint has at least its mAP and at most its cost, with one strict improvement. Use unrounded values. No weighted score is calculated.\n\n**Accuracy / inference:** ' + ', '.join(r['model'] for r in LATENCY) + '.\n\n' + result_table(LATENCY, compact=True)
    body += '\n**Accuracy / allocated VRAM:** ' + ', '.join(r['model'] for r in VRAM) + '.\n\n' + result_table(VRAM, compact=True)
    body += '\nThese are different objective sets. A model on the inference frontier can be dominated on the VRAM frontier. Dominance does not settle objectives omitted from the frontier, such as Recall, model-file size or a later deployment test.\n\n![Accuracy-latency Pareto](plots/09_pareto_accuracy_latency.png)\n\n## 8. Important Research Findings\n\n'
    body += f'''1. **Observation:** YOLO26 leads mAP and AP75 in every tier. YOLO26x reaches mAP {v('YOLO26x-Seg','mask_map50_95')}. **Interpretation:** these checkpoints are accuracy candidates under this protocol; this does not isolate an architectural cause.
2. **Observation:** YOLO26l has the fastest pipeline ({v('YOLO26l-Seg','pipeline_ms_mean')} ms), highest FPS ({v('YOLO26l-Seg','fps')}) and lowest allocated VRAM ({v('YOLO26l-Seg','peak_allocated_vram_mib')} MiB) across all models. **Interpretation:** it is a strong candidate when those system constraints matter, despite being larger than M/S/N checkpoints.
3. **Observation:** YOLO26x → YOLO26l changes mAP by {delta('YOLO26x-Seg','YOLO26l-Seg','mask_map50_95','delta_percentage_points')} pp and inference by {delta('YOLO26x-Seg','YOLO26l-Seg','inference_ms_mean','delta_relative_pct',2)}%. **Interpretation:** this measured reduction preserves most accuracy while reducing forward time substantially, if that accuracy loss is acceptable.
4. **Observation:** YOLO26s has mAP {v('YOLO26s-Seg','mask_map50_95')}, almost the same as YOLO11x and YOLOv9e, with inference {v('YOLO26s-Seg','inference_ms_mean')} ms. **Interpretation:** cross-generation size labels do not uniquely determine aggregate accuracy. A near tie does not imply identical errors or capacity.
5. **Observation:** s → n is the largest adjacent mAP drop within YOLO26, YOLO11 and YOLOv8: {delta('YOLO26s-Seg','YOLO26n-Seg','mask_map50_95','delta_percentage_points')}, {delta('YOLO11s-Seg','YOLO11n-Seg','mask_map50_95','delta_percentage_points')} and {delta('YOLOv8s-Seg','YOLOv8n-Seg','mask_map50_95','delta_percentage_points')} pp. **Interpretation:** the final size reduction deserves an explicit accuracy budget rather than a default assumption that smaller is better.
6. **Observation:** YOLOv8n has the fastest inference ({v('YOLOv8n-Seg','inference_ms_mean')} ms), but its pipeline is {v('YOLOv8n-Seg','pipeline_ms_mean')} ms. **Interpretation:** forward-only ranking does not establish application throughput.

## 9. Near Ties

A descriptive screen uses absolute mAP gap ≤0.001 (0.1 pp). It identifies reporting candidates, not statistical equivalence or practical interchangeability. No significance test was performed.

''' + near_table()
    body += '\nSame-tier visual checks remain in each tier’s PRESENTATION_SUMMARY_TH.md. The cross-tier near-tie pairs above were not newly inspected visually in this synthesis; aggregate score proximity alone does not establish identical prediction behavior.\n\n## 10. Anomalies\n\n'
    body += f'''- YOLO26l → YOLO26m reduces inference by {delta('YOLO26l-Seg','YOLO26m-Seg','inference_ms_mean','delta_relative_pct',2)}%, while pipeline changes by {delta('YOLO26l-Seg','YOLO26m-Seg','pipeline_ms_mean','delta_relative_pct',2)}% and allocated VRAM by {delta('YOLO26l-Seg','YOLO26m-Seg','peak_allocated_vram_mib','delta_relative_pct',2)}%. These are measured directions, not proof that small timing gaps are reproducible superiority.
- In Nano, YOLO26n leads AP75 but has Recall {v('YOLO26n-Seg','recall')}, below YOLO11n ({v('YOLO11n-Seg','recall')}) and YOLOv8n ({v('YOLOv8n-Seg','recall')}). Aggregate mask accuracy and fixed-threshold coverage differ.
- Postprocessing means increase from {cell('metrics/PLOT_READY_RESULTS.csv',{'model':'YOLOv8x-Seg'},'postprocessing_ms_mean',3)} ms for YOLOv8x to {cell('metrics/PLOT_READY_RESULTS.csv',{'model':'YOLOv8n-Seg'},'postprocessing_ms_mean',3)} ms for YOLOv8n. This stage trend is consistent with the nonmonotonic pipeline results; the synthesis does not establish why output workloads differ.
- Separately measured RLE preparation is excluded from every pipeline/FPS value. Adding it would change the throughput definition; the existing FPS values must not be presented as mask-saving throughput.

## 11. Limitations

This is frame-level Person instance segmentation, not MOTS tracking. GT counts are annotations across frames, not unique people. Consecutive frames are correlated. No statistical significance, equivalence or robustness test was conducted. TP-only IoU/Dice exclude unmatched instances and can compare different matched subsets. E/X and C/L are not equal capacities; pretraining and checkpoint construction also confound architectural causality.

Timing comes from accepted clean repetitions under one frozen environment but different sessions; contamination controls do not remove all session variation. Pipeline excludes image decode, loading, RLE preparation and output writing. VRAM is an allocator peak, not total GPU memory or an OOM limit. Family curves connect observed checkpoints and are not a fitted scaling law.

MOTS20 does not establish CCTV robustness to blur, lighting, camera angle or occlusion severity, nor deployment readiness. Existing tier qualitative cases are diagnostic selections, not representative samples. The two Pareto fronts cover only their named objectives.

## 12. Candidate Models for CCTV Robustness Evaluation

''' + selection_table()
    body += '\nThese are conditional candidates for later CCTV robustness evaluation. A single universal winner is not selected. The next step requires a separately authorized robustness dataset/protocol and deployment timing that includes the required I/O and output stages. This synthesis starts no additional experiment.\n\nFurther reading: [Thai research insights](RESEARCH_INSIGHTS_TH.md), [executive summary](EXECUTIVE_SUMMARY_TH.md), [meeting summary](MEETING_SUMMARY_TH.md), [metric guide](METRIC_GUIDE_TH.md).\n'
    publish(DOC, body)

    DOC = 'RESEARCH_INSIGHTS_TH.md'
    body = f'''# สรุปภาพรวม YOLO Instance Segmentation Model Scaling

## สรุปใน 1 นาที

- ทดสอบ pretrained segmentation 17 โมเดล ครบ 5 tiers บน MOTS20 2,862 เฟรม และ Person GT 26,894 instances โดยไม่มี fine-tuning
- YOLO26 นำ Mask mAP50-95 และ AP75 ทุก tier; YOLO26x ได้ mAP สูงสุด {v('YOLO26x-Seg','mask_map50_95')}
- YOLOv8n inference เร็วสุด {v('YOLOv8n-Seg','inference_ms_mean')} ms แต่ pipeline ไม่เร็วสุด
- YOLO26l pipeline เร็วสุด {v('YOLO26l-Seg','pipeline_ms_mean')} ms, FPS สูงสุด {v('YOLO26l-Seg','fps')} และ VRAM ต่ำสุด {v('YOLO26l-Seg','peak_allocated_vram_mib')} MiB
- การลดขนาดทั้ง 13 คู่ทำให้ forward เร็วขึ้น แต่ pipeline เพิ่มใน 8 คู่ และ VRAM เพิ่มใน 7 คู่
- s → n เป็นช่วงที่ mAP ตกมากที่สุดภายใน YOLO26/YOLO11/YOLOv8; ต้องกำหนด accuracy budget ก่อนเลือก
- มี near ties ทั้งใน tier เดียวกันและข้าม tier; ยังไม่มีการทดสอบ statistical significance หรือ CCTV robustness

## 1. ภาพรวมการทดลอง

รวม canonical CSV ของ Largest, Second-largest, Medium, Small และ Nano ภายใต้ config, evaluator, GT, preprocessing และวิธีจับเวลาเดียวกัน ใช้ FP32, imgsz640, batch1, Tesla T4 และเฉพาะ clean timing 3 รอบต่อโมเดล ขั้น Master ไม่เปิด raw predictions และไม่รัน inference ซ้ำ ตัวเลขเต็มอยู่ใน [MASTER_RESULTS.md](MASTER_RESULTS.md) และ [CSV](metrics/MASTER_17_MODELS.csv); วิธีทดลองอยู่ใน [methodology](METHODOLOGY_REFERENCE.md)

## 2. Winner ของแต่ละ Tier

''' + tier_winners_table()
    body += '\nค่าด้าน accuracy เป็น fraction; เวลาเป็น ms/frame และ VRAM เป็น peak allocated MiB ผู้ชนะ pipeline กับ FPS ตรงกันเพราะ FPS คำนวณจาก pipeline mean\n\n## 3. ภาพรวมทั้ง 17 Models\n\n' + result_table(compact=True)
    body += '\n![ภาพรวมความแม่นยำ](plots/01_master_map_by_model.png)\n\nหน่วยและตัวเลขเพิ่มเติม เช่น AP50, Precision, F1, TP-only quality และ GFLOPs อยู่ใน CSV หลัก ไม่ใช้คะแนนถ่วงน้ำหนักรวมหลายด้าน\n'
    notes = {
        'YOLO26': 'YOLO26x → l แลก accuracy ค่อนข้างน้อยกับ inference ที่ลดประมาณครึ่งหนึ่ง ขณะที่ l → m ทำให้ forward สั้นลง แต่ pipeline และ VRAM สูงขึ้นตามค่าที่วัด ส่วน m → s เหมาะพิจารณาเมื่อ forward budget สำคัญและยอมรับ mAP ที่ลดลงได้',
        'YOLO11': 'x → l ลด forward มากโดย mAP ลดน้อยกว่าช่วง m → s และ s → n หลัง l ลงมา pipeline และ VRAM เพิ่มทุกขั้น แม้ parameters ลดลง จึงไม่ควรใช้ชื่อขนาดแทนผลของระบบ',
        'YOLOv8': 'x → l สูญเสีย mAP น้อยที่สุดใน 13 คู่ แต่ไม่ใช่ accuracy leader ของ tier; l → m ลดทั้ง inference และ pipeline ส่วน m → s และ s → n ทำให้ forward เร็วขึ้นแต่ pipeline ช้าลง',
        'YOLOv9': 'มีเพียง e และ c ที่เป็น segmentation checkpoints ที่ถูกต้องในชุดนี้ e → c ลดเวลาและ parameters มาก แต่ VRAM ลดเพียงเล็กน้อย จึงห้ามเติมรุ่น m/s/n หรือเปรียบชื่อ e/c ว่า capacity เท่ากับ x/l'}
    for index, family in enumerate(FAMILIES, 4):
        body += f'\n## {index}. {family} Scaling\n\n' + delta_table(family)
        body += '\n**Observation:** ตารางแสดงค่าที่เปลี่ยนเมื่อขยับไป checkpoint ที่เล็กกว่า โดย ΔmAP เป็น percentage points และคอลัมน์ % ใช้ค่ารุ่นใหญ่เป็นฐาน\n\n**Interpretation:** ' + notes[family] + '\n'
    body += '\n## 8. เทียบ Generation ใน Tier เดียวกัน\n\n**Observation:** YOLO26 มี mAP/AP75 สูงสุดทั้งห้า tier แต่ Recall ใน Nano สูงสุดเป็น YOLOv8n การแบ่ง tier ควบคุมกลุ่มขนาดที่มีให้ใช้ ไม่ได้ทำให้จำนวน parameters หรือ pretraining เท่ากัน\n\n**Interpretation:** ใช้ผลเพื่อเลือก checkpoint ในเงื่อนไขนี้ได้ แต่ยังสรุปสาเหตุว่า architecture รุ่นใหม่ดีกว่าเสมอไม่ได้ รวมทั้งไม่ควรเรียกทุกมิติของ Nano ว่า YOLO26 ชนะ\n\n## 9. Accuracy vs Speed\n\nPareto ด้าน mAP–inference มี ' + ', '.join(r['model'] for r in LATENCY) + '\n\n![ความแม่นยำกับ inference](plots/07_accuracy_vs_inference.png)\n\n**Observation:** โมเดลบนเส้น Pareto มีข้อแลกเปลี่ยนต่างกัน; forward เร็วสุดกับ pipeline เร็วสุดเป็นคนละโมเดล\n\n**Interpretation:** ต้องระบุว่าต้องการลดเวลา forward หรือ pipeline ก่อน การเลือกจาก FPS ต้องยอมรับว่าค่านี้ยังไม่รวม decode/RLE/เขียนไฟล์\n\n## 10. Accuracy vs VRAM\n\nPareto ด้าน mAP–VRAM เหลือ YOLO26x และ YOLO26l โดย YOLO26l ใช้ VRAM ต่ำสุดในทั้ง 17 โมเดล\n\n![ความแม่นยำกับ VRAM](plots/08_accuracy_vs_vram.png)\n\n**Interpretation:** แม้โมเดลเล็กบางรุ่นจะอยู่บน Pareto ของ inference แต่ไม่ได้อยู่บน Pareto ของ allocated VRAM เพราะ peak memory ที่วัดขึ้นกับ runtime workload ด้วย ผลนี้ไม่ใช่ข้อพิสูจน์เชิงสาเหตุของการจัดสรรหน่วยความจำ\n\n## 11. จุดที่ลดขนาดแล้วคุ้ม\n\nคำว่า “คุ้ม” ต่อไปนี้เป็นการตีความตามข้อจำกัด ไม่มีคะแนนรวม: x → l ของ YOLO26/YOLO11/YOLOv8 และ e → c ของ YOLOv9 ลด inference มากโดย mAP ลดไม่เกิน 2 pp ในค่าที่วัด หากยอมรับ accuracy loss ได้ ส่วน YOLO26l เด่นเมื่อให้น้ำหนัก pipeline/VRAM เพราะนำสองด้านนี้ทั้งชุด\n'
    body += f"\nYOLO26x → l: ΔmAP {delta('YOLO26x-Seg','YOLO26l-Seg','mask_map50_95','delta_percentage_points')} pp, Δinference {delta('YOLO26x-Seg','YOLO26l-Seg','inference_ms_mean','delta_relative_pct',2)}%, ΔVRAM {delta('YOLO26x-Seg','YOLO26l-Seg','peak_allocated_vram_mib','delta_relative_pct',2)}% รุ่น Small อย่าง YOLO26s เป็นอีกตัวเลือกสำหรับ forward ที่สั้นลง แต่ไม่ใช่ global pipeline/VRAM winner\n"
    body += f'''\n## 12. จุดที่ Accuracy ตกชัด

**Observation:** s → n ลด mAP {delta('YOLO26s-Seg','YOLO26n-Seg','mask_map50_95','delta_percentage_points')} pp ใน YOLO26, {delta('YOLO11s-Seg','YOLO11n-Seg','mask_map50_95','delta_percentage_points')} pp ใน YOLO11 และ {delta('YOLOv8s-Seg','YOLOv8n-Seg','mask_map50_95','delta_percentage_points')} pp ใน YOLOv8 ซึ่งเป็น adjacent drop มากที่สุดภายในแต่ละ family

**Interpretation:** การลดรุ่นขั้นสุดท้ายควรเทียบกับค่าความแม่นยำขั้นต่ำที่งานยอมรับได้ คำว่า “ตกชัด” หมายถึงช่องว่างเชิงพรรณนาในค่าที่วัด ไม่ใช่ statistical significance

## 13. Near Ties

ใช้เกณฑ์เชิงพรรณนา |ΔmAP| ≤0.001 หรือ 0.1 pp เพื่อคัดคู่ที่ควรอ่านอย่างระมัดระวัง ไม่ใช่การทดสอบ equivalence

''' + near_table()
    body += '\nYOLO26s ใกล้ YOLO11x/YOLOv9e มากใน mAP แต่ capacity, latency และ error behavior อาจต่างกัน คู่ข้าม tier ยังไม่ได้ตรวจภาพใหม่ใน Master; ใช้ภาพใน tier เป็นหลักฐานเฉพาะของคู่ที่มีอยู่เท่านั้น\n\n## 14. สิ่งผิดปกติ\n\n- pipeline เพิ่มใน 8/13 ขั้นที่ลดขนาด และ allocated VRAM เพิ่มใน 7/13 ขั้น ทั้งที่ parameters ลดทุกขั้น\n- YOLO26l มี pipeline และ VRAM ดีกว่ารุ่นเล็กกว่าทั้งหมดในค่าที่วัด; ความต่างเวลาเล็กน้อยไม่ได้ผ่าน significance test\n- YOLO26n ชนะ mAP/AP75 แต่ Recall ต่ำสุดใน Nano; สะท้อนว่าความแม่น mask กับความครบถ้วนตาม fixed threshold เป็นคนละเรื่อง\n- postprocessing และ RLE preparation เป็นคนละ stage; RLE ไม่ได้รวมใน FPS และห้ามนำ diagnostic postprocess มาบวกซ้ำ\n\n## 15. ถ้าต้องเลือกโมเดลตามโจทย์\n\n' + selection_table(thai=True)
    body += '\nทั้งหมดเป็น candidate for later CCTV robustness evaluation โดยมีเงื่อนไขต่างกัน ไม่ใช่คำแนะนำ deployment ขั้นสุดท้าย\n\n## 16. สิ่งที่ผลนี้บอกได้\n\nอันดับและข้อแลกเปลี่ยนของ pretrained checkpoints บน Person masks ใน MOTS20 ภายใต้ protocol เดียวกัน รวมถึงการเปลี่ยนเชิงพรรณนาเมื่อขนาดเล็กลง และชุด nondominated candidates ภายใต้วัตถุประสงค์ที่ระบุ\n\n## 17. สิ่งที่ผลนี้ยังบอกไม่ได้\n\nความเป็นเหตุจาก architecture, statistical significance, ความเท่ากันของ error patterns, MOTS tracking, ความทนต่อ blur/low light/มุมกล้อง/ระดับ occlusion หรือความพร้อมใช้ CCTV จริง GT เป็น instance รายเฟรม ภาพวิดีโอสัมพันธ์กัน และ TP-only quality ใช้เฉพาะ instances ที่ match ได้\n\n## 18. ขั้นตอนถัดไป\n\nกำหนดชุดข้อมูลและเกณฑ์ CCTV robustness แยกต่างหาก เลือก candidate ตาม accuracy/forward/pipeline/memory budget และวัด deployment pipeline ที่รวมขั้นตอนที่ใช้งานจริง งาน Master นี้หยุดที่ผลสังเคราะห์ ไม่เริ่มการทดลองใหม่\n\n[Executive summary](EXECUTIVE_SUMMARY_TH.md) · [Meeting summary](MEETING_SUMMARY_TH.md) · [Metric guide](METRIC_GUIDE_TH.md) · [Source provenance](provenance/SOURCE_MANIFEST.json)\n'
    publish(DOC, body)

    DOC = 'EXECUTIVE_SUMMARY_TH.md'
    body = f'''# สรุปผลสำหรับตัดสินใจ — YOLO MOTS20 Scaling Study

## วัตถุประสงค์และขอบเขต

เปรียบเทียบ pretrained YOLO Instance Segmentation ตั้งแต่ขนาดใหญ่สุดถึง Nano เพื่อเลือก candidate ตาม accuracy, speed และ memory ทดสอบครบ 17 โมเดล: YOLO26/YOLO11/YOLOv8 family ละ 5 และ YOLOv9 อีก 2 บน MOTS20 2,862 เฟรมกับ Person GT 26,894 instances ภายใต้ protocol เดียวกัน ไม่มี training/fine-tuning ขั้น Master ใช้ CSV เดิมโดยไม่รัน inference ซ้ำ สถานะ **PASS WITH WARNINGS** จากคำเตือนของ source tiers

## ผลที่ควรรู้

| ด้าน | Model | ค่าที่วัด |
|---|---|---|
| Accuracy | YOLO26x-Seg | Mask mAP50-95 {v('YOLO26x-Seg','mask_map50_95')} |
| Inference | YOLOv8n-Seg | {v('YOLOv8n-Seg','inference_ms_mean')} ms/frame |
| Pipeline / FPS | YOLO26l-Seg | {v('YOLO26l-Seg','pipeline_ms_mean')} ms/frame / {v('YOLO26l-Seg','fps')} FPS |
| Low VRAM | YOLO26l-Seg | {v('YOLO26l-Seg','peak_allocated_vram_mib')} MiB peak allocated |

**Observation:** YOLO26 นำ mAP/AP75 ทุก tier แต่ใน Nano ผู้ชนะ Recall คือ YOLOv8n จึงไม่มีรุ่นที่ชนะทุก metric

## ขนาดที่ลดลงแลกอะไร

- YOLO26x → l: ΔmAP {delta('YOLO26x-Seg','YOLO26l-Seg','mask_map50_95','delta_percentage_points')} pp แต่ Δinference {delta('YOLO26x-Seg','YOLO26l-Seg','inference_ms_mean','delta_relative_pct',2)}%; เป็นจุดที่น่าสนใจเมื่อยอมรับ accuracy loss นี้ได้
- การลดขนาดทั้ง 13 คู่ทำให้ inference เร็วขึ้น แต่ pipeline เพิ่มใน 8 คู่และ allocated VRAM เพิ่มใน 7 คู่ ชื่อขนาดจึงไม่พอสำหรับตัดสินระบบ
- s → n เป็น accuracy drop มากที่สุดของ YOLO26/YOLO11/YOLOv8: {delta('YOLO26s-Seg','YOLO26n-Seg','mask_map50_95','delta_percentage_points')}, {delta('YOLO11s-Seg','YOLO11n-Seg','mask_map50_95','delta_percentage_points')} และ {delta('YOLOv8s-Seg','YOLOv8n-Seg','mask_map50_95','delta_percentage_points')} pp ตามลำดับ
- YOLO26s มี mAP ใกล้ YOLO11x/YOLOv9e มาก แม้เป็น Small; score ใกล้กันไม่แปลว่า output หรือ capacity เท่ากัน

## เลือก candidate ตามข้อจำกัด

''' + selection_table(thai=True)
    body += '\n**Interpretation:** YOLO26l เป็นจุดเริ่มต้นที่มีเหตุผลเมื่อ pipeline และ VRAM สำคัญร่วมกัน ส่วน YOLO26x ใช้เมื่อ accuracy สำคัญที่สุด และ YOLOv8n ใช้ทดสอบกรณีที่ forward latency เป็นข้อจำกัดหลัก ไม่ใช้ weighted score เพื่อประกาศผู้ชนะรวม\n\n## ข้อจำกัดและขั้นตอนถัดไป\n\nผลนี้เป็น frame-level segmentation ไม่ใช่ tracking หรือหลักฐานว่าโมเดลใดเหมาะที่สุดสำหรับ CCTV ภาพวิดีโอสัมพันธ์กันและไม่มี significance test; near ties เป็นเชิงพรรณนา รุ่น E/X และ C/L ไม่ได้มี capacity เท่ากัน และ TP-only quality ไม่นับคนที่ไม่ match\n\nPipeline/FPS ไม่รวม decode, RLE preparation และเขียนผล หน่วย VRAM เป็น allocator peak ไม่ใช่ GPU memory ทั้งหมด ขั้นต่อไปควรกำหนด robustness dataset และ deployment pipeline ให้ตรงงานจริง แล้วประเมิน candidates ด้วยเกณฑ์ที่อนุมัติแยกต่างหาก ขั้นนี้ไม่เริ่ม benchmark ใหม่\n\n[ผลเต็ม](MASTER_RESULTS.md) · [Research insights](RESEARCH_INSIGHTS_TH.md) · [ข้อมูลหลัก](metrics/MASTER_17_MODELS.csv)\n'
    publish(DOC, body)

    DOC = 'MEETING_SUMMARY_TH.md'
    body = '''# ทดลองประสิทธิภาพ YOLO Instance Segmentation บน MOTS20
## ตั้งแต่รุ่นขนาดใหญ่สุดจนถึง Nano

## ข้อมูลร่วมของการทดลอง

ครบ 17 official pretrained segmentation checkpoints: YOLO26/YOLO11/YOLOv8 family ละ 5 และ YOLOv9 อีก 2 ทุกโมเดลใช้ MOTS20 2,862 เฟรมกับ Person GT 26,894 instances, imgsz640, batch1, FP32, Tesla T4, preprocessing/evaluator/thresholds เดียวกัน ไม่มี training/fine-tuning ขั้น Master สังเคราะห์ CSV เท่านั้น สถานะ PASS WITH WARNINGS; ไม่รัน inference ซ้ำ

ค่าตารางเป็น mAP/Recall แบบ fraction, inference/pipeline เป็น ms/frame, FPS คำนวณจาก pipeline mean และ VRAM เป็น peak allocated MiB เวลา pipeline ไม่รวม decode, RLE และเขียนไฟล์ วิธีทดลองเต็มอยู่ใน [methodology](METHODOLOGY_REFERENCE.md)
'''
    tier_notes = {
        'largest': 'YOLO26x นำ mAP/AP75/Recall ส่วน YOLOv9e เร็วสุดทั้ง inference/pipeline และ VRAM ต่ำสุดใน tier นี้ YOLO11x/YOLOv9e มี mAP near-tied เชิงพรรณนา จึงควรเทียบข้อจำกัดด้านเวลาและความครบถ้วนร่วมด้วย',
        'second_largest': 'YOLO26l นำ mAP/AP75/Recall พร้อม pipeline และ VRAM ที่ดีที่สุดใน tier; YOLOv9c มี inference ต่ำสุด ขณะที่ผล pipeline ของสามรุ่นแรกใกล้กันและยังไม่มี significance test',
        'medium': 'YOLO26m นำ mAP/AP75/Recall; YOLOv8m forward เร็วสุด แต่ YOLO11m นำ pipeline/FPS/VRAM ตัวเลขนี้แสดงว่าการเลือกจาก inference เพียงคอลัมน์เดียวอาจให้คนละ candidate กับการเลือก pipeline',
        'small': 'YOLO26s นำ mAP/AP75/Recall และ pipeline/VRAM ใน tier; YOLOv8s forward เร็วสุด ส่วน YOLO11s/YOLOv8s near-tied ด้าน mAP แต่ไม่ควรเรียกว่า error behavior เหมือนกัน',
        'nano': 'YOLO26n นำ mAP/AP75 และ pipeline/FPS; YOLOv8n นำ Recall/inference และ YOLO11n ใช้ VRAM ต่ำสุดใน tier ผู้ชนะ mask accuracy ยังมี Recall ต่ำสุด จึงต้องระบุ priority ก่อนเลือก'}
    for i, (tier, slug, _) in enumerate(TIERS, 1):
        body += f'\n# {i}. {HUMAN[tier]}\n\n' + result_table([r for r in ROWS if r['tier'] == tier], compact=True)
        body += '\n**Observation:** ' + tier_notes[tier] + f'\n\nภาพและ failure analysis ของ tier: [PRESENTATION_SUMMARY_TH.md](https://github.com/folklazy/YOLO_{slug}_Seg_MOTS20_Benchmark/blob/main/PRESENTATION_SUMMARY_TH.md)\n'
    body += '\n# 6. สรุปรวมทุกขนาด\n\n## ภาพรวม 17 โมเดล\n\n' + family_table()
    body += f'''\n## ผู้ชนะและข้อแลกเปลี่ยนรวม

- Accuracy: YOLO26x-Seg, mAP {v('YOLO26x-Seg','mask_map50_95')}
- Inference: YOLOv8n-Seg, {v('YOLOv8n-Seg','inference_ms_mean')} ms
- Pipeline: YOLO26l-Seg, {v('YOLO26l-Seg','pipeline_ms_mean')} ms / {v('YOLO26l-Seg','fps')} FPS
- VRAM: YOLO26l-Seg, {v('YOLO26l-Seg','peak_allocated_vram_mib')} MiB

## Family scaling และจุดลดขนาด

ทุก family มี mAP/forward ลดลงเมื่อ checkpoint เล็กลง x → l และ e → c เป็นจุดที่ควรพิจารณาเมื่อยอมรับ accuracy loss ได้ โดย YOLO26x → l เปลี่ยน mAP {delta('YOLO26x-Seg','YOLO26l-Seg','mask_map50_95','delta_percentage_points')} pp และ inference {delta('YOLO26x-Seg','YOLO26l-Seg','inference_ms_mean','delta_relative_pct',2)}%

s → n เป็น adjacent mAP drop มากที่สุดของ YOLO26, YOLO11 และ YOLOv8: {delta('YOLO26s-Seg','YOLO26n-Seg','mask_map50_95','delta_percentage_points')}, {delta('YOLO11s-Seg','YOLO11n-Seg','mask_map50_95','delta_percentage_points')} และ {delta('YOLOv8s-Seg','YOLOv8n-Seg','mask_map50_95','delta_percentage_points')} pp การเรียกจุดใดว่า “คุ้ม” ต้องมี accuracy/latency budget ไม่ใช่คะแนนรวม

## Near ties

''' + near_table()
    body += '\nเป็น descriptive screening ที่ |ΔmAP| ≤0.001; ไม่ใช่ equivalence test คู่ข้าม tier ไม่ได้ตรวจภาพใหม่ในขั้นนี้\n\n## สิ่งผิดปกติและการตีความ\n\n**Observation:** pipeline เพิ่มใน 8/13 ขั้น และ VRAM เพิ่มใน 7/13 ขั้น แม้ parameters ลดทั้งหมด YOLO26l เป็น global pipeline/VRAM winner และ YOLO26n เป็น Nano accuracy leader แต่ไม่ใช่ Recall leader\n\n**Interpretation:** workload ของ native-mask postprocessing เป็นบริบทที่ช่วยอ่านแนวโน้ม แต่การสังเคราะห์นี้ไม่พิสูจน์สาเหตุ รุ่นเล็กจึงต้องเทียบค่า pipeline/VRAM จริง ไม่ตัดสินจากชื่อ n/s/m\n\n## ตารางเลือก candidate อย่างเป็นกลาง\n\n' + selection_table(thai=True)
    body += '\nPareto ด้าน inference มี 9 candidates; ด้าน VRAM มี YOLO26x/YOLO26l เท่านั้น วัตถุประสงค์ที่ต่างกันให้ชุดตัวเลือกต่างกัน ไม่มี weighted winner\n\n## ข้อสรุปภาพรวม\n\nผลครบทั้ง 17 โมเดลช่วยเลือก candidate for later CCTV robustness evaluation ตามข้อจำกัดที่กำหนดได้ แต่ไม่ยืนยันความทนต่อสภาพ CCTV หรือ deployment readiness โมเดล pretrained ต่าง capacity/pretraining, frames สัมพันธ์กัน และไม่มี significance test; TP-only quality มีข้อจำกัดจาก matching ขั้นต่อไปต้องกำหนด robustness protocol แยกและวัด pipeline ที่ใช้งานจริง\n\n[Master results](MASTER_RESULTS.md) · [Research insights](RESEARCH_INSIGHTS_TH.md) · [Scaling deltas](metrics/SCALING_DELTAS.csv) · [Source manifest](provenance/SOURCE_MANIFEST.json)\n'
    publish(DOC, body)

    DOC = 'METRIC_GUIDE_TH.md'
    body = '''# คู่มืออ่าน Metric — YOLO MOTS20 Scaling Study

## Accuracy และความครบถ้วน

| Metric | ความหมายในชุดนี้ | ข้อควรระวัง |
|---|---|---|
| Mask mAP50-95 | AP ของ Person masks เฉลี่ยที่ IoU 0.50:0.05:0.95 และ 101 recall points | เป็น segmentation AP แบบ pooled ไม่ใช่ box AP หรือ mean ของ sequence AP |
| AP50 / AP75 | AP ที่ mask IoU 0.50 / 0.75 | AP75 เข้มเรื่อง overlap มากขึ้น แต่ไม่แยก boundary quality ออกจากการหา instance ได้ |
| Precision | TP / (TP + FP) ที่ confidence ≥0.25 | FP คือ unmatched prediction; ไม่จำเป็นต้องเป็นคนที่ไม่มีจริง |
| Recall | TP / (TP + FN) ที่ matching mask IoU ≥0.50 | FN อาจมี mask อยู่แล้วแต่ overlap ไม่ผ่าน ไม่ได้แปลว่าขาด detection เสมอ |
| F1 | 2TP / (2TP + FP + FN) | เป็น fixed operating point ไม่ใช่ค่าแทน mAP |
| TP-only IoU / Dice | คุณภาพ mask เฉพาะคู่ที่ match ผ่าน | ไม่รวมคนที่พลาด และแต่ละโมเดลอาจ match GT คนละ subset |

GT Person เป็น class 2; model Person เป็น class 0 valid GT ได้สิทธิ์ match ก่อน ignore region จาก class 10 unmatched predictions ที่ overlap ignore union / prediction area ≥0.50 ถูก ignore หลัง matching ไม่ถูกนับ FP และไม่ clip Person masks

AP confidence floor 0.001; fixed metrics ใช้ 0.25; NMS box IoU 0.70 ไม่ใช่ mask matching IoU Model max_det=1000 ต่างจาก evaluator AP maxDet=200 ซึ่งผ่าน sensitivity เทียบ reference1000 ด้วย tolerance แบบ strict <0.0001

## เวลาและ throughput

| Metric | หน่วย / ขอบเขต | วิธีอ่าน |
|---|---|---|
| Inference | ms/frame | forward ของโมเดลหลัง preprocessing และ CUDA synchronization |
| Preprocessing | ms/frame | resize/letterbox/normalization และ H2D ตาม adapter ที่ freeze |
| Postprocessing | ms/frame | NMS, native masks, bit packing/transfer และ prediction objects |
| Pipeline | ms/frame | preprocessing + inference + postprocessing |
| FPS | frames/s | 1000 / mean pipeline ms; ไม่ใช่ mean ของ FPS รายเฟรม |
| RLE preparation | ms/frame แยกต่างหาก | ไม่ได้รวมใน pipeline หรือ FPS |
| Model loading | seconds แยกต่างหาก | ไม่นับใน steady-state pipeline |

ใช้ timing frames ที่ freeze 100 เฟรม, warmup10 และ clean measured3รอบต่อโมเดล รวม300 observations; mean/median/P50/P95/std คงค่าจาก source โดย std เป็น population std ไม่เฉลี่ย percentiles จากรายรอบ Largest timing รอบแรกที่ปนเปื้อนถูกเก็บไว้แต่ตัดออกทั้งหมด ใช้เฉพาะ clean_repetition

Pipeline ไม่รวม image decode/disk I/O, GT, evaluator, visualization, RLE preparation และการเขียนผล จึงไม่ใช่ end-to-end CCTV หรือ saved-mask throughput ค่า ultralytics_postprocess_inclusive เป็น diagnostic subset ห้ามนำมาบวก pipeline ซ้ำ

## หน่วยความจำและความซับซ้อน

| Metric | หน่วย | ข้อจำกัด |
|---|---|---|
| Peak allocated VRAM | MiB = 2^20 bytes | peak สูงสุดของ accepted rounds หลัง warmup รวม resident model; ไม่ใช่ memory GPU ทั้งหมดหรือ OOM threshold |
| Peak reserved VRAM | MiB | allocator reservation มีใน source timing; ไม่ใช้แทน allocated ใน Pareto นี้ |
| Parameters | จำนวนเต็ม | loaded checkpoint parameters; fused runtime count มีแยกใน complexity CSV |
| GFLOPs | estimate ที่ imgsz640 | ไม่ใช่ measured operations และไม่รับรอง latency |
| Checkpoint size | MB = 10^6 bytes | ขนาดไฟล์ไม่ใช่ VRAM ระหว่างประมวลผล |

## Scaling delta และ Pareto

Delta = smaller − larger; relative % = delta / larger ×100 ใช้ฐานรุ่นใหญ่เสมอ เช่น fraction ที่ลด0.01 คือ −1 percentage point ไม่ใช่ −1% ของค่ารุ่นใหญ่โดยอัตโนมัติ เครื่องหมายลบของเวลา/VRAMหมายถึงค่าลด ส่วนเครื่องหมายลบของ mAP หมายถึง accuracy ลด

Pareto หลัก maximize mAP / minimize inference; Pareto รอง maximize mAP / minimize allocated VRAM จุดถูก dominate เมื่ออีกโมเดลไม่แย่กว่าทั้งสองด้านและดีกว่าอย่างน้อยหนึ่งด้าน คำนวณจากค่าไม่ปัดเศษ จุดที่เท่ากันยังอยู่ทั้งคู่ ไม่มี weighted score และไม่ใช้ Pareto สองด้านแทนทุกข้อจำกัดของงาน

Near-tie screen ใช้ |ΔmAP| ≤0.001 หรือ0.1 pp เพื่อคัดคู่เชิงพรรณนา ไม่ใช่ significance/equivalence test หรือเกณฑ์เลือกผู้ชนะโดยทศนิยมท้าย

## ขอบเขตข้อสรุป

26,894 instances เป็น annotations รายเฟรม ไม่ใช่คนไม่ซ้ำ 2,862 เฟรมเป็นข้อมูลเดียวกันทุกโมเดล ไม่ใช่17ชุดอิสระ Frames วิดีโอสัมพันธ์กัน รุ่น E/X และ C/L ไม่ได้ capacity เท่ากัน และ pretrained checkpoints ต่าง pretraining ผลนี้ไม่ใช่ tracking หรือหลักฐาน CCTV robustness

[Master results](MASTER_RESULTS.md) · [Canonical schema](DATA_SCHEMA.md) · [Methodology](METHODOLOGY_REFERENCE.md) · [Source provenance](provenance/SOURCE_MANIFEST.json)
'''
    publish(DOC, body)

    DOC = 'README.md'
    body = '''# YOLO Instance Segmentation — Complete MOTS20 Scaling Study

**MASTER COMPLETE — PASS WITH WARNINGS.** This repository synthesizes the five completed tiers into exactly 17 pretrained checkpoints: YOLO26 = 5, YOLO11 = 5, YOLOv8 = 5, YOLOv9 = 2. Every checkpoint uses the same 2,862 MOTS20 frames and 26,894 frame-level Person GT instances. Synthesis performs no inference, tier rerun, training or adaptation. Source warnings remain documented.

## Read the results

| Document | Purpose | Language |
|---|---|---|
| [MASTER_RESULTS.md](MASTER_RESULTS.md) | Technical results, family scaling, Pareto analysis, limitations and conditional candidates | English |
| [RESEARCH_INSIGHTS_TH.md](RESEARCH_INSIGHTS_TH.md) | Detailed cross-tier interpretation | Thai prose |
| [EXECUTIVE_SUMMARY_TH.md](EXECUTIVE_SUMMARY_TH.md) | Short decision summary | Thai prose |
| [MEETING_SUMMARY_TH.md](MEETING_SUMMARY_TH.md) | Neutral reusable meeting/presentation summary | Thai prose |
| [METRIC_GUIDE_TH.md](METRIC_GUIDE_TH.md) | Metric definitions, units and interpretation limits | Thai prose |
| [METHODOLOGY_REFERENCE.md](METHODOLOGY_REFERENCE.md) | Shared frozen method and synthesis checks | English |
| [DATA_SCHEMA.md](DATA_SCHEMA.md) | Canonical source and derived table contract | English |

## Study Navigation

[Largest](https://github.com/folklazy/YOLO_Large_Seg_MOTS20_Benchmark) | [Second-largest](https://github.com/folklazy/YOLO_Second_Largest_Seg_MOTS20_Benchmark) | [Medium](https://github.com/folklazy/YOLO_Medium_Seg_MOTS20_Benchmark) | [Small](https://github.com/folklazy/YOLO_Small_Seg_MOTS20_Benchmark) | [Nano](https://github.com/folklazy/YOLO_Nano_Seg_MOTS20_Benchmark) | [Master Study](https://github.com/folklazy/YOLO_Instance_Segmentation_MOTS20_Scaling_Study)

## Overall measured winners

'''
    body += table(['Objective', 'Model', 'Value'], [[label, model, v(model, key) + unit] for label, model, key, unit in [
        ('Mask mAP50-95', 'YOLO26x-Seg', 'mask_map50_95', ''), ('Inference latency', 'YOLOv8n-Seg', 'inference_ms_mean', ' ms/frame'),
        ('Pipeline latency', 'YOLO26l-Seg', 'pipeline_ms_mean', ' ms/frame'), ('Pipeline FPS', 'YOLO26l-Seg', 'fps', ' frames/s'),
        ('Allocated VRAM', 'YOLO26l-Seg', 'peak_allocated_vram_mib', ' MiB')]])
    body += '''
## Machine-readable artifacts

- [MASTER_17_MODELS.csv](metrics/MASTER_17_MODELS.csv): source-preserving 17-model table.
- [TIER_WINNERS.csv](metrics/TIER_WINNERS.csv): seven objectives for each tier.
- [SCALING_DELTAS.csv](metrics/SCALING_DELTAS.csv): 13 adjacent size pairs × 11 metrics; smaller minus larger.
- [PARETO_FRONTIER.csv](metrics/PARETO_FRONTIER.csv): mAP / inference frontier.
- [PARETO_VRAM.csv](metrics/PARETO_VRAM.csv): mAP / peak allocated VRAM frontier.
- [PER_SEQUENCE_MASTER.csv](metrics/PER_SEQUENCE_MASTER.csv): 68 per-sequence rows, separate from pooled AP.
- [TIMING_MASTER.csv](metrics/TIMING_MASTER.csv) and [MODEL_COMPLEXITY_MASTER.csv](metrics/MODEL_COMPLEXITY_MASTER.csv): source stage measurements and complexity.
- [PLOT_READY_RESULTS.csv](metrics/PLOT_READY_RESULTS.csv) and [NEAR_TIES.csv](metrics/NEAR_TIES.csv): reproducible plotting and descriptive screening inputs.
- [SOURCE_MANIFEST.json](provenance/SOURCE_MANIFEST.json): exact source revisions, run IDs, SHA256, compatibility evidence and editorial archives.
- [MASTER_VALIDATION.json](provenance/MASTER_VALIDATION.json): final scientific/document checks.

## Ten Master plots

'''
    names = ['01_master_map_by_model', '02_family_accuracy_scaling', '03_family_inference_scaling', '04_family_fps_scaling', '05_family_vram_scaling', '06_accuracy_vs_parameters', '07_accuracy_vs_inference', '08_accuracy_vs_vram', '09_pareto_accuracy_latency', '10_adjacent_scaling_delta']
    body += '\n'.join(f'- [{name}.png](plots/{name}.png)' for name in names) + '\n'
    body += '''
## Reproduce and validate

Run from the workspace root with the existing environment; no packages need to change:

```bash
.venv/bin/python -B YOLO_Instance_Segmentation_MOTS20_Scaling_Study/scripts/build_master.py --check-inputs
.venv/bin/python -B YOLO_Instance_Segmentation_MOTS20_Scaling_Study/scripts/build_master.py --validate
.venv/bin/python -B YOLO_Instance_Segmentation_MOTS20_Scaling_Study/scripts/validate_master.py
.venv/bin/python -B -m unittest discover -s YOLO_Instance_Segmentation_MOTS20_Scaling_Study/tests -v
```

The initial build uses scripts/build_master.py, plots use scripts/plot_master.py, and documents use scripts/report_master.py. Different existing outputs are never silently overwritten; archive them before a newly authorized rebuild. Calculations preserve source strings and use Decimal for scaling. The manifest records source revisions and builder hashes. None of these synthesis scripts import a model runtime or open raw predictions.

## Interpretation and retained evidence

Separate measured observations from interpretation. Near ties are descriptive; no significance test is claimed. Pipeline FPS excludes RLE and output writing. Size reduction can lower forward time while raising pipeline latency or allocated VRAM. Pareto fronts are separate two-objective analyses, without a weighted winner. Results identify candidates for later CCTV robustness evaluation, not deployment readiness or final CCTV superiority.

Existing tier quantitative summaries and visual qualitative analyses retain their distinct roles. The [current diagnostic case-selection audit](provenance/TIER_CASE_SELECTION_20261006_V2.md) remains unchanged; this synthesis does not reselect frames or inspect raw predictions. Shared diagnostic frames are not independent samples.

[STUDY_STANDARD.md](STUDY_STANDARD.md), [STUDY_STATE.json](STUDY_STATE.json), [language policy](DOCUMENTATION_LANGUAGE_POLICY.md), [templates/](templates/) and frozen provenance remain available. Historical Stage 0 conversion/validation utilities retain their original assumptions; do not use them to regenerate current results or state. The previous read-only tier documentation validator remains supported after the authorized Master editorial transition.

## Stop condition

All five tiers and the 17-model Master synthesis are complete. No further benchmark or robustness experiment is started. Any next experiment requires a separate user instruction.
'''
    publish(DOC, body)
    write_new(MASTER / 'provenance/DOCUMENT_VALUE_CLAIMS.json', (json.dumps({'status': 'GENERATED', 'source_scope': 'Final synthesis CSVs only', 'claims': CLAIMS}, ensure_ascii=False, indent=2) + '\n').encode())
    print(f'[MASTER] reports: COMPLETE (six generated documents; {len(CLAIMS)} source-linked displayed values)')


if __name__ == '__main__':main()
