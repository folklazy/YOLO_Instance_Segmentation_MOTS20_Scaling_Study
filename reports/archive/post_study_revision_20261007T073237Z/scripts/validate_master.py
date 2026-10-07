"""Read-only final Master validation; no model runtime or raw prediction access."""
from collections import Counter
from decimal import Decimal as D
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from PIL import Image
from build_master import MASTER, WORKSPACE, TIERS, validate_inputs, validate_outputs, read_csv, load_json, sha, require
from plot_master import PLOTS
from report_master import formatted

REQUIRED = ['README.md','MASTER_RESULTS.md','RESEARCH_INSIGHTS_TH.md','EXECUTIVE_SUMMARY_TH.md','MEETING_SUMMARY_TH.md','METRIC_GUIDE_TH.md','METHODOLOGY_REFERENCE.md','DATA_SCHEMA.md']
MASTER_HEADINGS = ['1. Study Status','2. Experiment Map','3. Master 17-Model Results','4. Tier Winners','5. Family Scaling','6. Scaling Deltas','7. Pareto Frontier','8. Important Research Findings','9. Near Ties','10. Anomalies','11. Limitations','12. Candidate Models for CCTV Robustness Evaluation']
INSIGHT_HEADINGS = ['สรุปใน 1 นาที','1. ภาพรวมการทดลอง','2. Winner ของแต่ละ Tier','3. ภาพรวมทั้ง 17 Models','4. YOLO26 Scaling','5. YOLO11 Scaling','6. YOLOv8 Scaling','7. YOLOv9 Scaling','8. เทียบ Generation ใน Tier เดียวกัน','9. Accuracy vs Speed','10. Accuracy vs VRAM','11. จุดที่ลดขนาดแล้วคุ้ม','12. จุดที่ Accuracy ตกชัด','13. Near Ties','14. สิ่งผิดปกติ','15. ถ้าต้องเลือกโมเดลตามโจทย์','16. สิ่งที่ผลนี้บอกได้','17. สิ่งที่ผลนี้ยังบอกไม่ได้','18. ขั้นตอนถัดไป']


def validate(pre_state=False):
    inputs = validate_inputs()
    result = validate_outputs(inputs, MASTER)
    manifest = load_json(MASTER/'provenance/SOURCE_MANIFEST.json')
    require(manifest['source_tiers'] == inputs['sources'] and manifest['control_sha256'] == inputs['controls'], 'Source manifest provenance mismatch')
    require(manifest['builder_sha256'] == sha(MASTER/'scripts/build_master.py'), 'Builder hash mismatch')
    for path, h in manifest['artifact_sha256'].items():require(sha(MASTER/path) == h, f'Artifact hash: {path}')
    for revision in manifest['editorial_revisions']:
        require(sha(WORKSPACE/revision['archive']) == revision['before_sha256'], 'Editorial archive hash')
        require(sha(WORKSPACE/revision['path']) == revision['after_sha256'], 'Current editorial hash')
    for path, h in manifest['publication_sha256'].items():require(sha(MASTER/path) == h, f'Publication hash: {path}')
    original = load_json(WORKSPACE/next(x['archive'] for x in manifest['editorial_revisions'] if x['path'].endswith('/STUDY_STATE.json')))
    state = load_json(MASTER/'STUDY_STATE.json')
    for tier, _, _ in TIERS:require(state['tiers'][tier] == original['tiers'][tier], f'Tier state modified: {tier}')
    if not pre_state:
        require(state['stage'] == 'MASTER_SYNTHESIS' and state['tiers']['master']['completion_status'] == 'COMPLETE' and state['tiers']['master']['result_status'] == 'PASS_WITH_WARNINGS', 'Master completion state')
        require(state['stop_gate'] == '17_MODEL_MASTER_STUDY_COMPLETE', 'Master STOP gate')
    master_doc = (MASTER/'MASTER_RESULTS.md').read_text()
    require(re.findall(r'^## (.+)$',master_doc,re.M) == MASTER_HEADINGS, 'MASTER_RESULTS heading order')
    research = (MASTER/'RESEARCH_INSIGHTS_TH.md').read_text()
    require(re.findall(r'^## (.+)$',research,re.M) == INSIGHT_HEADINGS, 'RESEARCH_INSIGHTS heading order')
    meeting = (MASTER/'MEETING_SUMMARY_TH.md').read_text()
    require(meeting.startswith('# ทดลองประสิทธิภาพ YOLO Instance Segmentation บน MOTS20\n## ตั้งแต่รุ่นขนาดใหญ่สุดจนถึง Nano\n'), 'Meeting title')
    require(re.findall(r'^# (\d)\.',meeting,re.M) == ['1','2','3','4','5','6'], 'Meeting tier order')
    require(re.findall(r'^## (.+)$',meeting,re.M)[-1] == 'ข้อสรุปภาพรวม', 'Neutral meeting conclusion')
    require(meeting.count('MOTS20 2,862') == 1 and meeting.count('warmup') == 0, 'Repeated meeting methodology')
    for name in REQUIRED:
        doc = (MASTER/name).read_text()
        require(not re.search(r'สรุปสำหรับคุยกับพี่|สรุปสำหรับพี่|mentor-specific|อาจารย์',doc,re.I), f'Mentor-specific section: {name}')
        if '_TH' in name:require(re.search('[ก-๙]',doc), f'Thai prose missing: {name}')
        else:require(not re.search('[ก-๙]',doc), f'Thai text in English record: {name}')
        for link in re.findall(r'\]\(([^)]+)\)',doc):
            if not re.match(r'https?://|#',link):require((MASTER/link.split('#')[0]).exists(), f'Broken link: {name}: {link}')
    require(len((MASTER/'EXECUTIVE_SUMMARY_TH.md').read_text().splitlines()) <= 75, 'Executive summary exceeds compact layout')
    claims = load_json(MASTER/'provenance/DOCUMENT_VALUE_CLAIMS.json')['claims']
    for claim in claims:
        source = next(x for x in read_csv(MASTER/claim['source']) if all(x[k] == v for k,v in claim['keys'].items()))
        display = formatted(source[claim['field']], claim['field'], claim['decimals'], claim['scale'], claim['signed'])
        require(display == claim['display'] and display in (MASTER/claim['document']).read_text(), f'Unsupported displayed value: {claim}')
    # Validate complete main-table cells, not only substring claims.
    keys = ['mask_map50_95','ap75','recall','inference_ms_mean','pipeline_ms_mean','fps','peak_allocated_vram_mib','parameters','checkpoint_mb']
    section = master_doc.split('## 3. Master 17-Model Results\n',1)[1].split('\n## ',1)[0]
    table = [[v.strip() for v in line.strip('|').split('|')] for line in section.splitlines() if line.startswith('|')]
    require(len(table) == 19, '17-row Master report table')
    human = dict(zip([x[0] for x in TIERS],['Largest (X/E)','Second-largest (L/C)','Medium (M)','Small (S)','Nano (N)']))
    for actual,row in zip(table[2:],inputs['overall']):
        require(actual == [row['model'],human[row['tier']]] + [formatted(row[k],k) for k in keys], 'Report table != master CSV')
    deltas = read_csv(MASTER/'metrics/SCALING_DELTAS.csv')
    require(sum(D(x['delta_absolute']) > 0 for x in deltas if x['metric'] == 'pipeline_ms_mean') == 8, 'Pipeline anomaly count')
    require(sum(D(x['delta_absolute']) > 0 for x in deltas if x['metric'] == 'peak_allocated_vram_mib') == 7, 'VRAM anomaly count')
    for field in ['mask_map50_95','inference_ms_mean','parameters']:
        require(all(D(x['delta_absolute']) < 0 for x in deltas if x['metric'] == field), 'Monotonic scaling claim')
    for family in ['YOLO26','YOLO11','YOLOv8']:
        drops = [x for x in deltas if x['metric']=='mask_map50_95' and x['family']==family]
        require(min(drops,key=lambda x:D(x['delta_absolute']))['smaller_tier']=='nano', 'Largest accuracy-drop claim')
    files = sorted(p.name for p in (MASTER/'plots').glob('*.png'))
    require(files == sorted(PLOTS), 'Exactly ten requested plots required')
    for name in PLOTS:
        with Image.open(MASTER/'plots'/name) as im:
            require(im.format == 'PNG' and min(im.size) >= 1000, f'Plot integrity: {name}')
            im.verify()
    result.update(status='PASS_WITH_WARNINGS', checks='PASS', documents=len(REQUIRED), document_value_claims=len(claims), main_table_numeric_cells=17*len(keys), plots=10,
                  source_provenance='PASS', schema_units='PASS', scaling_math='PASS', pareto_math='PASS', summary_values='PASS', language_and_sections='PASS', tier_states_unchanged=True,
                  raw_predictions_opened=False, no_mentor_specific_section=True, master_state_checked=not pre_state)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--pre-state',action='store_true');args=parser.parse_args()
    result=validate(args.pre_state)
    print('[MASTER] final validation: '+result['status'])
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    try:main()
    except (ValueError,KeyError,OSError,StopIteration) as error:
        print(f'[MASTER] FAIL: {error}',file=sys.stderr);sys.exit(1)
