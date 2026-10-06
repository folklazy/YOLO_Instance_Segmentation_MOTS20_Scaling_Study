"""Read-only checks for documentation, same-frame evidence, and unchanged metrics."""
from pathlib import Path
import csv
import hashlib
import json
import gzip
import sys
import re
import argparse
import os
import subprocess
import urllib.request
from urllib.parse import quote
from PIL import Image
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
MASTER=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--remote-images',action='store_true',help='Check published GitHub heads, image blobs and HTTP responses')
args=parser.parse_args()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return list(csv.DictReader(p.open()))
def text(p):return re.sub(r'<!--.*?-->','',p.read_text(),flags=re.S)
def table_after(s,section):
    part=s.split(section+'\n',1)[1].split('\n## ',1)[0]
    return [[v.strip() for v in line.strip('|').split('|')] for line in part.splitlines() if line.startswith('|')]
result_fields=['mask_map50_95','ap75','recall','inference_ms_mean','pipeline_ms_mean','fps','peak_allocated_vram_mib']
categories=[('Mask mAP50-95','mask_map50_95',max,''),('AP75','ap75',max,''),('Recall','recall',max,''),
            ('Inference speed','inference_ms_mean',min,' ms'),('Pipeline speed','pipeline_ms_mean',min,' ms'),
            ('VRAM','peak_allocated_vram_mib',min,' MiB')]
def fmt(k,v):return f"{float(v):.{2 if k=='peak_allocated_vram_mib' else 3 if k in ['inference_ms_mean','pipeline_ms_mean','fps'] else 6}f}"

# Small's original numerical layouts remain archived. Current summaries use
# the shared quantitative/qualitative templates without changing measurements.
def completed_small():
    d=ROOT/'YOLO_Small_Seg_MOTS20_Benchmark'
    complete=d/'manifests/SMALL_COMPLETE.json'
    return complete.exists() and json.loads(complete.read_text())['scientific_validation']=='PASS'

def completed_nano():
    d=ROOT/'YOLO_Nano_Seg_MOTS20_Benchmark'
    complete=d/'manifests/NANO_COMPLETE.json'
    if not complete.exists():return False
    status=json.loads(complete.read_text())
    return status['scientific_validation']=='PASS' and status['document_validation']=='PASS'

def historical_measurement(path,expected):
    actual=ROOT/path
    if sha(actual)==expected:return
    assert ((path.startswith('YOLO_Small_Seg_MOTS20_Benchmark/metrics/') and completed_small()) or
            (path.startswith('YOLO_Nano_Seg_MOTS20_Benchmark/metrics/') and completed_nano())),path
    # Only allow an originally empty Small/Nano interface to acquire new results.
    # Existing measured values in completed tiers remain protected byte for byte.
    header=actual.read_bytes().splitlines()[0]
    assert expected in {hashlib.sha256(header+ending).hexdigest() for ending in [b'\n',b'\r\n']},path
    assert rows(actual),path
    transitioned_measurements.add(path)

transitioned_measurements=set()
def validate_small_completed_artifacts(d,data,r,p):
    alignment=json.loads((d/'manifests/VISUAL_ALIGNMENT.json').read_text())
    assert alignment['status']=='PASS' and alignment['active_presentation_profile']=='shared-visual-qualitative'
    assert alignment['inference_rerun'] is False and alignment['measured_values_changed'] is False
    for path,h in alignment['measurement_hashes'].items():assert sha(ROOT/path)==h,path
    for archive in alignment['archives']:assert sha(d/archive['archive'])==archive['sha256']
    quantitative=json.loads((d/'manifests/QUANTITATIVE_ALIGNMENT.json').read_text())
    assert quantitative['status']=='PASS' and quantitative['active_results_profile']=='shared-compact-quantitative'
    assert quantitative['inference_rerun'] is False and quantitative['measured_values_changed'] is False
    assert quantitative['presentation_changed'] is False
    # Document snapshots describe this edit; later authorized editorial changes
    # may update them. Measured CSVs and the frozen protocol stay immutable.
    for path,h in quantitative['unchanged_artifact_sha256'].items():
        if '/metrics/' in path or path.endswith(('EXPERIMENT_PROTOCOL.md','STANDARDIZATION.json')):
            assert sha(ROOT/path)==h,path
    for archive in quantitative['archives']:assert sha(d/archive['archive'])==archive['sha256']
    assert sha(d/'RESULTS_SUMMARY_TH.md')==quantitative['results_sha256']
    assert sha(d/'configs/report_templates/TIER_RESULTS_SUMMARY_TH_TEMPLATE.md')==quantitative['template_sha256']
    assert quantitative['template_sha256']==sha(MASTER/'templates/TIER_RESULTS_SUMMARY_TH_TEMPLATE.md')
    for claim in quantitative['delta_claims']:
        left=next(x for x in data if x['model']==claim['left_model'])
        right=next(x for x in data if x['model']==claim['right_model'])
        expected=f"{float(left[claim['field']])-float(right[claim['field']]):.{claim['decimals']}f}"
        assert expected==claim['display'] and expected in r,claim
    setup=json.loads((d/'manifests/SMALL_SETUP.json').read_text())
    assert setup['report_layout_override']=='Explicit latest user headings replace qualitative-layout template for Small only'
    assert completed_small()
    final=json.loads((d/'manifests/final_integrity.json').read_text())
    assert final['status']=='PASS' and len(final['checks'])==15
    assert all(x['status']=='PASS' for x in final['checks'])
    assert final['frames_per_model']==2862 and final['gt_instances']==26894 and final['timing_clean_rounds']==9
    assert [x['checkpoint'] for x in data]==['yolo26s-seg.pt','yolo11s-seg.pt','yolov8s-seg.pt']
    assert all(x['frames']=='2862' and x['gt_instances']=='26894' for x in data)
    record=json.loads((d/'manifests/DOCUMENT_VALIDATION.json').read_text())
    assert record['status']=='PASS' and record['numeric_table_cells_verified']>0
    for name,h in record['document_sha256'].items():assert sha(d/name)==h,name
    for name,h in record['template_sha256'].items():assert sha(d/'configs/report_templates'/name)==h,name
    for name in ['README','REPORT','RESULTS_SUMMARY_TH','PRESENTATION_SUMMARY_TH']:
        template=text(d/'configs/report_templates'/f'TIER_{name}_TEMPLATE.md')
        doc=text(d/(name+'.md'))
        if name!='PRESENTATION_SUMMARY_TH':
            assert re.findall(r'^## .+$',doc,re.M)==re.findall(r'^## .+$',template,re.M),name
        else:
            assert sha(d/'configs/report_templates/TIER_PRESENTATION_SUMMARY_TH_TEMPLATE.md')==sha(MASTER/'templates/TIER_PRESENTATION_SUMMARY_TH_TEMPLATE.md')
        assert not re.search(r'(?i)mentor|อาจารย์|สรุปสำหรับคุยกับพี่',doc)
    assert r.startswith('# สรุปผล Small YOLO Instance Segmentation\n') and p.startswith('# Small (S) — Visual and Qualitative Analysis\n')
    assert '## Case ' not in r and '![' not in r
    bullets=r.split('## สรุปใน 1 นาที\n')[1].split('\n## ผลลัพธ์หลัก')[0]
    assert 5<=sum(x.startswith('- ') for x in bullets.splitlines())<=8
    review=json.loads((d/'manifests/FINAL_DOCUMENT_REVIEW.json').read_text())
    assert review['status']=='PASS' and review['measurements_changed'] is False and review['inference_run'] is False
    for path,h in review['canonical_metrics_sha256'].items():assert sha(d/path)==h,path
    for claim in review['numeric_claims']:
        source=next(x for x in rows(d/claim['source']) if x['model']==claim['model'] and ('stage' not in claim or x['stage']==claim['stage']))
        value=float(source[claim['field']])*(100 if claim['percent'] else 1)
        display=f"{value:.{claim['decimals']}f}"+('%' if claim['percent'] else '')
        assert display==claim['display'],claim
    # Canonical values are independently compared to their measured sources.
    std=json.loads((d/'manifests/STANDARDIZATION.json').read_text())
    for path,h in std['source_artifact_sha256'].items():assert sha(d/path)==h,path
    run=std['source_run_ids'][0]
    accuracy={x['model']:x for x in rows(d/f'metrics/{run}/per_model.csv')}
    timing={x['model']:x for x in rows(d/f'timing/{run}/clean_repetition/summary.csv')}
    mapping={'mask_map50_95':'map50_95','tp_iou_mean':'matched_iou_mean','tp_dice_mean':'matched_dice_mean'}
    for row in data:
        measured=accuracy[row['checkpoint']];timed=timing[row['checkpoint']]
        for key in ['mask_map50_95','ap50','ap75','precision','recall','f1','tp_iou_mean','tp_dice_mean']:
            assert row[key]==measured[mapping.get(key,key)],key
        for key,source in [('inference_ms_mean','inference_ms_mean'),('pipeline_ms_mean','total_ms_mean'),('fps','fps'),('peak_allocated_vram_mib','peak_gpu_allocated_mib')]:
            assert row[key]==timed[source],key
    # Verify actual table cells rather than treating a PASS manifest as sufficient.
    fields={'Mask mAP50-95':'mask_map50_95','AP50':'ap50','AP75':'ap75','Precision':'precision','Recall':'recall','F1':'f1','TP-only IoU':'tp_iou_mean','TP-only Dice':'tp_dice_mean','Inference ms':'inference_ms_mean','Pipeline ms':'pipeline_ms_mean','FPS':'fps','Peak VRAM MiB':'peak_allocated_vram_mib','Peak VRAM allocated (MiB)':'peak_allocated_vram_mib','Params':'parameters','Parameters':'parameters','GFLOPs':'gflops','Checkpoint MB':'checkpoint_mb'}
    for name in record['document_sha256']:
        header=None
        for line in text(d/name).splitlines():
            if not line.startswith('|'):header=None;continue
            cells=[v.strip() for v in line.strip('|').split('|')]
            if 'Model' in cells and any(x in cells for x in ['Mask mAP50-95','Parameters']):header=cells;continue
            if header is None or all(re.fullmatch('[-:]+',x) for x in cells):continue
            row=next(x for x in data if x['model']==cells[header.index('Model')])
            for title,value in zip(header,cells):
                if title not in fields:continue
                key=fields[title]
                expected=f"{int(row[key]):,}" if key=='parameters' else f"{float(row[key]):.{3 if key in ['inference_ms_mean','pipeline_ms_mean','fps','gflops'] else 2 if key in ['peak_allocated_vram_mib','checkpoint_mb'] else 6}f}"
                assert value==expected,(name,title,value,expected)

def validate_qualitative(d,data,p):
    assert '## Failure Analysis' in p and '## Near-tie visual check' in p
    assert '## เมื่อดูทั้งตัวเลขและภาพร่วมกัน' in p
    assert '## 2. ผลรวมโมเดล' not in p
    expected=[x if not x.startswith('## Case ') else '## Case' for x in re.findall(r'^## .+$',text(ROOT/'YOLO_Medium_Seg_MOTS20_Benchmark/PRESENTATION_SUMMARY_TH.md'),re.M)]
    actual=[x if not x.startswith('## Case ') else '## Case' for x in re.findall(r'^## .+$',p,re.M)]
    assert actual==expected,d.name
    table=table_after(p,'## 1. ภาพรวมผลการทดลอง')
    assert len(table)==len(data)+2
    for actual,source in zip(table[2:],data):
        assert actual==[source['model']]+[fmt(k,source[k]) for k in ['mask_map50_95','ap75','recall']]
    assert len(re.findall(r'^## Case [1-4] — ',p,re.M))==4
    assert p.count('### สิ่งที่เห็นจากภาพ')==p.count('### วิเคราะห์')==p.count('### เชื่อมกับผลเชิงตัวเลข')==4
    assert p.count('### Observation ')==4 and p.count('**Interpretation:**')==4
    ev=json.loads((d/'outputs/visualizations/qualitative/CASE_EVIDENCE.json').read_text())
    assert ev['inference_rerun'] is False and ev['benchmark_values_changed'] is False
    assert len(ev['cases'])==4
    pool={(x['sequence'],x['frame']) for x in json.loads((d/'manifests/visualization_frames.json').read_text())}
    model_order=[x['model'] for x in data]
    for i,case in enumerate(ev['cases'],1):
        assert (case['sequence'],case['frame']) in pool
        assert sha(ROOT/case['image'])==case['image_sha256']
        gt=(ROOT/case['image']).parent.parent/'gt/gt.txt';assert sha(gt)==case['gt_sha256']
        image=d/f'outputs/visualizations/qualitative/case_{i:02d}_comparison.png'
        assert sha(image)==case['comparison_sha256']
        with Image.open(ROOT/case['image']) as im:height=round(im.height*960/im.width)+60
        with Image.open(image) as im:assert im.size==(1920,height*(len(data)+1))
        assert [m['model'] for m in case['models']]==model_order
        for m,model in zip(case['models'],data):
            assert sha(ROOT/m['prediction_path'])==m['prediction_sha256']
            per=next(x for x in rows(d/f"metrics/{ev['run_id']}/per_frame/{model['checkpoint']}.csv")
                     if x['sequence']==case['sequence'] and int(x['frame'])==case['frame'])
            for k in ['tp','fp','fn','ignored_predictions']:assert m[k]==int(per[k])
    if (d/'manifests/CASE_DECISION_AUDIT.json').exists():
        validate_case_decisions(d,data,p,ev)

def validate_case_decisions(d,data,p,ev):
    # Saved masks only; independent recalculation of the selected GT diagnostics.
    sys.path.insert(0,str(ROOT/'YOLO_Large_Seg_MOTS20_Benchmark/src/frozen_pilot'))
    from mots import load_frames
    from metrics import CompactPrediction,fixed_metrics
    out=d/'outputs/visualizations/qualitative'
    audit=json.loads((d/'manifests/CASE_DECISION_AUDIT.json').read_text())
    focus=json.loads((out/'FOCUS_EVIDENCE.json').read_text())
    assert audit['status']=='PASS_WITH_LIMITATIONS' and audit['cases_inspected']==4
    assert audit['pool_frames_rechecked']==12 and audit['selection_replaced'] is False
    assert audit['inference_rerun'] is False and audit['measured_values_changed'] is False
    assert sha(d/'PRESENTATION_SUMMARY_TH.md')==audit['presentation_sha256']
    assert sha(out/'FOCUS_EVIDENCE.json')==audit['focus_evidence_sha256']
    assert focus['inference_rerun'] is False and focus['measured_values_changed'] is False
    assert focus['original_evidence_sha256']==sha(out/'CASE_EVIDENCE.json')
    for archive in audit['archives']:assert sha(d/archive['archive'])==archive['sha256']
    for path,h in audit['canonical_metrics_sha256'].items():assert sha(d/path)==h,path
    assert p.count('### ใช้ประกอบการเลือกอย่างไร')==4
    assert len(focus['cases'])==len(audit['cases'])==4
    for i,(case,fc,review) in enumerate(zip(ev['cases'],focus['cases'],audit['cases']),1):
        assert (fc['sequence'],fc['frame'])==(case['sequence'],case['frame'])==(review['sequence'],review['frame'])
        assert fc['comparison_sha256']==case['comparison_sha256']==review['full_comparison_sha256']
        assert fc['image_sha256']==case['image_sha256'] and fc['gt_sha256']==case['gt_sha256']
        target=out/f'case_{i:02d}_focus.png'
        assert sha(target)==fc['focus_sha256']==review['focus_sha256']
        assert fc['focus_path']==target.name
        with Image.open(target) as im:
            assert list(im.size)==fc['focus_size']==[720*len(fc['regions']),482*(len(data)+1)]
            im.verify()
        frame=load_frames(ROOT/'datasets/MOTS/MOTS/train',case['sequence'],[case['frame']])[0]
        for region in fc['regions']:
            x0,y0,x1,y1=region['xyxy']
            assert 0<=x0<x1<=frame.width and 0<=y0<y1<=frame.height
            assert set(region['target_gt_ids'])<={g.object_id for g in frame.persons}
        part=p.split(f'## Case {i} —',1)[1].split('\n## ',1)[0]
        assert review['role'] in part and review['decision_use'] in part and review['limit'] in part
        assert f'outputs/visualizations/qualitative/case_{i:02d}_comparison.png' in part
        assert f'outputs/visualizations/qualitative/case_{i:02d}_focus.png' in part
        assert [x['model'] for x in fc['models']]==[x['model'] for x in data]
        for original,model in zip(case['models'],fc['models']):
            assert model['prediction_sha256']==original['prediction_sha256']
            saved=json.loads(gzip.decompress((ROOT/original['prediction_path']).read_bytes()))
            preds=[CompactPrediction(x['confidence'],x['class'],x['bbox_xyxy'],{'size':x['rle']['size'],'counts':x['rle']['counts'].encode('ascii')}) for x in saved['predictions']]
            fm=fixed_metrics(frame,preds,.25,.5,.5)
            assert model['full_frame_counts']=={k:fm[k] for k in ['tp','fp','fn','ignored_predictions']}
            for diagnostic in model['target_gt']:
                gt=next(g.decode() for g in frame.persons if g.object_id==diagnostic['gt_id'])
                match=next((x for x in fm['matches'] if x['object_id']==diagnostic['gt_id']),None)
                assert diagnostic['matched']==bool(match)
                assert diagnostic['matched_iou']==(match['iou'] if match else None)
                candidates=[(j,float(np.count_nonzero(gt & x.mask)/np.count_nonzero(gt | x.mask))) for j,x in enumerate(preds) if x.confidence>=.25]
                best=max((v for j,v in candidates),default=0.)
                assert abs(best-diagnostic['best_candidate_iou_at_conf025'])<1e-12
            if i==4 and 'case4_fp_diagnostics' in audit:
                claims=[x for x in audit['case4_fp_diagnostics'] if x['model']==model['model']]
                assert [x['prediction_index'] for x in claims]==fm['fp_indices']
                for claim in claims:
                    mask=preds[claim['prediction_index']].mask
                    expected=max((float(np.count_nonzero(mask & g.decode())/np.count_nonzero(mask)),g.object_id,float(np.count_nonzero(mask & g.decode())/np.count_nonzero(mask | g.decode()))) for g in frame.persons)
                    assert expected==(claim['prediction_ioa'],claim['best_overlap_gt_id'],claim['mask_iou'])
                    assert claim['gt_already_matched']==any(x['object_id']==claim['best_overlap_gt_id'] for x in fm['matches'])
        ids=sorted({x['gt_id'] for m in fc['models'] for x in m['target_gt']})
        for gid in ids[:2] if i==4 else ids:
            values=[]
            for model in fc['models']:
                g=next(x for x in model['target_gt'] if x['gt_id']==gid)
                v=g['matched_iou'] if g['matched'] else g['best_candidate_iou_at_conf025']
                values.append(('TP IoU ' if g['matched'] else 'FN; best IoU ')+f'{v:.3f}')
            assert '| '+str(gid)+' | '+' | '.join(values)+' |' in part


manifests=sorted((MASTER/'provenance').glob('DOCUMENTATION_REDESIGN_*.json'))
assert manifests
record=json.loads(manifests[-1].read_text())
for p,h in record['measurement_hashes_before'].items():historical_measurement(p,h)
for archive in record['archives']:assert sha(ROOT/archive['archive'])==archive['sha256']
decision_review=MASTER/'provenance/CASE_DECISION_REVIEW_20261006.json'
if decision_review.exists():
    decision_record=json.loads(decision_review.read_text())
    for path,h in decision_record['protected_artifacts_sha256'].items():assert sha(ROOT/path)==h,path
    for archive in decision_record['archives']:assert sha(MASTER/archive['archive'])==archive['sha256']
interpretation_manifests=sorted((MASTER/'provenance').glob('TABLE_INTERPRETATION_*.json'))
assert interpretation_manifests, 'Missing table interpretation provenance'
interpretation_record=json.loads(interpretation_manifests[-1].read_text())
for source,h in interpretation_record['sources'].items():historical_measurement(source,h)
for archive in interpretation_record['archives']:assert sha(ROOT/archive['archive'])==archive['sha256']
for claim in interpretation_record['numeric_claims']:
    d=ROOT/claim['repository']
    source=next(x for x in rows(d/'metrics/TIER_RESULTS.csv') if x['model']==claim['model'])
    field=claim['field']
    expected=f"{100*float(source[field]):.2f}%" if field=='recall' else fmt(field,source[field])
    assert claim['display']==expected,claim
    section=text(d/'RESULTS_SUMMARY_TH.md').split('## สรุปผลจากตาราง\n',1)[1].split('\n## ',1)[0]
    assert expected in section,(claim,'expected value absent from section')
summary=[]
for tier in ['Large','Second_Largest','Medium','Small','Nano']:
    d=ROOT/f'YOLO_{tier}_Seg_MOTS20_Benchmark'
    data=rows(d/'metrics/TIER_RESULTS.csv')
    r=text(d/'RESULTS_SUMMARY_TH.md');p=text(d/'PRESENTATION_SUMMARY_TH.md')
    for doc in ['RESULTS_SUMMARY_TH.md','PRESENTATION_SUMMARY_TH.md','REPORT.md']:
        for link in re.findall(r'\]\(([^)]+)\)',text(d/doc)):
            if not re.match(r'https?://|#',link):assert (d/link.split('#')[0]).exists(),(tier,doc,link)
    if tier=='Small' and data and completed_small():
        validate_small_completed_artifacts(d,data,r,p)
    assert re.findall(r'^## .+$',r,re.M)==re.findall(r'^## .+$',text(MASTER/'templates/TIER_RESULTS_SUMMARY_TH_TEMPLATE.md'),re.M),(tier,'quantitative heading order')
    assert '## Failure Analysis' in p and '## Near-tie visual check' in p
    assert '## เมื่อดูทั้งตัวเลขและภาพร่วมกัน' in p
    assert '## Case ' not in r and '![' not in r
    assert '## แต่ละโมเดลเด่นด้านไหน' not in r
    assert '## 2. ผลรวมโมเดล' not in p
    assert not re.search(r'(?i)mentor|อาจารย์',p)
    assert r.count('## สรุปผลจากตาราง')==1
    interpretation=r.split('## สรุปผลจากตาราง\n',1)[1].split('\n## ',1)[0]
    membership=json.loads((MASTER/'STUDY_STATE.json').read_text())['tiers']
    tier_key={'Large':'largest','Second_Largest':'second_largest','Medium':'medium','Small':'small','Nano':'nano'}[tier]
    if data:
        assert membership[tier_key]['completion_status']=='COMPLETE'
        assert re.findall(r'^### (.+)$',interpretation,re.M)==[x['model'] for x in data]
        assert '[canonical CSV](metrics/TIER_RESULTS.csv)' in interpretation
        assert 'TP-only' in interpretation
        valid_values={fmt(k,x[k]) for x in data for k in result_fields+['ap50','precision','f1','tp_iou_mean','tp_dice_mean']}
        valid_values.update(f"{100*float(x['recall']):.2f}%" for x in data)
        quoted_values=set(re.findall(r'\d+\.\d+(?:%)?',interpretation))
        assert quoted_values<=valid_values,(tier,'unsupported quoted numbers',quoted_values-valid_values)
    else:
        suffix='s' if tier=='Small' else 'n'
        assert re.findall(r'^### (.+)$',interpretation,re.M)==[x+suffix+'-Seg' for x in ['YOLO26','YOLO11','YOLOv8']]
        assert 'NOT_RUN' in interpretation
    if not data:
        assert 'NOT_RUN' in r and 'NOT_RUN' in p
        assert '![' not in p and '## Case ' not in p
        summary.append({'tier':tier,'status':'NOT_RUN','cases':0});continue
    bullets=r.split('## สรุปใน 1 นาที',1)[1].split('## ผลลัพธ์หลัก',1)[0]
    assert 5<=sum(x.startswith('- ') for x in bullets.splitlines())<=8
    for section,s,fields in [('## ผลลัพธ์หลัก',r,result_fields),('## 1. ภาพรวมผลการทดลอง',p,result_fields[:3])]:
        table=table_after(s,section)
        assert len(table)==len(data)+2
        for actual,source in zip(table[2:],data):
            expected=[source['model']]+[fmt(k,source[k]) for k in fields]
            assert actual==expected,(tier,section,actual,expected)
    winner=table_after(r,'## Winner ของแต่ละด้าน')
    assert len(winner)==8
    for actual,(title,k,fn,unit) in zip(winner[2:],categories):
        w=fn(data,key=lambda x:float(x[k]));assert actual==[title,w['model'],fmt(k,w[k])+unit]
    validate_qualitative(d,data,p)
    summary.append({'tier':tier,'status':'PASS','cases':4})
for name in ['TIER_RESULTS_SUMMARY_TH_TEMPLATE.md','TIER_PRESENTATION_SUMMARY_TH_TEMPLATE.md']:
    template=text(MASTER/'templates'/name)
    assert '## 2. ผลรวมโมเดล' not in template and '## แต่ละโมเดลเด่นด้านไหน' not in template

# Local existence does not establish publishability. Inspect all actual embeds,
# including existing benchmark insight plots, while excluding HTML comments.
published_images=[]
env=os.environ.copy();env['GIT_TERMINAL_PROMPT']='0';env['GIT_ASKPASS']='/bin/false'
def git(d,*arguments):
    return subprocess.check_output(['git','-C',str(d),*arguments],text=True,env=env,timeout=20).strip()
def http_json(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'MOTS20-documentation-validation'}),timeout=20) as response:
        return json.load(response)
for tier in ['Large','Second_Largest','Medium','Small','Nano']:
    d=ROOT/f'YOLO_{tier}_Seg_MOTS20_Benchmark'
    images=set()
    for doc in d.glob('*.md'):
        s=text(doc)
        links=re.findall(r'!\[[^]]*\]\(([^)]+)\)',s)+re.findall(r'<img[^>]*src=["\x27]([^"\x27]+)',s)
        for link in links:
            if re.match(r'https?://|data:',link):continue
            image=(doc.parent/link.split('#')[0]).resolve()
            assert image.is_relative_to(d.resolve()),(doc,link)
            images.add(str(image.relative_to(d.resolve())))
    if not images:continue
    remote_tree={}
    if args.remote_images:
        head=git(d,'rev-parse','HEAD')
        remote_head=git(d,'ls-remote','origin','refs/heads/main').split()[0]
        assert head==remote_head,(d.name,'remote head differs',head,remote_head)
        tree=http_json(f'https://api.github.com/repos/folklazy/{d.name}/git/trees/{head}?recursive=1')
        assert not tree.get('truncated'),d.name
        remote_tree={entry['path']:entry for entry in tree['tree']}
    for rel in sorted(images):
        image=d/rel
        assert image.exists(),image
        assert git(d,'ls-files','--error-unmatch',rel)==rel,(d.name,'image not tracked',rel)
        with Image.open(image) as im:im.verify()
        assert image.read_bytes()[:8]==b'\x89PNG\r\n\x1a\n',image
        item={'repository':d.name,'image':rel,'git_tracked':True,'png_integrity':'PASS'}
        if args.remote_images:
            local_blob=git(d,'hash-object',rel)
            assert remote_tree[rel]['type']=='blob' and remote_tree[rel]['sha']==local_blob,(d.name,rel,'remote blob mismatch')
            assert git(d,'rev-parse',f'HEAD:{rel}')==local_blob,(d.name,rel,'uncommitted image')
            # Use the branch URL the published relative Markdown resolves to.
            url=f'https://raw.githubusercontent.com/folklazy/{d.name}/main/{quote(rel)}'
            request=urllib.request.Request(url,headers={'Range':'bytes=0-31','User-Agent':'MOTS20-documentation-validation'})
            with urllib.request.urlopen(request,timeout=20) as response:
                assert response.status in {200,206},(url,response.status)
                assert response.headers.get_content_type()=='image/png',(url,response.headers.get('Content-Type'))
                assert response.read(8)==b'\x89PNG\r\n\x1a\n',(url,'not a PNG response')
                item.update(remote_blob_matches=True,http_status=response.status,content_type='image/png')
        published_images.append(item)
print(json.dumps({'documentation_status':'PASS','tiers':summary,'measurement_files_unchanged':len(record['measurement_hashes_before'])-len(set(record['measurement_hashes_before']) & transitioned_measurements),
                  'inference_rerun':False,'existing_measured_values_changed':False,
                  'new_benchmark_interfaces':sorted(transitioned_measurements),
                  'embedded_images_checked':len(published_images),'remote_images_checked':args.remote_images,
                  'images':published_images},ensure_ascii=False,indent=2))
