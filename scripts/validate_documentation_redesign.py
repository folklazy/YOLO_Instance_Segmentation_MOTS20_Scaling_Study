"""Read-only checks for documentation, same-frame evidence, and unchanged metrics."""
from pathlib import Path
import csv
import hashlib
import json
import re
import argparse
import os
import subprocess
import urllib.request
from urllib.parse import quote
from PIL import Image

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

# Small's latest explicit execution request pins numerical report layouts. It
# overrides the earlier qualitative design for this tier, not its protocol.
def completed_small():
    d=ROOT/'YOLO_Small_Seg_MOTS20_Benchmark'
    complete=d/'manifests/SMALL_COMPLETE.json'
    return complete.exists() and json.loads(complete.read_text())['scientific_validation']=='PASS'

def historical_measurement(path,expected):
    actual=ROOT/path
    if sha(actual)==expected:return
    assert path.startswith('YOLO_Small_Seg_MOTS20_Benchmark/metrics/') and completed_small(),path
    # Only allow an originally empty canonical interface to acquire new results.
    # Existing measured values in completed tiers remain protected byte for byte.
    header=actual.read_bytes().splitlines()[0]
    assert expected in {hashlib.sha256(header+ending).hexdigest() for ending in [b'\n',b'\r\n']},path
    assert rows(actual),path
    transitioned_measurements.add(path)

transitioned_measurements=set()
def validate_small_user_profile(d,data,r,p):
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
        assert re.findall(r'^## .+$',doc,re.M)==re.findall(r'^## .+$',template,re.M),name
        assert not re.search(r'(?i)mentor|อาจารย์|สรุปสำหรับคุยกับพี่',doc)
    assert r.startswith('# สรุปผล Small YOLO Instance Segmentation\n') and p.startswith('# Small\n')
    assert '## Case ' not in r and '![' not in r
    bullets=r.split('## สรุปใน 1 นาที\n')[1].split('\n## ผลหลัก')[0]
    assert 5<=sum(x.startswith('- ') for x in bullets.splitlines())<=8
    assert re.findall(r'^### (YOLO.+)$',p,re.M)==[x['model'] for x in data]
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

manifests=sorted((MASTER/'provenance').glob('DOCUMENTATION_REDESIGN_*.json'))
assert manifests
record=json.loads(manifests[-1].read_text())
for p,h in record['measurement_hashes_before'].items():historical_measurement(p,h)
for archive in record['archives']:assert sha(ROOT/archive['archive'])==archive['sha256']
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
        validate_small_user_profile(d,data,r,p)
        summary.append({'tier':tier,'status':'PASS','cases':0,'profile':'explicit-user-numerical-layout'})
        continue
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
        valid_values={fmt(k,x[k]) for x in data for k in result_fields+['ap50','tp_iou_mean','tp_dice_mean']}
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
                  'new_small_benchmark_interfaces':sorted(transitioned_measurements),
                  'embedded_images_checked':len(published_images),'remote_images_checked':args.remote_images,
                  'images':published_images},ensure_ascii=False,indent=2))
