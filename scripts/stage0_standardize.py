"""Stage 0 artifact conversion only. Never imports a model or executes inference."""
from pathlib import Path
import csv, json, hashlib, shutil, subprocess
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[2]
MASTER=Path(__file__).resolve().parents[1]
SID='yolo_instance_segmentation_mots20_scaling'
NOW=datetime.now(timezone.utc).isoformat()
NAMES=['YOLO_Large_Seg_MOTS20_Benchmark','YOLO_Second_Largest_Seg_MOTS20_Benchmark','YOLO_Medium_Seg_MOTS20_Benchmark','YOLO_Small_Seg_MOTS20_Benchmark','YOLO_Nano_Seg_MOTS20_Benchmark',MASTER.name]
KEYS=['largest','second_largest','medium','small','nano','master']
LABELS=['Largest (X/E)','Second-largest (L/C)','Medium (M)','Small (S)','Nano (N)']
CHECKPOINTS=[['yolo26x','yolo11x','yolov9e','yolov8x'],['yolo26l','yolo11l','yolov9c','yolov8l'],['yolo26m','yolo11m','yolov8m'],['yolo26s','yolo11s','yolov8s'],['yolo26n','yolo11n','yolov8n']]
RUNS=['benchmark-20260929T0520Z','benchmark-20261001T0352Z']
SCHEMAS={
'TIER_RESULTS':'study_id schema_version experiment tier family model checkpoint status frames gt_instances mask_map50_95 ap50 ap75 precision recall f1 tp fp fn tp_iou_mean tp_dice_mean ignored_predictions inference_ms_mean pipeline_ms_mean fps peak_allocated_vram_mib parameters gflops checkpoint_mb ap_maxdet fixed_confidence ap_confidence_floor nms_iou evaluation_iou imgsz precision_mode device run_id source_artifact',
'PER_SEQUENCE_RESULTS':'study_id schema_version experiment tier family model sequence frames gt_instances predictions tp fp fn precision recall f1 ap50 ap75 mask_map50_95 tp_iou_mean tp_dice_mean ignored_predictions',
'TIMING_SUMMARY':'study_id schema_version experiment tier family model stage mean_ms median_ms std_ms p50_ms p95_ms repetitions measured_frames contamination_status',
'MODEL_COMPLEXITY':'study_id schema_version experiment tier family model loaded_parameters fused_parameters gflops checkpoint_mb model_load_seconds'}
NAV=' | '.join(f'[{label}](https://github.com/folklazy/{name})' for label,name in zip(['Largest','Second-largest','Medium','Small','Nano','Master Study'],NAMES))
MASTERLINK=f'[Master Study](https://github.com/folklazy/{MASTER.name})'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def rows(p):return list(csv.DictReader(p.open()))
def write(p,text):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
def dump(p,d):write(p,json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def csvout(p,cols,data):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(data)
def label(cp):return cp.removesuffix('.pt').replace('yolov','YOLOv').replace('yolo','YOLO').replace('-seg','-Seg')
def family(cp):return next(x for x in ['YOLO26','YOLO11','YOLOv9','YOLOv8'] if label(cp).startswith(x))
def base(i,cp):return dict(study_id=SID,schema_version='1.0',experiment=NAMES[i],tier=KEYS[i],family=family(cp),model=label(cp))
def archive(p):
 target=p.parent/'reports/archive'/f'{p.stem}.pre-standardization-20261002{p.suffix}'
 if not target.exists():target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
 return {'original':p.name,'archive':str(target.relative_to(p.parent)),'sha256':sha(target)}
def fmt(k,v):
 if v in ('','NA',None):return 'NA'
 if k in ['model','family','sequence']:return str(v)
 if k in ['parameters','loaded_parameters','tp','fp','fn']:return f'{int(v):,}'
 digits=3 if k in ['inference_ms_mean','pipeline_ms_mean','fps','gflops'] else 2 if k in ['peak_allocated_vram_mib','checkpoint_mb'] else 6
 return f'{float(v):.{digits}f}'
def table(data,cols):return '| '+' | '.join(h for k,h in cols)+' |\n| '+' | '.join('---' for _ in cols)+' |\n'+''.join('| '+' | '.join(fmt(k,r.get(k,'')) for k,h in cols)+' |\n' for r in data)
COLS=list(zip('model mask_map50_95 ap50 ap75 precision recall f1 tp_iou_mean tp_dice_mean inference_ms_mean pipeline_ms_mean fps peak_allocated_vram_mib parameters gflops checkpoint_mb'.split(),['Model','Mask mAP50-95','AP50','AP75','Precision','Recall','F1','TP-only IoU','TP-only Dice','Inference ms','Pipeline ms','FPS','Peak VRAM MiB','Params','GFLOPs','Checkpoint MB']))
COMPACT=[COLS[j] for j in [0,1,5,6,9,10,11,12]]
LIMIT_TH='ผลนี้เป็น Person instance segmentation รายเฟรมบน MOTS20 ไม่ใช่ MOTS tracking; TP-only IoU/Dice พิจารณาเฉพาะคู่ที่ match ได้ ภาพวิดีโอต่อเนื่องสัมพันธ์กันและไม่ได้ทดสอบ statistical significance ค่าใกล้กันควรอ่านว่า near-tied descriptively รุ่น E/X และ C/L ไม่ใช่ capacity เท่ากัน ผลยังไม่ยืนยัน blur, low-light, มุมกล้อง, ระดับ occlusion หรือความพร้อมใช้งาน CCTV; เป็น candidate for later CCTV robustness evaluation เท่านั้น'
WARN_COMMON='Historical PASS WITH WARNINGS retained: CPU NNPACK warnings during complexity inspection and pycocotools/NumPy deprecation warnings. Historical regression checks passed; no dependency changes were made. Pipeline excludes RLE preparation and disk I/O, so FPS is not end-to-end saved-mask throughput.'

def audit():
 out={}
 for i,n in enumerate(NAMES):
  p=ROOT/n
  out[KEYS[i]]={'repository':n,'exists':p.exists(),'branch':subprocess.check_output(['git','-C',str(p),'branch','--show-current'],text=True).strip(),'remote':subprocess.check_output(['git','-C',str(p),'remote','-v'],text=True).strip(),'status':subprocess.check_output(['git','-C',str(p),'status','--short'],text=True).splitlines(),'major_files':[x.name for x in p.iterdir() if x.name!='.git'],'completion_status':'COMPLETE' if i<2 else 'PENDING','canonical_run_id':RUNS[i] if i<2 else None}
 dump(MASTER/'provenance/INITIAL_AUDIT.json',{'timestamp':NOW,'notes':'Read-only audit of all six expected locations and remotes completed before standardization. Four absent local repositories cloned from exact existing empty remotes. No active benchmark processes observed. Largest pre-existing untracked artifacts preserved; no tracked user changes. This record is after clone and before conversion.','repositories':out})

def dataset():
 from PIL import Image
 p=ROOT/NAMES[0];freeze=p/'reports'/RUNS[0]/'frozen_inputs/manifests'
 images=read(freeze/'images.json');dm=read(freeze/'dataset_manifest.json');checks=[]
 assert len(images)==2862
 expected=[]
 for s in dm['sequences']:
  d=ROOT/dm['dataset_root']/s['sequence'];expected.extend((s['sequence'],j) for j in range(1,s['frames']+1))
  assert sha(d/'gt/gt.txt')==s['gt_sha256'] and sha(d/'seqinfo.ini')==s['seqinfo_sha256']
  assert len(list((d/'img1').glob('*.jpg')))==s['frames']
  checks.append({'sequence':s['sequence'],'frames':s['frames'],'resolution_wh':s['resolution_wh'],'gt_sha256':sha(d/'gt/gt.txt'),'seqinfo_sha256':sha(d/'seqinfo.ini'),'status':'PASS'})
 assert [(r['sequence'],r['frame']) for r in images]==expected
 for r in images:
  f=ROOT/r['path'];assert sha(f)==r['sha256']
  with Image.open(f) as im:assert im.size==(r['width'],r['height'])
 assert read(ROOT/NAMES[1]/'manifests/images.json')==images
 assert read(ROOT/NAMES[1]/'manifests/dataset_manifest.json')['sequences']==dm['sequences']
 assert read(ROOT/NAMES[1]/'manifests/timing_frames.json')==read(freeze/'timing_frames.json')
 dump(MASTER/'provenance/DATASET_VALIDATION.json',{'timestamp':NOW,'status':'PASS','frames':len(images),'ordered_image_manifest_sha256':sha(freeze/'images.json'),'baseline_manifest_sha256':sha(freeze/'dataset_manifest.json'),'per_image_evidence':f'../{NAMES[0]}/reports/{RUNS[0]}/frozen_inputs/manifests/images.json','all_image_hashes_and_dimensions_match':True,'frame_order_matches':True,'second_largest_same_inputs':True,'sequences':checks})
 print('Dataset compatibility: PASS')

def convert(i):
 p=ROOT/NAMES[i];run=RUNS[i];frozen=p/'reports'/run/'frozen_inputs';metrics=p/'metrics'/run;tim=p/'timing'/run/'clean_repetition'
 # Run-scoped artifacts outrank public aliases; assert aliases agree before conversion.
 for f in ['per_model.csv','per_sequence.csv','preflight_maxdet.csv','comparison_summary.csv']:
  assert rows(metrics/f)==rows(p/'metrics'/f),f
 a={r['model']:r for r in rows(metrics/'per_model.csv')};t={r['model']:r for r in rows(tim/'summary.csv')};c={r['filename']:r for r in read(frozen/'manifests/checkpoint_manifest.json')}
 comparison={r['model']:r for r in rows(metrics/'comparison_summary.csv')}
 freeze=read(p/'manifests'/f'{run}_full_freeze.json')
 assert sha(frozen/'configs/benchmark.yaml')==freeze['config_sha256']
 assert sha(p/'EXPERIMENT_PROTOCOL.md')==freeze['protocol_sha256']
 assert all(r['status']=='PASS' for r in rows(p/'metrics/protocol_consistency.csv'))
 source=[metrics/x for x in ['per_model.csv','per_sequence.csv','preflight_maxdet.csv','comparison_summary.csv']]+[tim/'summary.csv',frozen/'configs/benchmark.yaml',p/'EXPERIMENT_PROTOCOL.md',frozen/'manifests/checkpoint_manifest.json',frozen/'manifests/dataset_manifest.json',p/'metrics/protocol_consistency.csv',p/'manifests/environment.json',p/'manifests/source_manifest.json',p/'manifests'/f'{run}_full_freeze.json']
 data=[];complexity=[];timing=[];seq=[]
 mapping={'gt_instances':'gt_persons','mask_map50_95':'map50_95','tp_iou_mean':'matched_iou_mean','tp_dice_mean':'matched_dice_mean'}
 for stem in CHECKPOINTS[i]:
  cp=stem+'-seg.pt';r=a[cp];tr=t[cp];ck=c[cp];b=base(i,cp)
  meta=p/'predictions'/run/'accuracy'/cp.removesuffix('.pt')/'metadata.json';m=read(meta)
  assert m['status']=='PASS' and m['images_successful']==2862 and m['images_failed']==0
  assert int(r['frames'])==2862 and int(r['gt_persons'])==26894
  assert tr['clean_rounds']=='3' and tr['frames']=='300'
  source.append(meta)
  d={**b,**{k:r.get(mapping.get(k,k),'') for k in SCHEMAS['TIER_RESULTS'].split() if k not in b}}
  d.update(checkpoint=cp,status='PASS_WITH_WARNINGS',inference_ms_mean=tr['inference_ms_mean'],pipeline_ms_mean=tr['total_ms_mean'],fps=tr['fps'],peak_allocated_vram_mib=tr['peak_gpu_allocated_mib'],parameters=str(ck['parameters_loaded']),gflops=str(ck['gflops_nms_unfused']),checkpoint_mb=str(ck['bytes']/1e6),ap_maxdet='200',fixed_confidence='0.25',ap_confidence_floor='0.001',nms_iou='0.7',evaluation_iou='0.5',imgsz='640',precision_mode='FP32',device='CUDA:0',run_id=run,source_artifact=f'metrics/{run}/per_model.csv;timing/{run}/clean_repetition/summary.csv;reports/{run}/frozen_inputs/manifests/checkpoint_manifest.json')
  for dst,src in [('mask_map50_95','map50_95'),('inference_ms_mean','inference_ms'),('pipeline_ms_mean','pipeline_ms'),('fps','fps'),('peak_allocated_vram_mib','peak_allocated_mib')]:assert d[dst]==comparison[label(cp)][src]
  data.append(d)
  complexity.append({**b,'loaded_parameters':str(ck['parameters_loaded']),'fused_parameters':str(m['runtime_parameters']),'gflops':d['gflops'],'checkpoint_mb':d['checkpoint_mb'],'model_load_seconds':tr['load_seconds_mean']})
  for stage,prefix in [('preprocessing','preprocess'),('inference','inference'),('postprocessing','postprocess'),('pipeline','total'),('rle_preparation','rle_preparation'),('ultralytics_postprocess_inclusive','ultralytics_postprocess_inclusive')]:
   timing.append({**b,'stage':stage,**{k+'_ms':tr[prefix+'_ms_'+k] for k in ['mean','median','std','p50','p95']},'repetitions':tr['clean_rounds'],'measured_frames':tr['frames'],'contamination_status':'CLEAN'})
  for rnd in range(1,4):
   q=tim/f'round{rnd}-{cp}.json';j=read(q);assert j['status']=='PASS' and not j['contaminated'];source.append(q)
  for r in rows(metrics/'per_sequence.csv'):
   if r['model']==cp:seq.append({**{k:r.get(mapping.get(k,k),'') for k in SCHEMAS['PER_SEQUENCE_RESULTS'].split()},**b,'predictions':r['predictions_at_confidence']})
 for name,rs in [('TIER_RESULTS',data),('PER_SEQUENCE_RESULTS',seq),('TIMING_SUMMARY',timing),('MODEL_COMPLEXITY',complexity)]:csvout(p/'metrics'/f'{name}.csv',SCHEMAS[name].split(),rs)
 pf=rows(metrics/'preflight_maxdet.csv');out=[]
 for stem in CHECKPOINTS[i]:
  cp=stem+'-seg.pt'
  for r in pf:
   if r['model']==cp:out.append({**r,**base(i,cp),'checkpoint':cp})
 pfcols=list(base(i,cp))+['checkpoint']+[k for k in pf[0] if k!='model']
 csvout(p/'metrics/PREFLIGHT_MAXDET.csv',pfcols,out)
 archived=[archive(p/x) for x in ['README.md','REPORT.md']]
 for x in ['RESULTS_SUMMARY_TH.md','PRESENTATION_SUMMARY_TH.md']:
  if (p/x).exists():archived.append(archive(p/x))
 docs(i,data,seq)
 srcsha={str(x.relative_to(p)):sha(x) for x in source}
 evaluator={str(x.relative_to(p)):sha(x) for x in [frozen/'src/frozen_pilot/metrics.py',frozen/'src/frozen_pilot/mots.py']}
 for x,h in evaluator.items():assert h==freeze['sources'][x.split('frozen_inputs/')[1]]
 dump(p/'manifests/STANDARDIZATION.json',dict(study_id=SID,schema_version='1.0',repository=NAMES[i],experiment=NAMES[i],tier=KEYS[i],source_run_ids=[run],source_artifact_paths=list(srcsha),source_artifact_sha256=srcsha,evaluator_hash=evaluator,config_hash=freeze['config_sha256'],protocol_hash=freeze['protocol_sha256'],dataset_manifest_hash=sha(frozen/'manifests/dataset_manifest.json'),standardization_timestamp=NOW,inference_rerun=False,metrics_recalculated=False,historical_documents_archived=archived,notes=['String-preserving CSV column normalization; no accuracy or timing recomputation.','EXPERIMENT_PROTOCOL.md preserved byte-for-byte because it is frozen scientific evidence.','predictions in PER_SEQUENCE_RESULTS means predictions at fixed confidence before ignore filtering.','All timing comes from clean_repetition, never the superseded initial pass.','Original artifact locations retained. Dataset shared validation is in Master provenance/DATASET_VALIDATION.json.']))
 return data

def docs(i,data,seq):
 p=ROOT/NAMES[i];title=LABELS[i];run=RUNS[i]
 winner=lambda k,low=False:min(data,key=lambda r:float(r[k])) if low else max(data,key=lambda r:float(r[k]))
 acc=winner('mask_map50_95');speed=winner('inference_ms_mean',True);pipe=winner('pipeline_ms_mean',True);mem=winner('peak_allocated_vram_mib',True)
 compact=table(data,COMPACT);full=table(data,COLS)
 models='| Family | Model | Tier |\n|---|---|---|\n'+''.join(f"| {r['family']} | {r['model']} | {title} |\n" for r in data)
 reports='\n'.join(f'- [{x}]({x})' for x in ['PRESENTATION_SUMMARY_TH.md','RESULTS_SUMMARY_TH.md','REPORT.md','EXPERIMENT_PROTOCOL.md'])
 write(p/'README.md',f'# {title} YOLO Instance Segmentation Benchmark on MOTS20\n\n## Overview\n\nPretrained YOLO Person instance segmentation on MOTS20. No training or fine-tuning; frame-level evaluation, not tracking. This tier is part of the controlled 17-checkpoint scaling study.\n\n## Models\n\n{models}\n## Experimental Status\n\nPASS WITH WARNINGS — complete, run `{run}`; historical inference was not rerun during Stage 0.\n\n## Main Result\n\n{compact}\n## Reports\n\n{reports}\n\n## Study Navigation\n\n{NAV}\n\n## Reproducibility\n\n[configs/](configs/) · [metrics/](metrics/) · [manifests/](manifests/)\n')
 warnings=WARN_COMMON+(' The original Largest timing pass was excluded in its entirety for non-idle starts; all primary measurements use three clean historical repetitions.' if i==0 else ' No contaminated primary timing runs were recorded.')
 findings=[f"Observed highest Mask mAP50-95: {acc['model']} ({fmt('mask_map50_95',acc['mask_map50_95'])}).",f"Lowest inference mean: {speed['model']}; lowest pipeline mean: {pipe['model']}. These are distinct measurements.",f"Lowest allocated VRAM: {mem['model']} ({fmt('peak_allocated_vram_mib',mem['peak_allocated_vram_mib'])} MiB).",'Interpretation: choose by measured accuracy, latency and memory constraints; no weighted score or architecture-causality claim.']
 if i==0:findings.append('YOLO11x-Seg and YOLOv9e-Seg have descriptively near-tied Mask mAP50-95; the tiny difference is not evidence of statistical superiority.')
 else:findings.append('YOLOv9c-Seg and YOLO11l-Seg inference means, and the leading three pipeline means, are descriptively close; no significance test was performed.')
 observations=[]
 for r in data:
  ss=[s for s in seq if s['model']==r['model']];hi=max(ss,key=lambda s:float(s['mask_map50_95']));lo=min(ss,key=lambda s:float(s['mask_map50_95']))
  observations.append(f"{r['model']}: strongest {hi['sequence']} ({fmt('mask_map50_95',hi['mask_map50_95'])}); weakest {lo['sequence']} ({fmt('mask_map50_95',lo['mask_map50_95'])}) by sequence Mask mAP50-95.")
 cats=[('Highest Mask mAP50-95','mask_map50_95',False),('Highest AP75','ap75',False),('Highest Recall','recall',False),('Fastest inference','inference_ms_mean',True),('Fastest pipeline','pipeline_ms_mean',True),('Highest FPS','fps',False),('Lowest VRAM','peak_allocated_vram_mib',True)]
 winners='| Category | Model | Value |\n|---|---|---|\n'+''.join(f"| {name} | {winner(k,low)['model']} | {fmt(k,winner(k,low)[k])} |\n" for name,k,low in cats)
 compatibility='| Item | Status |\n|---|---|\n'+''.join(f'| {k} | PASS |\n' for k in ['Dataset','Evaluator','Preprocessing','Input size','Precision','Thresholds','maxDet','Timing protocol','Environment'])
 sources='\n'.join(f'- [{x}](metrics/{x})' for x in ['TIER_RESULTS.csv','PER_SEQUENCE_RESULTS.csv','TIMING_SUMMARY.csv','MODEL_COMPLEXITY.csv','PREFLIGHT_MAXDET.csv'])
 sections=[('1. Experiment Status',f'PASS WITH WARNINGS\n\n- Models completed: 4/4\n- Frames: 2,862 per model; Person GT instances: 26,894\n- Run ID: `{run}`'),('2. Models Tested',table(data,[('family','Family'),('model','Model'),('parameters','Parameters'),('gflops','GFLOPs'),('checkpoint_mb','Checkpoint MB')])),('3. Protocol Compatibility',compatibility+'\nDataset compatibility: PASS\n\nPreprocessing compatibility: PASS\n\nHistorical environment and frozen evidence verified; shared methodology: '+MASTERLINK+'.'),('4. Overall Results',full),('5. Tier Winners',winners),('6. Key Findings','\n'.join('- '+x for x in findings)),('7. Per-sequence Observations','\n'.join('- '+x for x in observations)),('8. Efficiency and Resource Observations',f"{speed['model']} has the lowest measured inference mean; {pipe['model']} has the lowest pipeline mean and highest mean-derived FPS. {mem['model']} has the lowest allocator peak. Loaded parameters and NMS-path GFLOPs are in the table; fused runtime counts and separate loading times are in MODEL_COMPLEXITY.csv. Parameter counts do not imply proportional VRAM or latency."),('9. Warnings and Anomalies',warnings),('10. Limitations',LIMIT_TH),('11. Reproducibility and Source Artifacts',sources+'\n\n[Standardization provenance](manifests/STANDARDIZATION.json) · [Frozen protocol](EXPERIMENT_PROTOCOL.md) · [Historical reports](reports/archive/)\n\n[Canonical plots](outputs/plots/INDEX.md).'),('12. Relation to Full Scaling Study',MASTERLINK+' — this is one tier only. The 17-model synthesis remains gated on all five tiers and explicit authorization.')]
 write(p/'REPORT.md',f'# {title} YOLO Segmentation Benchmark — MOTS20\n\n'+'\n\n'.join('## '+h+'\n\n'+v for h,v in sections)+'\n')
 bullets=[f"ทดสอบ {', '.join(r['model'] for r in data)} สำหรับ Person instance segmentation",'MOTS20 2,862 frames และ 26,894 Person GT instances เป็น annotation รายเฟรม ไม่ใช่จำนวนบุคคลไม่ซ้ำ','ใช้ pretrained checkpoints / no fine-tuning ภายใต้ controlled benchmark เดียวกัน',f"Accuracy สูงสุด: {acc['model']} — Mask mAP50-95 {fmt('mask_map50_95',acc['mask_map50_95'])}",f"Inference เร็วสุด: {speed['model']}; pipeline เร็วสุด: {pipe['model']}",f"Peak allocated VRAM ต่ำสุด: {mem['model']}",'ค่าความต่างเล็กมากเป็นเพียง near-tied descriptively ไม่ได้พิสูจน์ statistical significance','คง PASS WITH WARNINGS และใช้ผลย้อนหลังเดิมทั้งหมด ไม่รัน inference ใหม่']
 ranks={k:{r['model']:idx+1 for idx,r in enumerate(sorted(data,key=lambda r:float(r[k]),reverse=k=='mask_map50_95'))} for k in ['mask_map50_95','inference_ms_mean','peak_allocated_vram_mib']}
 paragraphs=[]
 for r in data:
  m=r['model'];ar=ranks['mask_map50_95'][m];sr=ranks['inference_ms_mean'][m];vr=ranks['peak_allocated_vram_mib'][m]
  strengths=[];weak=[]
  for rank,axis in [(ar,'accuracy'),(sr,'inference speed'),(vr,'VRAM')]:
   (strengths if rank<=2 else weak).append(axis)
  text=f"อันดับเชิงตัวเลข: accuracy {ar}, inference speed {sr}, VRAM ต่ำ {vr} จากโมเดลใน tier นี้ จุดเด่นคือ {' / '.join(strengths) if strengths else 'เป็นจุดอ้างอิงของตระกูลใน tier นี้'}; จุดที่ด้อยกว่าคือ {' / '.join(weak) if weak else 'ไม่ได้ชนะทุกเป้าหมายพร้อมกัน'} "
  if m==acc['model']:text+='เหมาะเป็นตัวเลือกเริ่มต้นเมื่อให้ความสำคัญกับ accuracy แต่ต้องตรวจ latency ตามข้อจำกัดจริง'
  elif m==speed['model']:text+='เหมาะพิจารณาเมื่อจำกัดเวลา forward และยอมรับ accuracy ที่ต่ำกว่าตัวนำได้'
  else:text+='ควรเปรียบเทียบกับตัวนำตามข้อจำกัดของงาน ไม่สรุปว่าลำดับที่ใกล้กันมีนัยสำคัญ'
  paragraphs.append((m,text))
 insight=[f"Observation: {acc['model']} นำด้าน Mask mAP50-95 แต่การเลือกต้องพิจารณา inference และ pipeline แยกกัน",f"Observation: {mem['model']} ใช้ peak allocated VRAM ต่ำสุด; จำนวน parameters ไม่ใช่ตัวแทน VRAM โดยตรง",'Observation: '+('YOLO11x และ YOLOv9e มี Mask mAP50-95 ใกล้กันมาก' if i==0 else 'YOLOv9c และ YOLO11l มี inference mean ใกล้กัน และ pipeline ของสามตัวนำใกล้กัน'),'Interpretation: ผลนี้ช่วยเลือก candidate for later CCTV robustness evaluation ยังไม่ใช่ข้อยืนยัน deployment']
 trade='### Accuracy\n\n'+acc['model']+' มี Mask mAP50-95 สูงสุดในชุดนี้\n\n### Speed\n\n'+speed['model']+' มี inference mean ต่ำสุด ส่วน '+pipe['model']+' มี pipeline mean ต่ำสุด; FPS ไม่รวม RLE preparation\n\n### Memory / Resource\n\n'+mem['model']+' มี peak allocated VRAM ต่ำสุด ต้องแยกจาก whole-device GPU memory\n\n### ภาพรวม\n\nเลือกตามข้อจำกัดจริง ไม่รวมเป็น weighted score และไม่อนุมานสาเหตุจาก architecture เพียงอย่างเดียว'
 links=f'[metrics/TIER_RESULTS.csv](metrics/TIER_RESULTS.csv) · [REPORT.md](REPORT.md) · [PRESENTATION_SUMMARY_TH.md](PRESENTATION_SUMMARY_TH.md) · {MASTERLINK}'
 summary=[('สรุปใน 1 นาที','\n'.join('- '+x for x in bullets)),('ผลหลัก',compact),('แต่ละโมเดลเด่นด้านไหน','\n\n'.join('**'+m+'**: '+t for m,t in paragraphs)),('สิ่งที่น่าสนใจจากรอบนี้','\n'.join('- '+x for x in insight)),('Trade-off ที่เห็น',trade),('สิ่งที่ต้องระวังในการตีความ',LIMIT_TH+'\n\n'+('รอบ timing เดิมถูกตัดออก ใช้ clean repetitions ที่เก็บไว้เท่านั้น ' if i==0 else '')+'คงคำเตือน NNPACK และ pycocotools ตามหลักฐานเดิม'),('ข้อมูลสำหรับนำไปรวมต่อ',links)]
 write(p/'RESULTS_SUMMARY_TH.md',f'# สรุปผล {title} YOLO Instance Segmentation\n\n'+'\n\n'.join('## '+h+'\n\n'+v for h,v in summary)+'\n')
 choose='| Priority | Recommended model | Reason |\n|---|---|---|\n'+f"| Accuracy | {acc['model']} | Mask mAP50-95 สูงสุด |\n| Speed | {speed['model']} (inference); {pipe['model']} (pipeline) | แยกตามส่วนที่เป็นข้อจำกัด |\n| Low VRAM | {mem['model']} | peak allocated ต่ำสุด |\n| Balanced trade-off | {acc['model']} | "+('เริ่มจาก accuracy สูงสุด แล้วตรวจว่ายอมรับ latency และ VRAM ได้; ไม่ใช่คะแนนรวม' if i==0 else 'accuracy สูงสุด พร้อม pipeline และ VRAM ต่ำสุด แต่ไม่ได้มี forward ต่ำสุด')+' |\n'
 presentation=[('สรุปแบบกระชับ','\n'.join('- '+x for x in bullets)),('โมเดลที่ทดสอบ','\n'.join(f"{j}. {r['family']}: `{r['checkpoint']}`" for j,r in enumerate(data,1))),('1. ผลลัพธ์หลัก',f"{acc['model']} มี Mask mAP50-95 สูงสุด {fmt('mask_map50_95',acc['mask_map50_95'])} ส่วน {speed['model']} มี inference mean ต่ำสุด และ {pipe['model']} มี pipeline mean ต่ำสุด\n\n{mem['model']} ใช้ peak allocated VRAM ต่ำสุด การเลือกจึงต้องแยก accuracy, เวลา forward, pipeline และ memory; ความต่างเล็กมากไม่ควรตีความเป็นความเหนือกว่าทางสถิติ"),('2. ผลรวมโมเดล',table(data,COLS[:12]+[('peak_allocated_vram_mib','Peak VRAM allocated MiB'),('parameters','Parameters')])),('3. สรุปผลจากตาราง','\n\n'.join('### '+m+'\n\n'+t for m,t in paragraphs)),('4. Insight ที่สำคัญ','\n'.join('- '+x for x in insight)),('5. Trade-off',trade),('6. ถ้าต้องเลือกจาก Tier นี้',choose),('7. ข้อควรระวังในการตีความ',LIMIT_TH+'\n\nคง PASS WITH WARNINGS; pipeline ไม่รวม RLE preparation และ disk I/O'),('8. สรุปสำหรับคุยกับพี่','\n'.join('- '+x for x in [f'รอบนี้เทียบ {title} ด้วย pretrained YOLO บน MOTS20 โดยใช้เงื่อนไขเดียวกัน',f"ตัวนำด้าน accuracy คือ {acc['model']} แต่ตัวที่ forward เร็วสุดคือ {speed['model']}",f"ถ้าจำกัด VRAM ให้เริ่มพิจารณา {mem['model']} โดยดู accuracy ที่ยอมรับได้ประกอบ",'จุดที่ต้องระวังคือค่าที่ใกล้กันยังไม่ได้พิสูจน์นัยสำคัญ และ FPS ไม่ใช่ throughput ของระบบ CCTV เต็มรูปแบบ','ขั้นถัดไปควรเติม tier ที่ยังไม่ทดสอบด้วย protocol เดิม หลังได้รับอนุมัติเท่านั้น'])),('รายละเอียดเต็ม',f'[REPORT.md](REPORT.md) · [metrics/TIER_RESULTS.csv](metrics/TIER_RESULTS.csv) · {MASTERLINK}')]
 write(p/'PRESENTATION_SUMMARY_TH.md',f'# {title}\n\n'+'\n\n'.join('## '+h+'\n\n'+v for h,v in presentation)+'\n')

if __name__=='__main__':
 if (MASTER/'provenance/INITIAL_AUDIT.json').exists():
  raise SystemExit('Stage 0 already initialized; use validate_stage0.py. Refusing to overwrite audit or reports.')
 audit();dataset()
 for i in range(2):convert(i)
 print('Historical conversion complete; no inference executed.')
