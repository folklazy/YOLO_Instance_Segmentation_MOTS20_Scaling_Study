"""Read-only validation; optional --write-report stores validation evidence only."""
from stage0_standardize import *
import re,sys,math
schema=read(MASTER/'schemas.json');evidence={};numeric_cells=0
for i,n in enumerate(NAMES[:5]):
 p=ROOT/n;prov=read(p/'manifests/STANDARDIZATION.json')
 assert prov['inference_rerun'] is False and prov['metrics_recalculated'] is False
 for kind,cols in schema['schemas'].items():
  with (p/'metrics'/f'{kind}.csv').open() as f:assert next(csv.reader(f))==cols
 with (p/'metrics/PREFLIGHT_MAXDET.csv').open() as f:assert next(csv.reader(f))==schema['PREFLIGHT_MAXDET']
 data=rows(p/'metrics/TIER_RESULTS.csv')
 assert len(data)==(4 if i<2 else 0)
 if i>=2:
  assert all(not rows(p/'metrics'/f'{k}.csv') for k in [*schema['schemas'],'PREFLIGHT_MAXDET'])
  evidence[KEYS[i]]={'status':'PASS','result_status':'NOT_RUN','rows':0};continue
 assert [r['checkpoint'] for r in data]==[s+'-seg.pt' for s in CHECKPOINTS[i]]
 for path,h in prov['source_artifact_sha256'].items():assert sha(p/path)==h,(n,path)
 for entry in prov['historical_documents_archived']:assert sha(p/entry['archive'])==entry['sha256']
 assert sha(p/'EXPERIMENT_PROTOCOL.md')==prov['protocol_hash']
 assert sha(p/'configs/benchmark.yaml')==prov['config_hash']
 for path,h in prov['evaluator_hash'].items():assert sha(p/path)==h
 original={r['model']:r for r in rows(p/'metrics'/RUNS[i]/'per_model.csv')};tim={r['model']:r for r in rows(p/'timing'/RUNS[i]/'clean_repetition/summary.csv')}
 for r in data:
  raw=original[r['checkpoint']];t=tim[r['checkpoint']]
  for col in ['frames','tp','fp','fn','precision','recall','f1','ap50','ap75','ignored_predictions']:assert r[col]==raw[col]
  for col,old in [('gt_instances','gt_persons'),('mask_map50_95','map50_95'),('tp_iou_mean','matched_iou_mean'),('tp_dice_mean','matched_dice_mean')]:assert r[col]==raw[old]
  for col,old in [('inference_ms_mean','inference_ms_mean'),('pipeline_ms_mean','total_ms_mean'),('fps','fps'),('peak_allocated_vram_mib','peak_gpu_allocated_mib')]:assert r[col]==t[old]
  assert int(r['tp'])+int(r['fn'])==int(r['gt_instances'])==26894
  assert math.isclose(float(r['fps']),1000/float(r['pipeline_ms_mean']),rel_tol=1e-12)
 seq=rows(p/'metrics/PER_SEQUENCE_RESULTS.csv');assert len(seq)==16
 original_seq={(r['model'],r['sequence']):r for r in rows(p/'metrics'/RUNS[i]/'per_sequence.csv')}
 for s in seq:
  assert int(s['predictions'])==int(s['tp'])+int(s['fp'])+int(s['ignored_predictions'])
  cp=next(r['checkpoint'] for r in data if r['model']==s['model']);o=original_seq[cp,s['sequence']]
  for k in ['frames','tp','fp','fn','precision','recall','f1','ap50','ap75','ignored_predictions']:assert s[k]==o[k]
  for k,v in {'gt_instances':'gt_persons','predictions':'predictions_at_confidence','mask_map50_95':'map50_95','tp_iou_mean':'matched_iou_mean','tp_dice_mean':'matched_dice_mean'}.items():assert s[k]==o[v]
 for r in data:
  ss=[s for s in seq if s['model']==r['model']]
  for k in ['frames','gt_instances','tp','fp','fn','ignored_predictions']:assert sum(int(s[k]) for s in ss)==int(r[k])
 timing=rows(p/'metrics/TIMING_SUMMARY.csv');assert len(timing)==24
 for r in timing:
  cp=next(x['checkpoint'] for x in data if x['model']==r['model']);prefix={'preprocessing':'preprocess','inference':'inference','postprocessing':'postprocess','pipeline':'total','rle_preparation':'rle_preparation','ultralytics_postprocess_inclusive':'ultralytics_postprocess_inclusive'}[r['stage']]
  for k in ['mean','median','std','p50','p95']:assert r[k+'_ms']==tim[cp][prefix+'_ms_'+k]
  assert (r['repetitions'],r['measured_frames'],r['contamination_status'])==('3','300','CLEAN')
 complexity=rows(p/'metrics/MODEL_COMPLEXITY.csv');assert len(complexity)==4
 checkpoints={r['filename']:r for r in read(p/'reports'/RUNS[i]/'frozen_inputs/manifests/checkpoint_manifest.json')}
 for r in complexity:
  cp=next(x['checkpoint'] for x in data if x['model']==r['model']);ck=checkpoints[cp]
  assert r['loaded_parameters']==str(ck['parameters_loaded']) and r['gflops']==str(ck['gflops_nms_unfused'])
  assert r['checkpoint_mb']==str(ck['bytes']/1e6) and r['model_load_seconds']==tim[cp]['load_seconds_mean']
  m=read(p/'predictions'/RUNS[i]/'accuracy'/cp.removesuffix('.pt')/'metadata.json');assert r['fused_parameters']==str(m['runtime_parameters'])
 pf=rows(p/'metrics/PREFLIGHT_MAXDET.csv');original_pf={(r['model'],r['max_dets']):r for r in rows(p/'metrics'/RUNS[i]/'preflight_maxdet.csv')}
 assert len(pf)==16
 for r in pf:
  o=original_pf[r['checkpoint'],r['max_dets']]
  for k,v in o.items():
   if k!='model':assert r[k]==v
 assert set(r['max_dets'] for r in pf)=={'100','200','300','1000'}
 # Independently map published table headings to exact canonical fields and validate every numeric table cell.
 heading_map=dict((h,k) for k,h in COLS);heading_map.update({'Parameters':'parameters','Peak VRAM allocated MiB':'peak_allocated_vram_mib','Family':'family','Model':'model'})
 for doc in ['README.md','REPORT.md','RESULTS_SUMMARY_TH.md','PRESENTATION_SUMMARY_TH.md']:
  text=(p/doc).read_text();assert 'PASS WITH WARNINGS' in text
  lines=text.splitlines();header=None;order=[]
  for line in lines:
   if not line.startswith('|'):header=None;continue
   cells=[x.strip() for x in line.strip('|').split('|')]
   if 'Model' in cells and any(c in cells for c in ['Mask mAP50-95','Parameters']):header=cells;order=[];continue
   if header and all(re.fullmatch(r'[-:]+',x) for x in cells):continue
   if header:
    m=cells[header.index('Model')];r=next(x for x in data if x['model']==m);order.append(m)
    for h,val in zip(header,cells):
     if h in heading_map:
      k=heading_map[h];assert val==fmt(k,r[k]),(n,doc,h,val);numeric_cells+=k not in ['model','family']
    if len(order)==4:assert order==[x['model'] for x in data]
  if doc=='README.md':assert NAV in text
  # All six-decimal scalar values in canonical prose must exist in overall or sequence evidence.
  valid={f'{float(v):.6f}' for r in data+seq for k,v in r.items() if k in ['mask_map50_95','ap50','ap75','precision','recall','f1','tp_iou_mean','tp_dice_mean'] and v}
  for v in re.findall(r'(?<![\d.])0\.\d{6}(?!\d)',text):assert v in valid,(doc,v)
 plot=read(p/'manifests/PLOT_PROVENANCE.json');assert plot['source_sha256']==sha(p/'metrics/TIER_RESULTS.csv')
 for path,h in plot['outputs'].items():assert sha(p/path)==h
 evidence[KEYS[i]]={'status':'PASS','result_status':'PASS_WITH_WARNINGS','models':4,'source_hashes_verified':len(prov['source_artifact_sha256']),'accuracy_and_timing_values_preserved':True,'preflight_values_preserved':True,'inference_rerun':False,'archived_documents_verified':True}
for doc in ['README.md','REPORT.md','RESULTS_SUMMARY_TH.md','PRESENTATION_SUMMARY_TH.md']:
 headings=lambda p:re.findall(r'^## .*$',p.read_text(),re.M)
 assert headings(ROOT/NAMES[0]/doc)==headings(ROOT/NAMES[1]/doc)
for f in ['STUDY_STANDARD.md','STUDY_STATE.json','DATA_SCHEMA.md','METHODOLOGY_REFERENCE.md']:assert (MASTER/f).is_file()
assert len(list((MASTER/'templates').glob('TIER_*_TEMPLATE.md')))==4
assert not (MASTER/'metrics/MASTER_17_MODELS.csv').exists()
for n in NAMES:
 p=ROOT/n;url=subprocess.check_output(['git','-C',str(p),'remote','get-url','origin'],text=True).strip();assert url=='https://github.com/folklazy/'+n+'.git'
for n in ['datasets','models','.venv']:assert (ROOT/n).is_dir()
assert read(MASTER/'provenance/DATASET_VALIDATION.json')['status']=='PASS'
# Confirm fingerprints and current shared environment compatibility without model execution.
for path,h in read(MASTER/'provenance/BASELINE_REFERENCE.json')['implementation_sha256'].items():assert sha(ROOT/path)==h
result={'timestamp':NOW,'stage0_status':'PASS_WITH_WARNINGS','repositories_audited':6,'format_consistency':'PASS','canonical_schema_consistency':'PASS','verified_markdown_numeric_cells':numeric_cells,'tiers':evidence,'warnings':['Historical NNPACK and pycocotools warnings retained. Largest original timing remains excluded.','Future readiness does not authorize downloads or execution.','Workspace root README update is local; root is not a Git repository.'],'new_inference_executed':False,'new_checkpoints_downloaded':False,'master_synthesis_created':False}
if '--write-report' in sys.argv:dump(MASTER/'provenance/STAGE0_VALIDATION.json',result)
print(f'Stage 0 validation: PASS WITH WARNINGS; 6 repositories; {numeric_cells} Markdown numeric cells verified; schemas and report structure PASS.')
