"""Validate the saved-mask x/l extension without inference or benchmark writes."""
from pathlib import Path
import csv,gzip,json,sys,hashlib,re
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];MASTER=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'YOLO_Large_Seg_MOTS20_Benchmark/src/frozen_pilot'))
from mots import load_frames
from metrics import CompactPrediction,fixed_metrics
OUT=MASTER/'outputs/visualizations/yolo26x_vs_yolo26l/v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rr(p):return list(csv.DictReader(p.open()))
def main():
 e=json.loads((OUT/'EVIDENCE.json').read_text());assert e['inference_rerun'] is False and e['benchmark_values_changed'] is False
 assert len(e['cases'])==5 and sha(MASTER/'src/compare_yolo26x_l.py')==e['generator_sha256']
 for path,h in e['protected_measurement_sha256'].items():assert sha(ROOT/path)==h,path
 screen=e['frame_count_screen'];assert screen['x_more_tp']+screen['l_more_tp']+screen['equal_tp_count']==screen['frames']==2862
 sources={x['size']:{(r['sequence'],int(r['frame'])):r for r in rr(ROOT/x['per_frame_path'])} for x in e['models']}
 deltas=[int(sources['x'][k]['tp'])-int(sources['l'][k]['tp']) for k in sources['x']]
 assert(sum(d>0 for d in deltas),sum(d<0 for d in deltas),sum(d==0 for d in deltas))==(496,224,2142)
 for case in e['cases']:
  frame=load_frames(ROOT/'datasets/MOTS/MOTS/train',case['sequence'],[case['frame']])[0]
  assert sha(frame.image)==case['image_sha256'] and sha(frame.image.parent.parent/'gt/gt.txt')==case['gt_sha256']
  for region in case['regions']:
   x0,y0,x1,y1=region['xyxy'];assert 0<=x0<x1<=frame.width and 0<=y0<y1<=frame.height
   assert set(region['gt_ids'])<={g.object_id for g in frame.persons}
  for model in case['models']:
   p=ROOT/model['prediction_path'];assert sha(p)==model['prediction_sha256']
   saved=json.loads(gzip.decompress(p.read_bytes()));preds=[CompactPrediction(v['confidence'],v['class'],v['bbox_xyxy'],{'size':v['rle']['size'],'counts':v['rle']['counts'].encode('ascii')}) for v in saved['predictions']]
   fm=fixed_metrics(frame,preds,.25,.5,.5);assert model['counts']=={k:fm[k] for k in ['tp','fp','fn','ignored_predictions']}
   source=sources['x' if model['model']=='YOLO26x-Seg' else 'l'][(case['sequence'],case['frame'])]
   for key in model['counts']:assert fm[key]==int(source[key])
   for target in model['targets']:
    gt=next(g.decode() for g in frame.persons if g.object_id==target['gt_id']);assert target['gt_area_px']==int(gt.sum())
    match=next((v for v in fm['matches'] if v['object_id']==target['gt_id']),None)
    assert target['matched']==(match is not None) and target['matched_iou']==(match['iou'] if match else None)
    for key,floor in [('best_at_conf025',.25),('best_among_all_saved_predictions',0)]:
     if key not in target:continue
     diagnostic=target[key]
     expected=max((float(np.count_nonzero(gt&p.mask)/np.count_nonzero(gt|p.mask)) for p in preds if p.confidence>=floor),default=0)
     assert abs(expected-diagnostic['iou'])<1e-12
     if diagnostic['prediction_index'] is not None:assert preds[diagnostic['prediction_index']].confidence==diagnostic['confidence']
  for kind in ['comparison','focus']:
   p=MASTER/case[kind+'_path'];assert sha(p)==case[kind+'_sha256']
   with Image.open(p) as im:assert im.format=='PNG';im.verify()
 assert len(list(OUT.glob('*.png')))==10
 report=(MASTER/'YOLO26X_VS_YOLO26L_VISUAL_TH.md').read_text();assert report.count('## Case ')==5
 for link in re.findall(r'\]\(([^)]+)\)',report):
  if not re.match(r'https?://|#',link):assert(MASTER/link).exists(),link
 assert '## Failure Analysis' in report and 'confidence' in report and 'counterexample' in report
 result={'status':'PASS','cases':5,'images':10,'same_frame_and_roi':'PASS','per_frame_counts':'PASS','per_gt_diagnostics':'PASS','source_hashes':'PASS','model_inference_rerun':False,'canonical_measurements_changed':False,'saved_prediction_artifacts_read':True,'frame_count_screen':screen}
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
