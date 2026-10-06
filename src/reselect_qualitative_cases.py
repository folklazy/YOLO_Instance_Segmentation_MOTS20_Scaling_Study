"""Select diagnostic cases from the frozen twelve-frame pool; no inference.

Reuse unchanged comparisons/focus views when selected. Render only six new
same-frame comparisons into selection_v2, never overwrite earlier artifacts.
Run build first, inspect PNG/GT evidence, then use focus to render targeted ROIs.
"""
from pathlib import Path
import argparse,csv,gzip,hashlib,json,sys
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];MASTER=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(MASTER/'src'));sys.path.insert(0,str(ROOT/'YOLO_Large_Seg_MOTS20_Benchmark/src/frozen_pilot'))
from mots import load_frames
from metrics import CompactPrediction,fixed_metrics
from build_qualitative_comparisons import panel,PALETTE
from build_qualitative_focus import render,crop_box
TIERS=['Large','Second_Largest','Medium','Small','Nano']
SELECTION={
'Large':[('MOTS20-09',525,'tier diagnostic: additional GT and fewer unmatched outputs'),('MOTS20-09',263,'shared anchor: common failure'),('MOTS20-02',600,'counterexample and near-tied pair'),('MOTS20-11',450,'near-tied pair: same valid GT, different extra outputs')],
'Second_Largest':[('MOTS20-05',419,'tier diagnostic: small GT recovered by accuracy leader'),('MOTS20-09',263,'shared anchor: common failure'),('MOTS20-02',1,'counterexample: equal counts, different GT and TP/FP trade-off'),('MOTS20-11',1,'nearby-mAP pair: extra matched GT despite lower aggregate Recall')],
'Medium':[('MOTS20-05',419,'tier diagnostic: small GT and extra output'),('MOTS20-09',263,'shared anchor: common failure'),('MOTS20-02',1,'counterexample: more TP with extra outputs'),('MOTS20-11',450,'control plus error: equal valid GT coverage, different FP')],
'Small':[('MOTS20-02',300,'tier diagnostic: largest TP spread in frozen pool'),('MOTS20-09',263,'shared anchor: common failure'),('MOTS20-02',600,'counterexample and near-tied pair'),('MOTS20-11',900,'near-tied pair: different missed GT and extra outputs')],
'Nano':[('MOTS20-11',1,'tier diagnostic: equal TP for two models, unequal FP and different GT'),('MOTS20-09',263,'shared anchor: common failure'),('MOTS20-02',300,'trade-off: added valid GT and more unmatched outputs'),('MOTS20-09',1,'similar coverage but extra masks on already matched GT')]
}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def load_preds(path):
 s=json.loads(gzip.decompress(path.read_bytes()));return [CompactPrediction(x['confidence'],x['class'],x['bbox_xyxy'],{'size':x['rle']['size'],'counts':x['rle']['counts'].encode('ascii')}) for x in s['predictions']]
def build():
 for tier in TIERS:
  d=ROOT/f'YOLO_{tier}_Seg_MOTS20_Benchmark';oldout=d/'outputs/visualizations/qualitative';out=oldout/'selection_v2';out.mkdir(exist_ok=True);target=out/'CASE_EVIDENCE.json';assert not target.exists()
  old=json.loads((oldout/'CASE_EVIDENCE.json').read_text());of=json.loads((oldout/'FOCUS_EVIDENCE.json').read_text());rows=list(csv.DictReader((d/'metrics/TIER_RESULTS.csv').open()));run=rows[0]['run_id'];pool=json.loads((d/'manifests/visualization_frames.json').read_text());poolkeys={(x['sequence'],x['frame']) for x in pool}
  evidence={k:v for k,v in old.items() if k!='cases'};evidence['selection_version']=2;evidence['cases']=[];focus={k:v for k,v in of.items() if k not in ['cases','original_evidence_sha256']};focus['cases']=[];candidates=[]
  per={r['checkpoint']:{(x['sequence'],int(x['frame'])):x for x in csv.DictReader((d/f"metrics/{run}/per_frame/{r['checkpoint']}.csv").open())} for r in rows}
  for x in pool:candidates.append({'sequence':x['sequence'],'frame':x['frame'],'models':[{'model':r['model'],**{k:int(per[r['checkpoint']][x['sequence'],x['frame']][k]) for k in ['tp','fp','fn']}} for r in rows]})
  for n,(seq,num,reason) in enumerate(SELECTION[tier],1):
   assert (seq,num) in poolkeys;prior=next(((i,c) for i,c in enumerate(old['cases'],1) if (c['sequence'],c['frame'])==(seq,num)),None)
   if prior:
    i,c=prior;rec=dict(c);rec.update(reason=reason,comparison_path=f'outputs/visualizations/qualitative/case_{i:02d}_comparison.png',reused=True);fc=dict(of['cases'][i-1]);fc['focus_path']=f'outputs/visualizations/qualitative/case_{i:02d}_focus.png';focus['cases'].append(fc)
   else:
    frame=load_frames(ROOT/'datasets/MOTS/MOTS/train',seq,[num])[0];rgb=np.array(Image.open(frame.image).convert('RGB'));gt=[(g.decode(),PALETTE[i%len(PALETTE)],str(g.object_id)) for i,g in enumerate(frame.persons)];panels=[panel(rgb,f'{seq} / {num:06d} | Original',[]),panel(rgb,'GT Person IDs (ignore regions excluded)',gt)];rec={'sequence':seq,'frame':num,'reason':reason,'image':str(frame.image.relative_to(ROOT)),'image_sha256':sha(frame.image),'gt_sha256':sha(frame.image.parent.parent/'gt/gt.txt'),'models':[],'reused':False}
    for r in rows:
     path=d/f"predictions/{run}/accuracy/{r['checkpoint'].removesuffix('.pt')}/predictions/{seq}_{num:06d}.json.gz";preds=load_preds(path);fm=fixed_metrics(frame,preds,.25,.5,.5);canonical=per[r['checkpoint']][seq,num]
     for k in ['tp','fp','fn','ignored_predictions']:assert fm[k]==int(canonical[k])
     matches={m['prediction_index']:m for m in fm['matches']};items=[]
     for i,p in enumerate(preds):
      if p.confidence<.25:continue
      color=PALETTE[matches[i]['gt_index']%len(PALETTE)] if i in matches else (255,50,50) if i in fm['fp_indices'] else (150,150,150);label=str(matches[i]['object_id']) if i in matches else f'FP P{i}' if i in fm['fp_indices'] else 'IGN';items.append((p.mask,color,label))
     errors=[(frame.persons[i].decode(),(255,170,30),f'FN {frame.persons[i].object_id}') for i in fm['fn_indices']]+[(preds[i].mask,(255,50,50),f'FP P{i}') for i in fm['fp_indices']]
     panels.extend([panel(rgb,r['model']+' | conf >= 0.25 | gray ignored',items),panel(rgb,f"TP {fm['tp']} / FP {fm['fp']} / FN {fm['fn']} | full-frame",errors)])
     rec['models'].append({'model':r['model'],'prediction_path':str(path.relative_to(ROOT)),'prediction_sha256':sha(path),**{k:fm[k] for k in ['tp','fp','fn','ignored_predictions']},'fn_ids':[frame.persons[i].object_id for i in fm['fn_indices']],'matches':[{'gt_id':m['object_id'],'iou':m['iou']} for m in fm['matches']],'matched_mask_iou':canonical['matched_mask_iou']})
    canvas=Image.new('RGB',(1920,panels[0].height*(len(rows)+1)))
    for i,p in enumerate(panels):canvas.paste(p,((i%2)*960,(i//2)*p.height))
    path=out/f'case_{n:02d}_comparison.png';assert not path.exists();canvas.save(path,optimize=True);rec.update(comparison_path=str(path.relative_to(d)),comparison_sha256=sha(path));focus['cases'].append(None)
   evidence['cases'].append(rec)
  write(target,evidence);focus['original_evidence_sha256']=sha(target);write(out/'FOCUS_REUSE.json',focus);write(out/'CANDIDATE_POOL.json',candidates);print(tier,[(c['sequence'],c['frame'],c['reused']) for c in evidence['cases']],flush=True)
def build_focus():
 for tier in TIERS:
  d=ROOT/f'YOLO_{tier}_Seg_MOTS20_Benchmark';out=d/'outputs/visualizations/qualitative/selection_v2';ev=json.loads((out/'CASE_EVIDENCE.json').read_text());focus=json.loads((out/'FOCUS_REUSE.json').read_text());selection=json.loads((out/'INSPECTED_ROIS.json').read_text());assert not (out/'FOCUS_EVIDENCE.json').exists()
  for n,(c,fc) in enumerate(zip(ev['cases'],focus['cases']),1):
   if fc is not None:continue
   frame=load_frames(ROOT/'datasets/MOTS/MOTS/train',c['sequence'],[c['frame']])[0];rgb=np.array(Image.open(frame.image).convert('RGB'));regions=selection[str(n)];panels=[render(rgb,frame,a['xyxy'],a['target_gt_ids'],title='Original + GT | '+c['sequence']+f" / {c['frame']:06d}") for a in regions];allids=sorted({g for a in regions for g in a['target_gt_ids']});rec={'sequence':c['sequence'],'frame':c['frame'],'regions':regions,'comparison_sha256':c['comparison_sha256'],'image_sha256':c['image_sha256'],'gt_sha256':c['gt_sha256'],'models':[]}
   for model in c['models']:
    preds=load_preds(ROOT/model['prediction_path']);fm=fixed_metrics(frame,preds,.25,.5,.5);panels.extend(render(rgb,frame,a['xyxy'],a['target_gt_ids'],preds,fm,model['model']) for a in regions);mr={'model':model['model'],'prediction_sha256':model['prediction_sha256'],'full_frame_counts':{k:fm[k] for k in ['tp','fp','fn','ignored_predictions']},'target_gt':[]}
    for gid in allids:
     gt=next(g.decode() for g in frame.persons if g.object_id==gid);match=next((x for x in fm['matches'] if x['object_id']==gid),None);values=[(i,float(np.count_nonzero(gt & p.mask)/np.count_nonzero(gt | p.mask))) for i,p in enumerate(preds) if p.confidence>=.25];best=max((v for i,v in values),default=0.);bestindex=next((i for i,v in values if v==best and best>0),None);mr['target_gt'].append({'gt_id':gid,'matched':bool(match),'matched_iou':match['iou'] if match else None,'best_candidate_iou_at_conf025':best,'best_candidate_prediction_index':bestindex})
    rec['models'].append(mr)
   canvas=Image.new('RGB',(720*len(regions),482*(len(c['models'])+1)),'#171717')
   for i,p in enumerate(panels):canvas.paste(p,(i%len(regions)*720,i//len(regions)*482))
   path=out/f'case_{n:02d}_focus.png';assert not path.exists();canvas.save(path,optimize=True);rec.update(focus_path=str(path.relative_to(d)),focus_sha256=sha(path),focus_size=list(canvas.size));focus['cases'][n-1]=rec
  focus['generator_sha256']=sha(Path(__file__));write(out/'FOCUS_EVIDENCE.json',focus);print(tier,'focus COMPLETE',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['build','focus']);a=p.parse_args();build() if a.stage=='build' else build_focus()
