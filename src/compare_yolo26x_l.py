"""Same-frame YOLO26x/l visual diagnostics from saved masks; no model inference."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import csv,gzip,hashlib,json,sys
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[2];MASTER=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'YOLO_Large_Seg_MOTS20_Benchmark/src/frozen_pilot'))
from mots import load_frames
from metrics import CompactPrediction,fixed_metrics
OUT=MASTER/'outputs/visualizations/yolo26x_vs_yolo26l/v1'
CONFIG=[('x','Large','yolo26x-seg.pt'),('l','Second_Largest','yolo26l-seg.pt')]
CASES=[
 ('MOTS20-02',336,[[2029],[2036]],'x coverage advantage'),
 ('MOTS20-05',503,[[2054],[2128]],'same coverage, different mask quality'),
 ('MOTS20-11',6,[[2015,2016],[2029]],'l counterexample'),
 ('MOTS20-09',1,[[2015,2016],[2021]],'similar-output control'),
 ('MOTS20-09',263,[[2001,2002,2011],[2007]],'common failure'),
]
GREEN=(65,245,110);PINK=(255,80,215);RED=(255,60,50);GREY=(165,165,165)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rr(p):return list(csv.DictReader(p.open()))
def edge(mask):
 inside=mask.copy();inside[1:]&=mask[:-1];inside[:-1]&=mask[1:];inside[:,1:]&=mask[:,:-1];inside[:,:-1]&=mask[:,1:]
 return mask&~inside

def roi(frame,ids):
 selected=[g.decode() for g in frame.persons if g.object_id in ids]
 assert len(selected)==len(ids),(ids,[g.object_id for g in frame.persons])
 yy,xx=np.nonzero(np.logical_or.reduce(selected));bw=xx.max()-xx.min()+1;bh=yy.max()-yy.min()+1
 w=max(bw*1.7,frame.width*.085);h=max(bh*1.45,frame.height*.10);w=max(w,h*1.25);h=max(h,w/1.8);w=min(frame.width,round(w));h=min(frame.height,round(h));cx=(xx.min()+xx.max()+1)/2;cy=(yy.min()+yy.max()+1)/2
 x=max(0,min(round(cx-w/2),frame.width-w));y=max(0,min(round(cy-h/2),frame.height-h));return[x,y,x+w,y+h]

def panel(rgb,frame,box,ids,title,preds=None,fm=None,width=720,height=460):
 pixels=rgb.copy();labels=[]
 if preds is not None:
  for i,p in enumerate(preds):
   if p.confidence<.25:continue
   mask=p.mask;col=GREY if i in fm['ignored_indices'] else RED if i in fm['fp_indices'] else (40,170,255)
   alpha=.07 if i in fm['ignored_indices'] else .28
   pixels[mask]=((1-alpha)*pixels[mask]+alpha*np.array(col)).astype(np.uint8)
   pixels[edge(mask)]=GREY if i in fm['ignored_indices'] else RED if i in fm['fp_indices'] else PINK
 for g in frame.persons:
  mask=g.decode();pixels[edge(mask)]=GREEN
  if g.object_id in ids:
   yy,xx=np.nonzero(mask);labels.append((int(xx.mean()),int(yy.mean()),str(g.object_id)))
 x0,y0,x1,y1=box;crop=Image.fromarray(pixels[y0:y1,x0:x1]);scale=min(width/crop.width,height/crop.height);w,h=round(crop.width*scale),round(crop.height*scale);ox=(width-w)//2;oy=80+(height-h)//2
 out=Image.new('RGB',(width,height+80),'#141a22');out.paste(crop.resize((w,h),Image.Resampling.LANCZOS),(ox,oy));draw=ImageDraw.Draw(out)
 draw.text((14,9),title,fill='white',font=ImageFont.load_default(size=24))
 draw.text((14,44),'Green: GT | Pink: matched prediction | Red: FP | Gray: ignored',fill='#dddddd',font=ImageFont.load_default(size=15))
 for x,y,label in labels:
  if x0<=x<x1 and y0<=y<y1:draw.text((ox+round((x-x0)*scale),oy+round((y-y0)*scale)),label,fill=GREEN,font=ImageFont.load_default(size=22),stroke_width=2,stroke_fill='black')
 return out

def best_overlap(gt,preds,confidence):
 best=(0.0,None,None)
 for i,p in enumerate(preds):
  if p.confidence<confidence:continue
  overlap=np.count_nonzero(gt&p.mask)
  if not overlap:continue
  value=float(overlap/np.count_nonzero(gt|p.mask))
  if value>best[0]:best=(value,i,p.confidence)
 return {'iou':best[0],'prediction_index':best[1],'confidence':best[2]}

def main():
 assert not OUT.exists(),'Refusing overwrite of existing diagnostic output'
 OUT.mkdir(parents=True)
 master_source=MASTER/'provenance/SOURCE_MANIFEST.json';meta={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'scope':'Post-completion pairwise visual diagnostics, not a new benchmark','inference_rerun':False,'benchmark_values_changed':False,'generator_sha256':sha(Path(__file__)),'source_master_manifest_sha256':sha(master_source),'confidence':.25,'matching_iou':.5,'ignore_prediction_ioa':.5,'models':[],'cases':[]}
 sources={};protected={}
 for size,tier,checkpoint in CONFIG:
  d=ROOT/f'YOLO_{tier}_Seg_MOTS20_Benchmark';std=json.loads((d/'manifests/STANDARDIZATION.json').read_text());run=std['source_run_ids'][0]
  overall=next(r for r in rr(d/'metrics/TIER_RESULTS.csv') if r['checkpoint']==checkpoint)
  per_path=d/f'metrics/{run}/per_frame/{checkpoint}.csv';per={(r['sequence'],int(r['frame'])):r for r in rr(per_path)}
  sources[size]=(d,run,per)
  canonical=[d/p for p in std['source_artifact_sha256'] if p.endswith('/per_model.csv')]
  for p in canonical:assert sha(p)==std['source_artifact_sha256'][str(p.relative_to(d))]
  for p in (d/'metrics').glob('*.csv'):protected[str(p.relative_to(ROOT))]=sha(p)
  protected[str(per_path.relative_to(ROOT))]=sha(per_path)
  metadata_path=d/f'predictions/{run}/accuracy/{checkpoint.removesuffix(".pt")}/metadata.json';assert sha(metadata_path)==std['source_artifact_sha256'][str(metadata_path.relative_to(d))]
  meta['models'].append({'size':size,'repository':d.name,'run_id':run,'checkpoint':checkpoint,'canonical':overall,'per_frame_path':str(per_path.relative_to(ROOT)),'per_frame_sha256':sha(per_path),'metadata_sha256':sha(metadata_path)})
 assert list(sources['x'][2])==list(sources['l'][2]) and len(sources['x'][2])==2862
 differences=Counter(int(sources['x'][2][k]['tp'])-int(sources['l'][2][k]['tp']) for k in sources['x'][2])
 meta['frame_count_screen']={'frames':2862,'x_more_tp':sum(v for k,v in differences.items() if k>0),'l_more_tp':sum(v for k,v in differences.items() if k<0),'equal_tp_count':differences[0],'tp_difference_histogram':dict(sorted(differences.items())),'equal_counts_do_not_imply_equal_gt_or_masks':True,'selection_pool':'All 2862 per-frame count rows screened; six candidate frames inspected; five cases selected; no representative sampling or significance claimed','inspected_candidates':[{'sequence':seq,'frame':num} for seq,num in [('MOTS20-02',115)]+[(s,n) for s,n,_,_ in CASES]]}
 for n,(seq,num,targets,reason) in enumerate(CASES,1):
  frame=load_frames(ROOT/'datasets/MOTS/MOTS/train',seq,[num])[0];rgb=np.array(Image.open(frame.image).convert('RGB'))
  allids={g.object_id for g in frame.persons}
  # Similar-output control deliberately focuses existing GT in the same view.
  if n==4:targets=[[g.object_id for g in frame.persons if g.object_id in [2001,2002,2015,2016,2019,2021]][:2]]
  assert targets and all(set(ids)<=allids for ids in targets),(seq,num,targets,allids)
  record={'case':n,'sequence':seq,'frame':num,'selection_reason':reason,'image_path':str(frame.image.relative_to(ROOT)),'image_sha256':sha(frame.image),'gt_sha256':sha(frame.image.parent.parent/'gt/gt.txt'),'regions':[{'gt_ids':ids,'xyxy':roi(frame,ids)} for ids in targets],'models':[]}
  full=[panel(rgb,frame,[0,0,frame.width,frame.height],[],'Original + GT',width=900,height=round(900*frame.height/frame.width))]
  focus_rows=[[panel(rgb,frame,r['xyxy'],r['gt_ids'],f'Original + GT | {seq}/{num:06d}')] for r in record['regions']]
  for size,tier,checkpoint in CONFIG:
   d,run,per=sources[size];p=d/f'predictions/{run}/accuracy/{checkpoint.removesuffix(".pt")}/predictions/{seq}_{num:06d}.json.gz';saved=json.loads(gzip.decompress(p.read_bytes()));assert(saved['sequence'],saved['frame'],saved['width'],saved['height'])==(seq,num,frame.width,frame.height)
   preds=[CompactPrediction(v['confidence'],v['class'],v['bbox_xyxy'],{'size':v['rle']['size'],'counts':v['rle']['counts'].encode('ascii')}) for v in saved['predictions']];assert all(p.class_id==0 for p in preds)
   fm=fixed_metrics(frame,preds,.25,.5,.5)
   for key in ['tp','fp','fn','ignored_predictions']:assert fm[key]==int(per[(seq,num)][key]),(seq,num,size,key)
   model={'model':f'YOLO26{size}-Seg','prediction_path':str(p.relative_to(ROOT)),'prediction_sha256':sha(p),'counts':{k:fm[k] for k in ['tp','fp','fn','ignored_predictions']},'matched_gt_ids':[x['object_id'] for x in fm['matches']],'fn_gt_ids':[frame.persons[i].object_id for i in fm['fn_indices']],'targets':[]}
   title=f'YOLO26{size} | TP {fm["tp"]} / FP {fm["fp"]} / FN {fm["fn"]}'
   full.append(panel(rgb,frame,[0,0,frame.width,frame.height],[],title,preds,fm,width=900,height=round(900*frame.height/frame.width)))
   for row,region in zip(focus_rows,record['regions']):row.append(panel(rgb,frame,region['xyxy'],region['gt_ids'],f'YOLO26{size} | GT '+','.join(map(str,region['gt_ids'])),preds,fm))
   for gid in sorted({g for r in record['regions'] for g in r['gt_ids']}):
    gt=next(g.decode() for g in frame.persons if g.object_id==gid);match=next((v for v in fm['matches'] if v['object_id']==gid),None)
    diagnostic={'gt_id':gid,'gt_area_px':int(gt.sum()),'matched':match is not None,'matched_iou':match['iou'] if match else None,'best_at_conf025':best_overlap(gt,preds,.25)}
    if match is None:diagnostic['best_among_all_saved_predictions']=best_overlap(gt,preds,0)
    model['targets'].append(diagnostic)
   record['models'].append(model)
  for kind,panels,cols in [('comparison',full,3),('focus',[p for row in focus_rows for p in row],3)]:
   canvas=Image.new('RGB',(panels[0].width*cols,panels[0].height*(len(panels)//cols)),'#141a22')
   for i,panel_image in enumerate(panels):canvas.paste(panel_image,((i%cols)*panel_image.width,(i//cols)*panel_image.height))
   path=OUT/f'case_{n:02d}_{kind}.png';canvas.save(path,optimize=True);record[kind+'_path']=str(path.relative_to(MASTER));record[kind+'_sha256']=sha(path)
  meta['cases'].append(record)
 meta['protected_measurement_sha256']=protected
 assert all(sha(ROOT/p)==h for p,h in protected.items())
 (OUT/'EVIDENCE.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
 print('YOLO26x/l: five same-frame cases, ten images; matching agrees with all saved per-frame counts; no inference.')
 for c in meta['cases']:
  print(c['case'],c['sequence'],c['frame'],[(m['model'],m['counts']) for m in c['models']])
  for m in c['models']:
   print(m['model'],'target diagnostics',m['targets'])
if __name__=='__main__':main()
