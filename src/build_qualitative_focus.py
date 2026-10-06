"""Add identical-ROI diagnostic views to existing cases using saved RLE only.

Original comparisons and measured files are read-only. New outputs refuse overwrite.
GT outlines / fixed-metric assignments come from the frozen evaluator. Per-target
best IoU is a diagnostic over predictions at conf >= .25, not a new dataset score.
"""
from pathlib import Path
import argparse,csv,gzip,hashlib,json,sys
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'YOLO_Large_Seg_MOTS20_Benchmark/src/frozen_pilot'))
from mots import load_frames
from metrics import CompactPrediction,fixed_metrics
TIERS=['Large','Second_Largest','Medium','Small','Nano']
PALETTE=[(50,220,90),(40,170,255),(255,210,50),(220,100,255),(255,140,60),(70,240,220)]
TARGETS={
'Large':[[[2002],[2054]],[[2001,2002,2007,2011]],[[2029],[]],[[2021,2022]]],
'Second_Largest':[[[2002],[2054]],[[2001,2002,2007,2011]],[[2020,2023,2026],[2017]],[[2021,2022]]],
'Medium':[[[2002],[2054]],[[2001,2002,2011,2018]],[[2020,2023,2026],[2017]],[[2021,2022]]],
'Small':[[[2026,2027,2028,2029],[2003,2040]],[[2001,2002,2007,2011,2018]],[[2043],[]],[[]]],
'Nano':[[[2029,2043,2048]],[[2001,2002,2007,2011,2018]],[[2026],[2040]],[[],[2019]]],
}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def edge(mask):
 interior=mask.copy();interior[1:]&=mask[:-1];interior[:-1]&=mask[1:];interior[:,1:]&=mask[:,:-1];interior[:,:-1]&=mask[:,1:]
 return mask & ~interior

def crop_box(frame,ids,case,region):
 w,h=frame.width,frame.height
 if not ids:
  if case==4:return [round(w*.08),round(h*.28),round(w*.34),round(h*.75)]
  return [round(w*.80),round(h*.30),w,round(h*.70)]
 masks=[g.decode() for g in frame.persons if g.object_id in ids];assert len(masks)==len(ids)
 yy,xx=np.nonzero(np.logical_or.reduce(masks));cx=(xx.min()+xx.max()+1)/2;cy=(yy.min()+yy.max()+1)/2
 cw=max(xx.max()-xx.min()+1+w*.035,w*.17);ch=max(yy.max()-yy.min()+1+h*.06,h*.20)
 cw=max(cw,ch*1.45);ch=max(ch,cw/1.9);cw=min(cw,w);ch=min(ch,h)
 x=max(0,min(round(cx-cw/2),round(w-cw)));y=max(0,min(round(cy-ch/2),round(h-ch)))
 return [x,y,min(w,round(x+cw)),min(h,round(y+ch))]

def render(rgb,frame,box,ids,preds=None,fm=None,title='',height=400,width=720):
 pixels=rgb.copy();labels=[]
 if preds is None:
  for i,g in enumerate(frame.persons):
   mask=g.decode();color=PALETTE[i%len(PALETTE)];pixels[mask]=(.7*pixels[mask]+.3*np.array(color)).astype(np.uint8);pixels[edge(mask)]=(255,255,255)
 else:
  matches={m['prediction_index']:m for m in fm['matches']}
  for i,p in enumerate(preds):
   if p.confidence<.25:continue
   col=PALETTE[matches[i]['gt_index']%len(PALETTE)] if i in matches else (255,50,50) if i in fm['fp_indices'] else (150,150,150)
   pixels[p.mask]=(.6*pixels[p.mask]+.4*np.array(col)).astype(np.uint8)
   if i in fm['fp_indices']:
    yy,xx=np.nonzero(p.mask)
    if len(xx):labels.append((int(xx.mean()),int(yy.mean()),f'FP P{i}',(255,50,50)))
  for i,g in enumerate(frame.persons):pixels[edge(g.decode())]=(255,170,30) if i in fm['fn_indices'] else (255,255,255)
 for i,g in enumerate(frame.persons):
  if g.object_id not in ids:continue
  yy,xx=np.nonzero(g.decode());color=(255,170,30) if fm and i in fm['fn_indices'] else (255,255,255);labels.append((int(xx.mean()),int(yy.mean()),str(g.object_id),color))
 x0,y0,x1,y1=box;crop=Image.fromarray(pixels[y0:y1,x0:x1]);scale=min(width/crop.width,height/crop.height);cw,ch=round(crop.width*scale),round(crop.height*scale);ox=(width-cw)//2;oy=82+(height-ch)//2
 out=Image.new('RGB',(width,height+82),'#171717');out.paste(crop.resize((cw,ch),Image.Resampling.LANCZOS),(ox,oy));draw=ImageDraw.Draw(out);font=ImageFont.load_default(size=21)
 draw.text((12,8),title,fill='white',font=ImageFont.load_default(size=25));draw.text((12,40),f'ROI xyxy: {box} | white GT / orange FN / red FP',fill='white',font=ImageFont.load_default(size=16))
 for x,y,label,col in labels:
  if x0<=x<x1 and y0<=y<y1:
   px=ox+round((x-x0)*scale);py=oy+round((y-y0)*scale);draw.text((px,py),label,fill=col,font=font,stroke_width=2,stroke_fill='black')
 return out

def build(tier):
 d=ROOT/f'YOLO_{tier}_Seg_MOTS20_Benchmark';out=d/'outputs/visualizations/qualitative';ev=json.loads((out/'CASE_EVIDENCE.json').read_text());data=list(csv.DictReader((d/'metrics/TIER_RESULTS.csv').open()))
 guard={p:sha(p) for p in list((d/'metrics').glob('*.csv'))+[out/'CASE_EVIDENCE.json']+list(out.glob('case_0*_comparison.png'))}
 evidence={'tier':tier,'inference_rerun':False,'measured_values_changed':False,'original_evidence_sha256':sha(out/'CASE_EVIDENCE.json'),'generator_sha256':sha(Path(__file__)),'confidence':.25,'matching_iou':.5,'cases':[]}
 for n,c in enumerate(ev['cases'],1):
  target=out/f'case_{n:02d}_focus.png';assert not target.exists(),target
  frame=load_frames(ROOT/'datasets/MOTS/MOTS/train',c['sequence'],[c['frame']])[0];rgb=np.array(Image.open(frame.image).convert('RGB'));regions=[{'target_gt_ids':ids,'xyxy':crop_box(frame,ids,n,j)} for j,ids in enumerate(TARGETS[tier][n-1])];allids=sorted({i for a in regions for i in a['target_gt_ids']})
  panels=[render(rgb,frame,a['xyxy'],a['target_gt_ids'],title='Original + GT | '+c['sequence']+f" / {c['frame']:06d}") for a in regions];rec={'sequence':c['sequence'],'frame':c['frame'],'regions':regions,'comparison_sha256':c['comparison_sha256'],'image_sha256':c['image_sha256'],'gt_sha256':c['gt_sha256'],'models':[]}
  for row,old in zip(data,c['models']):
   path=ROOT/old['prediction_path'];assert sha(path)==old['prediction_sha256'];saved=json.loads(gzip.decompress(path.read_bytes()));preds=[CompactPrediction(x['confidence'],x['class'],x['bbox_xyxy'],{'size':x['rle']['size'],'counts':x['rle']['counts'].encode('ascii')}) for x in saved['predictions']];fm=fixed_metrics(frame,preds,.25,.5,.5)
   for k in ['tp','fp','fn','ignored_predictions']:assert fm[k]==old[k]
   panels.extend(render(rgb,frame,a['xyxy'],a['target_gt_ids'],preds,fm,row['model']) for a in regions)
   mr={'model':row['model'],'prediction_sha256':old['prediction_sha256'],'full_frame_counts':{k:fm[k] for k in ['tp','fp','fn','ignored_predictions']},'target_gt':[]}
   for gid in allids:
    gt=next(g.decode() for g in frame.persons if g.object_id==gid);match=next((x for x in fm['matches'] if x['object_id']==gid),None);best=0.;bestindex=None
    for i,p in enumerate(preds):
     if p.confidence<.25:continue
     intersection=np.count_nonzero(gt & p.mask)
     if not intersection:continue
     iou=intersection/np.count_nonzero(gt | p.mask)
     if iou>best:best=iou;bestindex=i
    mr['target_gt'].append({'gt_id':gid,'matched':bool(match),'matched_iou':match['iou'] if match else None,'best_candidate_iou_at_conf025':best,'best_candidate_prediction_index':bestindex})
   rec['models'].append(mr)
  canvas=Image.new('RGB',(720*len(regions),482*(len(data)+1)),'#171717')
  for i,panel in enumerate(panels):canvas.paste(panel,(i%len(regions)*720,i//len(regions)*482))
  canvas.save(target,optimize=True);rec.update(focus_path=target.name,focus_sha256=sha(target),focus_size=list(canvas.size));evidence['cases'].append(rec)
 p=out/'FOCUS_EVIDENCE.json';assert not p.exists();p.write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n');assert all(sha(p)==h for p,h in guard.items());print(tier,'four focus views; original comparisons and measured CSVs unchanged.')
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--tier',choices=TIERS+['all'],required=True);args=parser.parse_args()
 for tier in TIERS if args.tier=='all' else [args.tier]:build(tier)
