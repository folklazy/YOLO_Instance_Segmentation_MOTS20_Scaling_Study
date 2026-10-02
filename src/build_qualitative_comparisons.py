"""Render four same-frame comparisons per completed tier from saved RLE only.

Run with workspace .venv/bin/python. No torch, model loading, inference, or
benchmark writes. Existing qualitative output files are never overwritten.
"""
from pathlib import Path
import argparse
import csv
import gzip
import hashlib
import json
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as coco_mask

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'YOLO_Large_Seg_MOTS20_Benchmark/src/frozen_pilot'
sys.path.insert(0, str(BASE))
from mots import load_frames
from metrics import CompactPrediction, fixed_metrics

TIERS = ['Large', 'Second_Largest', 'Medium']
PALETTE = [(50,220,90),(40,170,255),(255,210,50),(220,100,255),(255,140,60),(70,240,220)]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def panel(rgb, title, items, width=960):
    pixels = rgb.copy()
    for mask, color, label in items:
        pixels[mask] = (.45*pixels[mask]+.55*np.array(color)).astype(np.uint8)
    im = Image.fromarray(pixels)
    height = round(im.height * width / im.width)
    im = im.resize((width,height),Image.Resampling.LANCZOS)
    out = Image.new('RGB',(width,height+60),'#171717')
    out.paste(im,(0,60))
    draw = ImageDraw.Draw(out)
    draw.text((12,8),title,fill='white',font=ImageFont.load_default(size=23))
    for mask, color, label in items:
        yy,xx=np.nonzero(mask)
        if len(xx):
            xy=(int(xx.mean()*width/rgb.shape[1]),60+int(yy.mean()*height/rgb.shape[0]))
            draw.text(xy,label,fill=color,font=ImageFont.load_default(size=22),stroke_width=2,stroke_fill='black')
    return out

def build(tier):
    d=ROOT/f'YOLO_{tier}_Seg_MOTS20_Benchmark'
    rows=list(csv.DictReader((d/'metrics/TIER_RESULTS.csv').open()))
    assert rows and all(int(r['frames'])==2862 and int(r['gt_instances'])==26894 for r in rows)
    run=rows[0]['run_id']
    # Four predefined diagnostic roles from the twelve frozen visualization frames.
    cases=[('MOTS20-05',419,'additional valid instance'),('MOTS20-09',263,'shared misses and false positives'),
           ('MOTS20-02',600 if tier=='Large' else 1,'counterexample / detection trade-off'),
           ('MOTS20-09',1,'similar outputs / near-tie check')]
    out=d/'outputs/visualizations/qualitative'
    out.mkdir(parents=True,exist_ok=True)
    evidence={'tier':tier,'run_id':run,'inference_rerun':False,'benchmark_values_changed':False,
              'confidence':0.25,'matching_iou':0.5,'ignore_prediction_ioa':0.5,'cases':[]}
    for n,(seq,num,reason) in enumerate(cases,1):
        target=out/f'case_{n:02d}_comparison.png'
        assert not target.exists(),target
        frame=load_frames(ROOT/'datasets/MOTS/MOTS/train',seq,[num])[0]
        rgb=np.array(Image.open(frame.image).convert('RGB'))
        gt=[(g.decode(),PALETTE[i%len(PALETTE)],str(g.object_id)) for i,g in enumerate(frame.persons)]
        panels=[panel(rgb,f'{seq} / {num:06d} | Original',[]),panel(rgb,'GT Person IDs (ignore regions excluded)',gt)]
        rec={'sequence':seq,'frame':num,'reason':reason,'image':str(frame.image.relative_to(ROOT)),
             'image_sha256':sha(frame.image),'gt_sha256':sha(frame.image.parent.parent/'gt/gt.txt'),'models':[]}
        for r in rows:
            p=d/f"predictions/{run}/accuracy/{r['checkpoint'].removesuffix('.pt')}/predictions/{seq}_{num:06d}.json.gz"
            with gzip.open(p,'rt') as stream:
                saved=json.load(stream)
            assert (saved['sequence'],saved['frame'],saved['width'],saved['height'])==(seq,num,frame.width,frame.height)
            preds=[CompactPrediction(x['confidence'],x['class'],x['bbox_xyxy'],
                   {'size':x['rle']['size'],'counts':x['rle']['counts'].encode('ascii')}) for x in saved['predictions']]
            fm=fixed_metrics(frame,preds,.25,.5,.5)
            canonical=next(x for x in csv.DictReader((d/f"metrics/{run}/per_frame/{r['checkpoint']}.csv").open())
                           if x['sequence']==seq and int(x['frame'])==num)
            for key in ['tp','fp','fn','ignored_predictions']:
                assert int(canonical[key])==fm[key],(r['model'],key)
            matches={x['prediction_index']:x for x in fm['matches']}
            items=[]
            for i,x in enumerate(preds):
                if x.confidence<.25: continue
                if i in matches:
                    m=matches[i]; color=PALETTE[m['gt_index']%len(PALETTE)]; label=str(m['object_id'])
                elif i in fm['fp_indices']:color=(255,50,50);label=f'FP P{i}'
                else:color=(150,150,150);label='IGN'
                items.append((x.mask,color,label))
            errors=[(frame.persons[i].decode(),(255,170,30),f'FN {frame.persons[i].object_id}') for i in fm['fn_indices']]
            errors += [(preds[i].mask,(255,50,50),f'FP P{i}') for i in fm['fp_indices']]
            panels += [panel(rgb,f"{r['model']} | conf >= 0.25 | gray = ignored",items),
                       panel(rgb,f"TP {fm['tp']} / FP {fm['fp']} / FN {fm['fn']} | orange FN, red FP",errors)]
            rec['models'].append({'model':r['model'],'prediction_path':str(p.relative_to(ROOT)),
                'prediction_sha256':sha(p),'tp':fm['tp'],'fp':fm['fp'],'fn':fm['fn'],
                'ignored_predictions':fm['ignored_predictions'],'fn_ids':[frame.persons[i].object_id for i in fm['fn_indices']],
                'matches':[{'gt_id':m['object_id'],'iou':m['iou']} for m in fm['matches']],
                'matched_mask_iou':canonical['matched_mask_iou']})
        canvas=Image.new('RGB',(panels[0].width*2,panels[0].height*(len(panels)//2)))
        for i,im in enumerate(panels):canvas.paste(im,((i%2)*im.width,(i//2)*im.height))
        canvas.save(target,optimize=True)
        rec['comparison_sha256']=sha(target)
        evidence['cases'].append(rec)
    path=out/'CASE_EVIDENCE.json'
    assert not path.exists(),path
    path.write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    print(tier,[(x['sequence'],x['frame'],[(m['model'],m['tp'],m['fp'],m['fn'],m['fn_ids']) for m in x['models']]) for x in evidence['cases']])

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tier',choices=TIERS+['all'],required=True)
    args=parser.parse_args()
    for tier in TIERS if args.tier=='all' else [args.tier]:build(tier)
