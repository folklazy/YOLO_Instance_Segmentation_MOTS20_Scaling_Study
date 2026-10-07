"""Bounded saved-prediction visual diagnostics; no inference/model imports."""
from pathlib import Path
import sys,csv,gzip,json,hashlib
from datetime import datetime,timezone
import numpy as np
from PIL import Image
from compare_yolo26x_l import panel,roi,best_overlap

WORKSPACE=Path(__file__).resolve().parents[2];MASTER=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORKSPACE/'YOLO_Large_Seg_MOTS20_Benchmark/src/frozen_pilot'))
from mots import load_frames
from metrics import CompactPrediction,fixed_metrics

LADDER=[('Large','yolo26x-seg.pt'),('Second_Largest','yolo26l-seg.pt'),('Medium','yolo26m-seg.pt'),('Small','yolo26s-seg.pt'),('Nano','yolo26n-seg.pt')]
NEAR=[('Small','yolo26s-seg.pt'),('Large','yolo11x-seg.pt'),('Large','yolov9e-seg.pt')]
CONFIGS=[('yolo26_scaling_ladder',LADDER,3,[
    ('MOTS20-09',1,[2019],'similar coverage control'),
    ('MOTS20-02',300,[2026,2040],'scaling degradation with l/s coverage counterexample'),
    ('MOTS20-09',263,[2007,2011,2018],'common failure and Nano matching loss')]),
    ('near_tie_26s_11x_v9e',NEAR,2,[
    ('MOTS20-09',1,[2019],'same coverage control'),
    ('MOTS20-09',525,[2003,2004],'different GT coverage despite near-equal pooled mAP'),
    ('MOTS20-11',450,[2010],'same coverage and differing extra masks'),
    ('MOTS20-09',263,[2007,2011],'counterexample and common failure')])]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with p.open() as f:return list(csv.DictReader(f))


def sources(config):
    out=[]
    for tier,checkpoint in config:
        root=WORKSPACE/f'YOLO_{tier}_Seg_MOTS20_Benchmark'
        std=json.loads((root/'manifests/STANDARDIZATION.json').read_text());run=std['source_run_ids'][0]
        model=next(r['model'] for r in rows(root/'metrics/TIER_RESULTS.csv') if r['checkpoint']==checkpoint)
        per=root/f'metrics/{run}/per_frame/{checkpoint}.csv'
        by_frame={(r['sequence'],int(r['frame'])):r for r in rows(per)}
        metadata=root/f'predictions/{run}/accuracy/{checkpoint.removesuffix(".pt")}/metadata.json'
        assert sha(metadata)==std['source_artifact_sha256'][str(metadata.relative_to(root))]
        assert len(by_frame)==2862
        out.append({'root':root,'run':run,'checkpoint':checkpoint,'model':model,'per':per,'rows':by_frame})
    return out


def build():
    for slug,config,cols,cases in CONFIGS:
        dest=MASTER/f'outputs/visualizations/{slug}/v1';assert not dest.exists(),dest
        ss=sources(config)
        # Refuse partial reconstruction without inference: check every selected artifact first.
        for s in ss:
            for seq,num,ids,reason in cases:
                assert (s['root']/f'predictions/{s["run"]}/accuracy/{s["checkpoint"].removesuffix(".pt")}/predictions/{seq}_{num:06d}.json.gz').is_file()
        dest.mkdir(parents=True)
        evidence={'scope':'Diagnostic qualitative analysis from existing saved predictions only',
                  'timestamp_utc':datetime.now(timezone.utc).isoformat(),'generator_sha256':sha(Path(__file__)),
                  'saved_predictions_only':True,'inference_rerun':False,'benchmark_values_changed':False,
                  'confidence':.25,'matching_iou':.5,'ignore_prediction_ioa':.5,
                  'selection_pool':'Existing diagnostic frame pool and canonical per-frame rows; balanced control, differences and common failures. Not representative or full-dataset extrema.',
                  'sources':[{'model':s['model'],'checkpoint':s['checkpoint'],'per_frame_path':str(s['per'].relative_to(WORKSPACE)),'per_frame_sha256':sha(s['per'])} for s in ss],
                  'cases':[]}
        for number,(seq,num,ids,reason) in enumerate(cases,1):
            frame=load_frames(WORKSPACE/'datasets/MOTS/MOTS/train',seq,[num])[0]
            image=np.array(Image.open(frame.image).convert('RGB'));box=roi(frame,ids)
            record={'case':number,'sequence':seq,'frame':num,'selection_reason':reason,
                    'image_path':str(frame.image.relative_to(WORKSPACE)),'image_sha256':sha(frame.image),
                    'gt_path':str((frame.image.parent.parent/'gt/gt.txt').relative_to(WORKSPACE)),
                    'gt_sha256':sha(frame.image.parent.parent/'gt/gt.txt'),'roi_xyxy':box,'gt_ids':ids,'models':[]}
            full=[panel(image,frame,[0,0,frame.width,frame.height],[],f'Original + GT | {seq}/{num:06d}',width=900,height=round(900*frame.height/frame.width))]
            focus=[panel(image,frame,box,ids,f'Original + GT | {seq}/{num:06d}')]
            for source in ss:
                path=source['root']/f'predictions/{source["run"]}/accuracy/{source["checkpoint"].removesuffix(".pt")}/predictions/{seq}_{num:06d}.json.gz'
                saved=json.loads(gzip.decompress(path.read_bytes()))
                assert (saved['sequence'],saved['frame'],saved['width'],saved['height'])==(seq,num,frame.width,frame.height)
                predictions=[CompactPrediction(p['confidence'],p['class'],p['bbox_xyxy'],{'size':p['rle']['size'],'counts':p['rle']['counts'].encode('ascii')}) for p in saved['predictions']]
                assert all(p.class_id==0 for p in predictions)
                result=fixed_metrics(frame,predictions,.25,.5,.5)
                for key in ['tp','fp','fn','ignored_predictions']:assert result[key]==int(source['rows'][seq,num][key]),(source['model'],seq,num,key)
                model={'model':source['model'],'prediction_path':str(path.relative_to(WORKSPACE)),
                       'prediction_sha256':sha(path),'counts':{k:result[k] for k in ['tp','fp','fn','ignored_predictions']},
                       'matched_gt_ids':[x['object_id'] for x in result['matches']],
                       'fn_gt_ids':[frame.persons[i].object_id for i in result['fn_indices']],
                       'targets':[],'fp_overlap_diagnostics':[]}
                for gid in ids:
                    gt=next(g.decode() for g in frame.persons if g.object_id==gid)
                    match=next((r for r in result['matches'] if r['object_id']==gid),None)
                    target={'gt_id':gid,'matched':match is not None,'matched_iou':match['iou'] if match else None,
                            'best_at_conf025':best_overlap(gt,predictions,.25)}
                    if match is None:target['best_among_all_saved_predictions']=best_overlap(gt,predictions,0)
                    model['targets'].append(target)
                for index in result['fp_indices']:
                    p=predictions[index];best=(0,None)
                    for gt in frame.persons:
                        mask=gt.decode();union=np.count_nonzero(mask|p.mask);iou=np.count_nonzero(mask&p.mask)/union if union else 0
                        if iou>best[0]:best=(iou,gt.object_id)
                    model['fp_overlap_diagnostics'].append({'prediction_index':index,'best_valid_gt_id':best[1],
                                                           'iou':best[0],'gt_already_matched':best[1] in model['matched_gt_ids']})
                record['models'].append(model)
                title=f'{source["model"]} | TP {result["tp"]} FP {result["fp"]} FN {result["fn"]}'
                full.append(panel(image,frame,[0,0,frame.width,frame.height],[],title,predictions,result,width=900,height=round(900*frame.height/frame.width)))
                focus.append(panel(image,frame,box,ids,title,predictions,result))
            for kind,panels in [('comparison',full),('focus',focus)]:
                assert len(panels)%cols==0
                canvas=Image.new('RGB',(panels[0].width*cols,panels[0].height*(len(panels)//cols)),'#141a22')
                for i,p in enumerate(panels):canvas.paste(p,((i%cols)*p.width,(i//cols)*p.height))
                path=dest/f'case_{number:02d}_{kind}.png';canvas.save(path,optimize=True)
                record[kind+'_path']=str(path.relative_to(MASTER));record[kind+'_sha256']=sha(path)
            evidence['cases'].append(record)
        (dest/'EVIDENCE.json').write_text(json.dumps(evidence,indent=2)+'\n')
        lines=['# Post-study diagnostic case selection','',evidence['selection_pool'],'',
               '| Case | Sequence/frame | Reason |','|---|---|---|']
        lines += [f"| {c['case']} | {c['sequence']}/{c['frame']:06d} | {c['selection_reason']} |" for c in evidence['cases']]
        lines += ['','All models use identical original/GT, full-frame extent, ROI coordinates, thresholds and scale. Full frames are retained. Counts agree with existing per-frame CSVs.','', '[Evidence](EVIDENCE.json)']
        (dest/'CASE_SELECTION.md').write_text('\n'.join(lines)+'\n')
        print(slug,len(cases),'cases; no inference; canonical counts matched')


if __name__=='__main__':build()
