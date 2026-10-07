"""Read-only independent validation of post-study math, evidence and editorial archives."""
from pathlib import Path
from decimal import Decimal as D, localcontext
from collections import Counter
import argparse,csv,json,hashlib,re,subprocess,sys
from PIL import Image

MASTER=Path(__file__).resolve().parents[1];ROOT=MASTER.parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def rows(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))
def require(condition,message):
    if not condition:raise ValueError(message)
def numeric(actual,expected,tol='1e-40'):
    require(abs(D(str(actual))-D(str(expected)))<=D(tol),f'Numeric mismatch: {actual} != {expected}')
def table(doc,heading):
    block=doc.split(heading+'\n',1)[1].split('\n## ',1)[0]
    return [[v.strip() for v in line.strip('|').split('|')] for line in block.splitlines() if line.startswith('|')]
def fmt(k,x):return f'{float(x):.{2 if k=="peak_allocated_vram_mib" else 3 if k in ["inference_ms_mean","pipeline_ms_mean","fps"] else 6}f}'


def validate_math():
    overall=rows(MASTER/'metrics/MASTER_17_MODELS.csv');models={r['model']:r for r in overall}
    require(len(overall)==len(models)==17,'17 unique models')
    require(Counter(r['family'] for r in overall)==Counter({'YOLO26':5,'YOLO11':5,'YOLOv8':5,'YOLOv9':2}),'Family membership')
    pareto=rows(MASTER/'metrics/PARETO_PIPELINE.csv')
    require([r['model'] for r in pareto]==list(models),'Pipeline Pareto order/coverage')
    # Independent sorted skyline, handling equal accuracy/cost groups exactly.
    sorted_models=sorted(overall,key=lambda r:(D(r['pipeline_ms_mean']),-D(r['mask_map50_95'])))
    best=None;best_cost=None;front=set()
    for r in sorted_models:
        acc,cost=D(r['mask_map50_95']),D(r['pipeline_ms_mean'])
        if best is None or acc>best or (acc==best and cost==best_cost):
            front.add(r['model']);best,best_cost=acc,cost
    for r in pareto:
        source=models[r['model']]
        for k in r:
            if k!='is_pareto':require(r[k]==source[k],f'Pareto source cell {r["model"]}/{k}')
        require((r['is_pareto']=='True')==(r['model'] in front),'Pipeline Pareto membership')
    require(front=={'YOLO26x-Seg','YOLO26l-Seg'},'Unexpected pipeline frontier')
    time={(r['model'],r['stage']):r for r in rows(MASTER/'metrics/TIMING_MASTER.csv')}
    composition=rows(MASTER/'metrics/TIMING_COMPOSITION.csv');require(len(composition)==17,'Timing coverage')
    stage_fields={'preprocess_ms':'preprocessing','inference_ms':'inference','postprocess_ms':'postprocessing',
                  'pipeline_ms':'pipeline','rle_preparation_ms':'rle_preparation','ultralytics_postprocess_inclusive_ms':'ultralytics_postprocess_inclusive'}
    for r in composition:
        for field,stage in stage_fields.items():require(r[field]==time[r['model'],stage]['mean_ms'],'Timing source precision')
        total=sum(D(r[k]) for k in ['preprocess_ms','inference_ms','postprocess_ms'])
        residual=total-D(r['pipeline_ms']);numeric(r['component_sum_minus_pipeline_ms'],residual)
        require(abs(residual)<=D('1e-10'),'Timing stage sum')
        for field in ['preprocess','inference','postprocess']:numeric(r[field+'_share_pct'],D(r[field+'_ms'])/D(r['pipeline_ms'])*100)
        numeric(sum(D(r[k+'_share_pct']) for k in ['preprocess','inference','postprocess']),100,'1e-10')
    require(sum(D(r['postprocess_share_pct'])>50 for r in composition)==12,'Postprocessing dominance count')
    sequence=rows(MASTER/'metrics/PER_SEQUENCE_MASTER.csv');lookup={(r['model'],r['sequence']):r for r in sequence}
    analysis=rows(MASTER/'metrics/PER_SEQUENCE_SCALING_ANALYSIS.csv');require(len(analysis)==68,'Sequence coverage')
    ladders={f:[f+s+'-Seg' for s in sizes] for f,sizes in [('YOLO26','xlmsn'),('YOLO11','xlmsn'),('YOLOv8','xlmsn'),('YOLOv9','ec')]}
    winners=[]
    for r in analysis:
        source=lookup[r['model'],r['sequence']]
        for k in ['model','tier','family','sequence','mask_map50_95','recall']:require(r[k]==source[k],'Sequence source precision')
        pool=[x for x in sequence if x['sequence']==r['sequence'] and x['tier']==r['tier']]
        rank=1+sum(D(x['mask_map50_95'])>D(source['mask_map50_95']) for x in pool)
        pooled=1+sum(x['tier']==r['tier'] and D(x['mask_map50_95'])>D(models[r['model']]['mask_map50_95']) for x in overall)
        require(int(r['within_tier_sequence_map_rank'])==rank and int(r['pooled_tier_map_rank'])==pooled,'Sequence rank')
        require(int(r['sequence_rank_minus_pooled_rank'])==rank-pooled,'Rank delta')
        ladder=ladders[r['family']];large,small=lookup[ladder[0],r['sequence']],lookup[ladder[-1],r['sequence']]
        require(r['endpoint_larger_model']==large['model'] and r['endpoint_smaller_model']==small['model'],'Endpoint names')
        d=D(small['mask_map50_95'])-D(large['mask_map50_95'])
        numeric(r['endpoint_map_delta_smaller_minus_larger'],d);numeric(r['endpoint_map_delta_pp'],d*100)
        numeric(r['endpoint_recall_delta_pp'],(D(small['recall'])-D(large['recall']))*100)
        index=ladder.index(r['model'])
        if index:
            prev=lookup[ladder[index-1],r['sequence']];require(r['adjacent_larger_model']==prev['model'],'Adjacent identity')
            numeric(r['adjacent_map_delta_pp'],(D(source['mask_map50_95'])-D(prev['mask_map50_95']))*100)
        else:require(r['adjacent_larger_model']==r['adjacent_map_delta_pp']=='','Missing adjacent marker')
        if rank==1:winners.append(r)
    require(len(winners)==20 and all(r['family']=='YOLO26' for r in winners),'20 sequence winners')
    for model in models:require(min((r for r in sequence if r['model']==model),key=lambda x:D(x['mask_map50_95']))['sequence']=='MOTS20-02','Lowest-sequence claim')
    for family,ladder in ladders.items():
        endpoints=[r for r in analysis if r['family']==family and r['model']==ladder[0]]
        require(min(endpoints,key=lambda r:D(r['endpoint_map_delta_pp']))['sequence']=='MOTS20-02','Largest endpoint loss')
        require(max(endpoints,key=lambda r:D(r['endpoint_map_delta_pp']))['sequence']==('MOTS20-09' if family=='YOLOv9' else 'MOTS20-11'),'Smallest endpoint loss')
    deltas=rows(MASTER/'metrics/SCALING_DELTAS.csv')
    require(sum(D(r['delta_absolute'])>0 for r in deltas if r['metric']=='pipeline_ms_mean')==8,'Pipeline inversion count')
    require(sum(D(r['delta_absolute'])>0 for r in deltas if r['metric']=='peak_allocated_vram_mib')==7,'Memory inversion count')
    for family in ['YOLO26','YOLO11','YOLOv8']:
        drops=[r for r in deltas if r['family']==family and r['metric']=='mask_map50_95']
        require(min(drops,key=lambda r:D(r['delta_absolute']))['smaller_tier']=='nano','Scaling-cliff claim')
    near=rows(MASTER/'metrics/NEAR_TIE_RESOURCE_DELTAS.csv');original=rows(MASTER/'metrics/NEAR_TIES.csv')
    require([(r['model_a'],r['model_b']) for r in near]==[(r['model_a'],r['model_b']) for r in original],'Near-tie pairs')
    for r in near:
        a,b=models[r['model_a']],models[r['model_b']];gap=abs(D(b['mask_map50_95'])-D(a['mask_map50_95']))
        require(r['direction']=='model_b_minus_model_a' and r['statistical_equivalence_tested']=='False','Near-tie semantics')
        require(gap<=D('0.001') and r['screen_threshold']=='0.001','Near-tie cutoff')
        numeric(r['map_abs_gap'],gap);numeric(r['map_abs_gap_pp'],gap*100)
        numeric(r['map_signed_delta_pp'],(D(b['mask_map50_95'])-D(a['mask_map50_95']))*100)
        for short,field in [('inference','inference_ms_mean'),('pipeline','pipeline_ms_mean'),('vram','peak_allocated_vram_mib'),('parameters','parameters'),('checkpoint_size','checkpoint_mb')]:
            require(r[short+'_a']==a[field] and r[short+'_b']==b[field],'Near-tie source cell')
            d=D(b[field])-D(a[field]);numeric(r[short+'_delta_b_minus_a'],d)
            numeric(r[short+'_relative_pct'],d/D(a[field])*100)
    return overall


def validate_history(revision):
    for path,h in revision['protected_sha256'].items():require(sha(ROOT/path)==h,f'Protected artifact changed: {path}')
    for edit in revision['editorial_revisions']:
        require(sha(ROOT/edit['archived_path'])==edit['sha256_before'],f'Archive mismatch: {edit["original_path"]}')
        require(edit['sha256_after'] and sha(ROOT/edit['original_path'])==edit['sha256_after'],f'After hash mismatch: {edit["original_path"]}')
    # Validate older immutable/editorial chains against preserved snapshots rather
    # than rewriting original provenance to describe the new authorized document.
    original=load(MASTER/'provenance/SOURCE_MANIFEST.json')
    language=load(sorted((MASTER/'provenance').glob('DOCUMENT_LANGUAGE_BY_ROLE_*.json'))[-1])
    candidates={}
    for e in revision['editorial_revisions']:candidates.setdefault(e['original_path'],[]).append((e['archived_path'],e['sha256_before']))
    for e in original['editorial_revisions']:candidates.setdefault(e['path'],[]).append((e['archive'],e['before_sha256']))
    for e in language['documents']:
        for snapshot in e['snapshots']:candidates.setdefault(e['path'],[]).append((snapshot['path'],snapshot['sha256']))
    expected=dict(language['immutable_artifact_sha256'])
    expected.update({MASTER.name+'/'+p:h for p,h in original['publication_sha256'].items()})
    expected.update({MASTER.name+'/'+p:h for p,h in original['control_sha256'].items()})
    expected[MASTER.name+'/scripts/build_master.py']=original['builder_sha256']
    for path,h in expected.items():
        if sha(ROOT/path)==h:continue
        require(any(h==old and sha(ROOT/archive)==h for archive,old in candidates.get(path,[])),f'Historical snapshot unavailable: {path}')
        changed=next((e for e in revision['editorial_revisions'] if e['original_path']==path),None)
        if changed:require(sha(ROOT/path)==changed['sha256_after'],'Authorized current hash')
        else:
            prior=next((e for e in original['editorial_revisions'] if e['path']==path),None)
            require(prior is not None and sha(ROOT/path)==prior['after_sha256'],f'Unrecorded editorial change: {path}')
    for e in revision.get('derived_plot_draft_archives',[]):require(sha(MASTER/e['archive'])==e['sha256'],'Draft plot archive')


def validate_docs(overall):
    revision=load(MASTER/'provenance/POST_STUDY_REVISION.json')
    for item in revision['active_markdown']:
        doc=ROOT/item['path'];require(doc.exists(),f'Active Markdown missing: {doc}')
        content=doc.read_text()
        if item['role']=='english_technical':require(not re.search('[ก-๙]',content),f'Thai prose in English role: {doc}')
        else:require('_TH' in doc.name and re.search('[ก-๙]',content),f'Thai summary role: {doc}')
    template=(MASTER/'templates/TIER_RESULTS_SUMMARY_TH_TEMPLATE.md').read_text()
    headings=re.findall(r'^## .+$',template,re.M);fields=['mask_map50_95','ap75','recall','inference_ms_mean','pipeline_ms_mean','fps','peak_allocated_vram_mib']
    tier_docs=[]
    report_fields=['mask_map50_95','ap50','ap75','precision','recall','f1','tp_iou_mean','tp_dice_mean','inference_ms_mean','pipeline_ms_mean','fps','peak_allocated_vram_mib','parameters','gflops','checkpoint_mb']
    for tier,slug in [('largest','Large'),('second_largest','Second_Largest'),('medium','Medium'),('small','Small'),('nano','Nano')]:
        root=ROOT/f'YOLO_{slug}_Seg_MOTS20_Benchmark';r=(root/'RESULTS_SUMMARY_TH.md').read_text();p=(root/'PRESENTATION_SUMMARY_TH.md').read_text();source=[x for x in overall if x['tier']==tier]
        require(re.findall(r'^## .+$',r,re.M)==headings,'Shared RESULTS structure')
        require('## สรุปผลจากตาราง' not in r and '![' not in r and '## Case ' not in r,'RESULTS role')
        require(len(r.encode())<len(p.encode())/2,'RESULTS must be substantially shorter')
        bullets=r.split('## สรุปใน 1 นาที',1)[1].split('## ผลลัพธ์หลัก',1)[0]
        require(5<=sum(line.startswith('- ') for line in bullets.splitlines())<=8,'Quick bullets')
        main=table(r,'## ผลลัพธ์หลัก');require(len(main)==len(source)+2,'Compact table coverage')
        for actual,s in zip(main[2:],source):require(actual==[s['model']]+[fmt(k,s[k]) for k in fields],'Tier displayed source values')
        winner=table(r,'## Winner ของแต่ละด้าน')
        categories=[('Mask mAP50-95','mask_map50_95',max,''),('AP75','ap75',max,''),('Recall','recall',max,''),('Inference speed','inference_ms_mean',min,' ms'),('Pipeline speed','pipeline_ms_mean',min,' ms'),('VRAM','peak_allocated_vram_mib',min,' MiB')]
        require(len(winner)==8,'Winner coverage')
        for actual,(title,k,fn,unit) in zip(winner[2:],categories):
            s=fn(source,key=lambda x:D(x[k]));require(actual==[title,s['model'],fmt(k,s[k])+unit],'Tier winners')
        require([x[0] for x in table(r,'## บทบาทของแต่ละโมเดล')[2:]]==[s['model'] for s in source],'Model role membership')
        # Check qualitative resource labels in the compact role rows against
        # measured extrema, in addition to checking all displayed numeric cells.
        for cells in table(r,'## บทบาทของแต่ละโมเดล')[2:]:
            model=cells[0];prose=' '.join(cells[1:]).lower()
            for pattern,key,fn in [(r'vram\s*ต่ำสุด|vram, parameters และ checkpoint ต่ำสุด','peak_allocated_vram_mib',min),
                                   (r'vram\s*สูงสุด','peak_allocated_vram_mib',max),
                                   (r'pipeline\s*(?:เร็วสุด|ต่ำสุด)','pipeline_ms_mean',min),
                                   (r'pipeline\s*ช้าที่สุด','pipeline_ms_mean',max),
                                   (r'(?:forward|inference)\s*เร็วสุด','inference_ms_mean',min),
                                   (r'(?:forward|inference)\s*ช้า(?:ที่สุด|สุด)','inference_ms_mean',max),
                                   (r'recall\s*ต่ำสุด','recall',min),(r'recall\s*สูงสุด','recall',max)]:
                if re.search(pattern,prose):require(model==fn(source,key=lambda x:D(x[key]))['model'],f'Unsupported model role: {model}/{key}')
        require(len(re.findall(r'^## Case ',p,re.M))==4,'Four preserved qualitative cases')
        for label in ['Failure Analysis','Near-tie visual check','สิ่งที่เรียนรู้จากภาพจริง','เมื่อดูทั้งตัวเลขและภาพร่วมกัน','ข้อจำกัด']:require('## '+label in p,'Qualitative depth')
        require('Observation' in p and 'Interpretation' in p,'Observation/Interpretation')
        require(len(re.findall(r'!\[',p))>=8,'Full frame plus ROI preserved')
        context=table(p,'## 1. ภาพรวมผลการทดลอง')
        for actual,s in zip(context[2:],source):require(actual==[s['model']]+[fmt(k,s[k]) for k in fields[:3]],'PRESENTATION context numbers')
        report=(root/'REPORT.md').read_text();main=table(report,'## 4. Overall Results')
        for actual,s in zip(main[2:],source):
            expected=[s['model']]
            for k in report_fields:
                expected.append(f'{int(s[k]):,}' if k=='parameters' else f'{float(s[k]):.2f}' if k=='checkpoint_mb' else f'{float(s[k]):.3f}' if k=='gflops' else fmt(k,s[k]))
            require(actual==expected,'REPORT technical table')
        require('All five tiers and the compatible 17-model Master synthesis are complete.' in report,'Current Master relation')
        tier_docs += [root/n for n in ['README.md','REPORT.md','RESULTS_SUMMARY_TH.md','PRESENTATION_SUMMARY_TH.md','EXPERIMENT_PROTOCOL.md']]
    master_docs=[MASTER/n for n in ['README.md','MASTER_RESULTS.md','RESEARCH_INSIGHTS_TH.md','MEETING_SUMMARY_TH.md','EXECUTIVE_SUMMARY_TH.md','YOLO26_SCALING_VISUAL_TH.md','NEAR_TIE_26S_11X_V9E_VISUAL_TH.md','YOLO26X_VS_YOLO26L_VISUAL_TH.md','YOLO_PARSING_SELECTION_PLAN_TH.md']]
    images=set()
    for doc in tier_docs+master_docs:
        content=doc.read_text();require(not re.search(r'สรุปสำหรับคุยกับพี่|สรุปสำหรับพี่|^#{1,3} .*mentor|อาจารย์',content,re.M|re.I),'Mentor-specific section')
        require(bool(re.search('[ก-๙]',content))==('_TH' in doc.name),'Language role')
        for link in re.findall(r'\]\(([^)]+)\)',content):
            if not re.match(r'https?://|#',link):require((doc.parent/link.split('#')[0]).exists(),f'Broken link: {doc.name}/{link}')
        images.update((doc.parent/link).resolve() for link in re.findall(r'!\[[^]]*\]\(([^)]+)\)',content) if not link.startswith('http'))
    for path in images:
        with Image.open(path) as im:im.verify()
    research=(MASTER/'RESEARCH_INSIGHTS_TH.md').read_text();meeting=(MASTER/'MEETING_SUMMARY_TH.md').read_text();technical=(MASTER/'MASTER_RESULTS.md').read_text()
    require(len(re.findall(r'^## ',research,re.M))==12,'Research question structure')
    require(len(re.findall(r'^## ',meeting,re.M))==11 and meeting.startswith('# YOLO Instance Segmentation Scaling Study — MOTS20'),'Meeting narrative structure')
    require(meeting.count('| Model |')<=1 and 'plots/13_timing_stage_composition.png' in meeting,'Meeting is not repeated tier tables')
    require(len(re.findall(r'^## ',technical,re.M))==16 and not re.search('[ก-๙]',technical),'Master technical role')
    require('VRAM เพิ่มใน YOLO26/YOLO11 และลดใน YOLOv8' in research,'s-to-n memory interpretation matches signed deltas')
    require(len((MASTER/'EXECUTIVE_SUMMARY_TH.md').read_text().splitlines())<=75,'Compact executive memo')
    claims=load(MASTER/'provenance/POST_STUDY_DOCUMENT_VALUE_CLAIMS.json')['claims']
    for claim in claims:
        row=next(r for r in rows(MASTER/claim['source']) if all(r[k]==v for k,v in claim['keys'].items()))
        number=D(row[claim['field']])*D(claim['scale'])
        display=f'{number:+.{claim["decimals"]}f}' if claim['signed'] else f'{number:.{claim["decimals"]}f}'
        require(display==claim['display'] and display in (MASTER/claim['document']).read_text(),'Numeric document claim')
    # Original complete 17-model table is preserved, including all 153 cells.
    original_claims=load(MASTER/'provenance/DOCUMENT_VALUE_CLAIMS.json')['claims']
    for claim in original_claims:
        if claim['document']!='MASTER_RESULTS.md':continue
        require(claim['display'] in technical,'Preserved original technical value')
    main=table(technical,'## 3. Master 17-Model Results')
    require(len(main)==19,'17-model technical table')
    human={'largest':'Largest (X/E)','second_largest':'Second-largest (L/C)','medium':'Medium (M)','small':'Small (S)','nano':'Nano (N)'}
    for actual,s in zip(main[2:],overall):
        expected=[s['model'],human[s['tier']]]+[fmt(k,s[k]) for k in fields]+[f"{int(s['parameters']):,}",f"{float(s['checkpoint_mb']):.2f}"]
        require(actual==expected,'Master 153 numeric cells')
    return len(images),len(claims)


def validate_visuals(recheck_predictions=False):
    checked=0
    for slug,expected in [('yolo26_scaling_ladder',3),('near_tie_26s_11x_v9e',4)]:
        evidence=load(MASTER/f'outputs/visualizations/{slug}/v1/EVIDENCE.json');require(len(evidence['cases'])==expected,'Visual case count')
        require(evidence['saved_predictions_only'] and not evidence['inference_rerun'],'Saved-only evidence')
        for source in evidence['sources']:require(sha(ROOT/source['per_frame_path'])==source['per_frame_sha256'],'Visual count source')
        source_lookup={s['model']:{(r['sequence'],int(r['frame'])):r for r in rows(ROOT/s['per_frame_path'])} for s in evidence['sources']}
        for case in evidence['cases']:
            for k in ['image','gt']:require(sha(ROOT/case[k+'_path'])==case[k+'_sha256'],'Original/GT hash')
            for kind in ['comparison','focus']:require(sha(MASTER/case[kind+'_path'])==case[kind+'_sha256'],'Visual PNG hash')
            require(len(case['roi_xyxy'])==4 and case['roi_xyxy'][0]<case['roi_xyxy'][2] and case['roi_xyxy'][1]<case['roi_xyxy'][3],'Identical ROI')
            for model in case['models']:
                require(sha(ROOT/model['prediction_path'])==model['prediction_sha256'],'Saved prediction hash')
                canonical=source_lookup[model['model']][case['sequence'],case['frame']]
                for k,v in model['counts'].items():require(v==int(canonical[k]),'Per-frame canonical count')
                require(len(model['matched_gt_ids'])==model['counts']['tp'] and len(model['fn_gt_ids'])==model['counts']['fn'],'GT sets')
                if recheck_predictions:
                    import gzip
                    sys.path.insert(0,str(ROOT/'YOLO_Large_Seg_MOTS20_Benchmark/src/frozen_pilot'))
                    from mots import load_frames
                    from metrics import CompactPrediction,fixed_metrics
                    frame=load_frames(ROOT/'datasets/MOTS/MOTS/train',case['sequence'],[case['frame']])[0]
                    saved=json.loads(gzip.decompress((ROOT/model['prediction_path']).read_bytes()))
                    require((saved['sequence'],saved['frame'],saved['width'],saved['height'])==(case['sequence'],case['frame'],frame.width,frame.height),'Saved frame identity')
                    predictions=[CompactPrediction(p['confidence'],p['class'],p['bbox_xyxy'],{'size':p['rle']['size'],'counts':p['rle']['counts'].encode('ascii')}) for p in saved['predictions']]
                    result=fixed_metrics(frame,predictions,.25,.5,.5)
                    for k,v in model['counts'].items():require(result[k]==v,'Rechecked saved-mask counts')
                    for target in model['targets']:
                        match=next((r for r in result['matches'] if r['object_id']==target['gt_id']),None)
                        require((match is not None)==target['matched'],'GT matching diagnostic')
                        if match:numeric(target['matched_iou'],match['iou'],'1e-12')
                checked+=1
    return checked


def validate(recheck_predictions=False):
    revision=load(MASTER/'provenance/POST_STUDY_REVISION.json')
    for field in ['measured_values_changed','inference_rerun','tier_benchmarks_rerun','canonical_measured_csv_changed']:require(revision[field] is False,'Scientific invariant '+field)
    state=load(MASTER/'STUDY_STATE.json');require(all(t['completion_status']=='COMPLETE' for t in state['tiers'].values()),'Completed state')
    validate_history(revision)
    overall=validate_math();images,claims=validate_docs(overall);visuals=validate_visuals(recheck_predictions)
    for field,h in revision.get('new_artifact_sha256',{}).items():require(sha(ROOT/field)==h,f'New artifact hash: {field}')
    # Source and merged rows must remain cell-for-cell identical.
    merged=[]
    for t in ['largest','second_largest','medium','small','nano']:
        repository=state['tiers'][t]['repository'];root=ROOT/repository
        require(subprocess.check_output(['git','-C',str(root),'branch','--show-current'],text=True).strip()=='main','Git branch')
        require(subprocess.check_output(['git','-C',str(root),'remote','get-url','origin'],text=True).strip()==f'https://github.com/folklazy/{repository}.git','Git remote')
        merged+=rows(root/'metrics/TIER_RESULTS.csv')
    require(merged==overall,'Merged canonical source cells')
    return {'status':'PASS_WITH_WARNINGS','scientific_invariants':'PASS','pipeline_pareto_validation':'PASS',
            'timing_composition_validation':'PASS','per_sequence_analysis_validation':'PASS','near_tie_resource_delta_validation':'PASS',
            'saved_predictions_only':True,'new_visual_inference_rerun':False,'same_frame_comparison_validation':'PASS',
            'active_visual_links_resolve':'PASS','results_presentation_role_separation':'PASS','research_insights_role':'PASS',
            'meeting_summary_role':'PASS','master_results_role':'PASS','executive_summary_role':'PASS','no_mentor_specific_section':True,
            'archives_created':'PASS','before_after_hashes_recorded':'PASS','protected_hashes_unchanged':'PASS',
            'protected_artifacts':len(revision['protected_sha256']),'new_document_value_claims':claims,'active_images_checked':images,
            'new_saved_prediction_comparisons_checked':visuals,'saved_masks_rechecked':recheck_predictions,
            'measured_values_changed':False,'inference_rerun':False,'tier_benchmarks_rerun':False,'canonical_measured_csv_changed':False,
            'git_validation':'PASS','warnings':['Inherited source tiers remain PASS_WITH_WARNINGS.','No statistical significance/equivalence test; correlated video frames.','Diagnostic visual cases are not representative sampling.','Timing sessions vary; small latency differences remain descriptive.']}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--recheck-saved-masks',action='store_true');parser.add_argument('--documents-only',action='store_true');parser.add_argument('--remote-images',action='store_true');args=parser.parse_args()
    with localcontext() as ctx:
        ctx.prec=50;result=validate(args.recheck_saved_masks)
    if args.remote_images:
        from verify_post_study_publication import verify
        result['publication']=verify()
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    try:main()
    except (ValueError,KeyError,OSError,AssertionError,StopIteration) as exc:
        print('[POST-STUDY] FAIL: '+str(exc),file=sys.stderr);sys.exit(1)
