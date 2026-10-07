"""Derived post-study analyses only: existing canonical CSVs, no model runtime."""
from pathlib import Path
from decimal import Decimal as D, localcontext
import csv
import json
import argparse

MASTER = Path(__file__).resolve().parents[1]
SEQUENCES = ['MOTS20-02', 'MOTS20-05', 'MOTS20-09', 'MOTS20-11']
FAMILIES = {'YOLO26': 'xlmsn', 'YOLO11': 'xlmsn', 'YOLOv8': 'xlmsn', 'YOLOv9': 'ec'}
NEW_CSVS = ['PARETO_PIPELINE.csv', 'TIMING_COMPOSITION.csv',
            'PER_SEQUENCE_SCALING_ANALYSIS.csv', 'NEAR_TIE_RESOURCE_DELTAS.csv']


def read(name):
    with (MASTER/'metrics'/name).open(newline='') as handle:
        return list(csv.DictReader(handle))


def write(name, rows):
    path = MASTER/'metrics'/name
    assert not path.exists(), f'Refusing to overwrite {path}'
    with path.open('x', newline='') as handle:
        out = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        out.writeheader()
        out.writerows(rows)


def relative(delta, base):
    return str(delta/D(base)*100) if D(base) else ''


def dominates(a, b, cost='pipeline_ms_mean'):
    aa, ba = D(a['mask_map50_95']), D(b['mask_map50_95'])
    ac, bc = D(a[cost]), D(b[cost])
    return aa >= ba and ac <= bc and (aa > ba or ac < bc)


def rank(row, pool):
    return 1 + sum(D(x['mask_map50_95']) > D(row['mask_map50_95']) for x in pool)


def derive():
    overall = read('MASTER_17_MODELS.csv')
    assert len(overall) == len({r['model'] for r in overall}) == 17
    models = {r['model']: r for r in overall}
    keys = ['model','tier','family','mask_map50_95','pipeline_ms_mean','fps',
            'inference_ms_mean','peak_allocated_vram_mib']
    pareto = [{**{k:r[k] for k in keys}, 'is_pareto':
               str(not any(dominates(x, r) for x in overall))} for r in overall]
    write('PARETO_PIPELINE.csv', pareto)
    stages = {(r['model'],r['stage']):r for r in read('TIMING_MASTER.csv')}
    composition = []
    for r in overall:
        get = lambda stage: stages[r['model'],stage]['mean_ms']
        fields = {'preprocess_ms':get('preprocessing'),'inference_ms':get('inference'),
                  'postprocess_ms':get('postprocessing'),'pipeline_ms':get('pipeline'),
                  'rle_preparation_ms':get('rle_preparation'),
                  'ultralytics_postprocess_inclusive_ms':get('ultralytics_postprocess_inclusive')}
        residual = sum(D(fields[k]) for k in ['preprocess_ms','inference_ms','postprocess_ms'])-D(fields['pipeline_ms'])
        assert abs(residual) <= D('1e-10'), (r['model'],residual)
        shares = {f'{s}_share_pct':str(D(fields[f'{s}_ms'])/D(fields['pipeline_ms'])*100)
                  for s in ['inference','postprocess','preprocess']}
        composition.append({'model':r['model'],'tier':r['tier'],'family':r['family'],
                            **fields,**shares,'component_sum_minus_pipeline_ms':str(residual)})
    write('TIMING_COMPOSITION.csv', composition)
    seq = read('PER_SEQUENCE_MASTER.csv')
    lookup = {(r['model'],r['sequence']):r for r in seq}
    sequence_analysis = []
    for r in seq:
        family = r['family']; sizes = FAMILIES[family]
        ladder = [family+s+'-Seg' for s in sizes]; index = ladder.index(r['model'])
        large, small = lookup[ladder[0],r['sequence']], lookup[ladder[-1],r['sequence']]
        previous = lookup[ladder[index-1],r['sequence']] if index else None
        pool = [x for x in seq if x['tier']==r['tier'] and x['sequence']==r['sequence']]
        pooled = rank(models[r['model']], [x for x in overall if x['tier']==r['tier']])
        sequence_analysis.append({
            'model':r['model'],'tier':r['tier'],'family':family,'sequence':r['sequence'],
            'mask_map50_95':r['mask_map50_95'],'recall':r['recall'],
            'within_tier_sequence_map_rank':rank(r,pool),'pooled_tier_map_rank':pooled,
            'sequence_rank_minus_pooled_rank':rank(r,pool)-pooled,
            'endpoint_larger_model':large['model'],'endpoint_smaller_model':small['model'],
            'endpoint_map_delta_smaller_minus_larger':str(D(small['mask_map50_95'])-D(large['mask_map50_95'])),
            'endpoint_map_delta_pp':str((D(small['mask_map50_95'])-D(large['mask_map50_95']))*100),
            'endpoint_recall_delta_pp':str((D(small['recall'])-D(large['recall']))*100),
            'adjacent_larger_model':previous['model'] if previous else '',
            'adjacent_map_delta_pp':str((D(r['mask_map50_95'])-D(previous['mask_map50_95']))*100) if previous else '',
        })
    write('PER_SEQUENCE_SCALING_ANALYSIS.csv', sequence_analysis)
    pairs = read('NEAR_TIES.csv'); near = []
    for pair in pairs:
        a,b = models[pair['model_a']],models[pair['model_b']]
        gap = abs(D(a['mask_map50_95'])-D(b['mask_map50_95']))
        assert gap <= D('0.001')
        out = {'model_a':a['model'],'model_b':b['model'],'tier_a':a['tier'],'tier_b':b['tier'],
               'direction':'model_b_minus_model_a','map_a':a['mask_map50_95'],'map_b':b['mask_map50_95'],
               'map_abs_gap':str(gap),'map_abs_gap_pp':str(gap*100),
               'map_signed_delta_pp':str((D(b['mask_map50_95'])-D(a['mask_map50_95']))*100),
               'screen_threshold':'0.001','statistical_equivalence_tested':'False'}
        for short,field in [('inference','inference_ms_mean'),('pipeline','pipeline_ms_mean'),
                            ('vram','peak_allocated_vram_mib'),('parameters','parameters'),
                            ('checkpoint_size','checkpoint_mb')]:
            delta = D(b[field])-D(a[field])
            out.update({short+'_a':a[field],short+'_b':b[field],
                        short+'_delta_b_minus_a':str(delta),short+'_relative_pct':relative(delta,a[field])})
        near.append(out)
    write('NEAR_TIE_RESOURCE_DELTAS.csv', near)
    summary = {'scope':'DERIVED FROM EXISTING CANONICAL RESULTS',
               'inference_rerun':False,'timing_rerun':False,'measured_values_changed':False,
               'pipeline_pareto':[r['model'] for r in pareto if r['is_pareto']=='True'],
               'component_sum_tolerance_ms':'1e-10',
               'postprocessing_over_half_pipeline_models':[r['model'] for r in composition if D(r['postprocess_share_pct'])>50],
               'tier_sequence_winners':[{k:r[k] for k in ['tier','sequence','model','mask_map50_95']} for r in sequence_analysis if r['within_tier_sequence_map_rank']==1],
               'rank_inversions':[{k:r[k] for k in ['tier','sequence','model','within_tier_sequence_map_rank','pooled_tier_map_rank']} for r in sequence_analysis if r['sequence_rank_minus_pooled_rank']],
               'near_tie_pairs':len(near)}
    path=MASTER/'provenance/POST_STUDY_ANALYSIS.json'
    assert not path.exists()
    path.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


def plot(refine_drafts=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    overall=read('MASTER_17_MODELS.csv');pareto=read('PARETO_PIPELINE.csv')
    colors={'YOLO26':'#1b7f5a','YOLO11':'#276fbf','YOLOv8':'#c75b12','YOLOv9':'#8856a7'}
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    front=[r for r in pareto if r['is_pareto']=='True']
    def save(fig,name):
        target=MASTER/'plots'/name
        import io
        buffer=io.BytesIO();fig.savefig(buffer,dpi=170,bbox_inches='tight');payload=buffer.getvalue()
        if target.exists() and target.read_bytes()==payload:
            plt.close(fig);return
        if target.exists():
            assert refine_drafts, target
            import shutil,hashlib
            revision=json.loads((MASTER/'provenance/POST_STUDY_REVISION.json').read_text())
            archived=MASTER/'reports/archive'/revision['revision_id']/'draft_plots'/name
            archived.parent.mkdir(parents=True,exist_ok=True)
            if archived.exists():
                index=2
                while archived.with_name(archived.stem+f'.iteration_{index:02d}.png').exists():index+=1
                archived=archived.with_name(archived.stem+f'.iteration_{index:02d}.png')
            shutil.copyfile(target,archived)
            revision.setdefault('derived_plot_draft_archives',[]).append({'path':str(target.relative_to(MASTER)),
                'archive':str(archived.relative_to(MASTER)),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                'reason':'Improve close-point label readability and show shares to two decimals'})
            (MASTER/'provenance/POST_STUDY_REVISION.json').write_text(json.dumps(revision,ensure_ascii=False,indent=2)+'\n')
        target.write_bytes(payload);plt.close(fig)
    for n,title,draw_front in [(11,'Accuracy versus measured pipeline latency',False),(12,'Accuracy / pipeline Pareto frontier',True)]:
        fig,ax=plt.subplots(figsize=(13,8))
        for r in overall:
            x,y=float(r['pipeline_ms_mean']),float(r['mask_map50_95'])*100
            ax.scatter(x,y,color=colors[r['family']],s=70)
            offset={'YOLOv9c-Seg':(35,15),'YOLO11m-Seg':(145,-10)}.get(r['model'],(5,7))
            ax.annotate(r['model'].replace('-Seg',''),(x,y),xytext=offset,textcoords='offset points',fontsize=9,
                        arrowprops={'arrowstyle':'-','color':'#888888','lw':.7} if offset!=(5,7) else None)
        if draw_front:
            ordered=sorted(front,key=lambda r:float(r['pipeline_ms_mean']))
            ax.plot([float(r['pipeline_ms_mean']) for r in ordered],[float(r['mask_map50_95'])*100 for r in ordered],color='black',lw=1.5,label='Nondominated frontier')
            for r in ordered:ax.scatter(float(r['pipeline_ms_mean']),float(r['mask_map50_95'])*100,s=190,facecolors='none',edgecolors='black',lw=1.5)
            ax.legend()
        ax.set(xlabel='Mean pipeline latency (ms/frame; lower is better)',ylabel='Mask mAP50-95 (%)',title=title)
        ax.grid(alpha=.2);fig.text(.1,.01,'Derived from unchanged canonical results; pipeline excludes decode, RLE preparation and output writing.',fontsize=10)
        save(fig,f'{n:02d}_'+('pareto_accuracy_pipeline' if draw_front else 'accuracy_vs_pipeline')+'.png')
    rows=read('TIMING_COMPOSITION.csv');fig,ax=plt.subplots(figsize=(13,10));y=np.arange(len(rows));left=np.zeros(len(rows))
    for field,label,col in [('preprocess_ms','Preprocessing','#9ecae1'),('inference_ms','Inference','#3182bd'),('postprocess_ms','Postprocessing','#e6550d')]:
        values=np.array([float(r[field]) for r in rows]);ax.barh(y,values,left=left,color=col,label=label);left+=values
    ax.set_yticks(y,[r['model'].replace('-Seg','') for r in rows]);ax.invert_yaxis()
    ax.set(xlabel='Mean stage latency (ms/frame)',title='Native-mask pipeline stage composition')
    ax.legend(loc='lower right');ax.grid(axis='x',alpha=.2)
    for i,r in enumerate(rows):ax.text(left[i]+1,i,f"post {float(r['postprocess_share_pct']):.2f}%",va='center',fontsize=9)
    ax.set_xlim(0,max(left)+24);fig.text(.1,.01,'Only preprocessing + inference + postprocessing are stacked. RLE is separate; inclusive diagnostic is not added.',fontsize=9)
    save(fig,'13_timing_stage_composition.png')
    lookup={(r['model'],r['sequence']):float(r['mask_map50_95'])*100 for r in read('PER_SEQUENCE_MASTER.csv')}
    values=np.array([[lookup[r['model'],s] for s in SEQUENCES] for r in overall])
    fig,ax=plt.subplots(figsize=(10,12));im=ax.imshow(values,cmap='YlGnBu',vmin=values.min(),vmax=values.max(),aspect='auto')
    ax.set_xticks(range(4),SEQUENCES);ax.set_yticks(range(17),[r['model'].replace('-Seg','') for r in overall])
    for i in range(17):
        for j in range(4):ax.text(j,i,f'{values[i,j]:.2f}',ha='center',va='center',color='white' if values[i,j]>52 else 'black')
    for boundary in [3.5,7.5,10.5,13.5]:ax.axhline(boundary,color='white',lw=2)
    fig.colorbar(im,ax=ax,label='Per-sequence Mask mAP50-95 (%)');ax.set_title('17-model per-sequence mask accuracy')
    fig.text(.1,.01,'Source values retained; display rounded. Per-sequence AP is separate from pooled AP. No scene-cause attribution.',fontsize=9)
    save(fig,'14_per_sequence_map_heatmap.png')
    print('[POST-STUDY] four derived plots generated')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plots-only',action='store_true');parser.add_argument('--refine-drafts',action='store_true');args=parser.parse_args()
    with localcontext() as ctx:
        ctx.prec=50
        if not args.plots_only:derive()
        assert not args.refine_drafts or args.plots_only
        plot(args.refine_drafts)
