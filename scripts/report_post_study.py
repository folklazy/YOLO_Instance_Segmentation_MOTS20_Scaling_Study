"""Editorial reports from existing canonical and explicitly derived CSVs only."""
from pathlib import Path
from decimal import Decimal as D, localcontext
import csv,json,re
from post_study_analysis import MASTER, read

WORKSPACE=MASTER.parent
TIERS=[('largest','Large','Largest (X/E)'),('second_largest','Second_Largest','Second-largest (L/C)'),
       ('medium','Medium','Medium'),('small','Small','Small'),('nano','Nano','Nano')]
ROWS=read('MASTER_17_MODELS.csv');MODELS={r['model']:r for r in ROWS}
CLAIMS=[];CURRENT=''


def value(source,keys,field,decimals=3,scale=1,signed=False):
    rows=read(source);r=next(x for x in rows if all(x[k]==v for k,v in keys.items()))
    number=D(r[field])*D(scale)
    text=f'{number:+.{decimals}f}' if signed else f'{number:.{decimals}f}'
    CLAIMS.append({'document':CURRENT,'source':'metrics/'+source,'keys':keys,'field':field,
                   'decimals':decimals,'scale':str(scale),'signed':signed,'display':text})
    return text


def v(model,field,decimals=3,scale=1):
    return value('MASTER_17_MODELS.csv',{'model':model},field,decimals,scale)


def delta(family,smaller,metric,field='delta_relative_pct',decimals=2):
    return value('SCALING_DELTAS.csv',{'family':family,'smaller_model':smaller,'metric':metric},field,decimals,signed=True)


def timing(model,field,decimals=2):
    return value('TIMING_COMPOSITION.csv',{'model':model},field,decimals)


def near(a,b,field,decimals=2):
    return value('NEAR_TIE_RESOURCE_DELTAS.csv',{'model_a':a,'model_b':b},field,decimals,signed='delta' in field or 'relative' in field)


def save(name,body):
    (MASTER/name).write_text(body.strip()+'\n')


def original(name):
    revision=json.loads((MASTER/'provenance/POST_STUDY_REVISION.json').read_text())
    entry=next(e for e in revision['editorial_revisions'] if e['original_path']==MASTER.name+'/'+name)
    return (WORKSPACE/entry['archived_path']).read_text()


def tier_summaries():
    roles={
      'YOLO26x-Seg':['นำ mAP/AP75/Recall และ TP-only quality','Inference ช้าที่สุด; VRAM สูงกว่า YOLOv9e','ยอมเพิ่ม latency เพื่อ accuracy'],
      'YOLO11x-Seg':['mAP near-tied กับ YOLOv9e; AP75 สูงกว่าเล็กน้อย','Recall ต่ำกว่าและ latency/VRAM สูงกว่า YOLOv9e','ตรวจ trade-off ของ mask overlap กับ coverage'],
      'YOLOv9e-Seg':['Inference/pipeline เร็วสุด; VRAM ต่ำสุด','mAP/AP75 ต่ำกว่า YOLO26x','Latency หรือ memory เป็นข้อจำกัด'],
      'YOLOv8x-Seg':['Forward เร็วกว่า YOLO11x/YOLO26x','mAP ต่ำสุด; VRAM สูงสุด; pipeline ใกล้ YOLO26x','ต้องการ baseline รุ่นก่อน'],
      'YOLO26l-Seg':['นำ mAP/AP75/Recall; pipeline และ VRAM ต่ำสุด','Forward ช้ากว่า YOLO11l/YOLOv9c','เน้น accuracy ร่วมกับ pipeline/memory'],
      'YOLO11l-Seg':['mAP/AP75 อันดับสอง; forward ใกล้ YOLOv9c','mAP ต่ำกว่า YOLO26l; pipeline/VRAM สูงกว่า','Forward เป็นข้อจำกัดและต้องการเทียบกับ YOLOv9c'],
      'YOLOv9c-Seg':['Inference ต่ำสุดตาม mean ที่วัด','mAP ต่ำสุด; pipeline/VRAM ไม่ต่ำสุด','สนใจ forward โดยไม่ถือช่องว่างเล็กว่ามีนัยสำคัญ'],
      'YOLOv8l-Seg':['mAP/Recall สูงกว่า YOLOv9c','Inference/pipeline ช้าสุด; VRAM สูงสุด','ต้องการ baseline และตรวจอันดับต่างกันตาม metric'],
      'YOLO26m-Seg':['นำ mAP/AP75/Recall และ TP-only quality','Forward ช้าสุด; VRAM สูงกว่า YOLO11m','Accuracy สำคัญกว่า forward latency'],
      'YOLO11m-Seg':['Pipeline และ VRAM ต่ำสุดใน tier','mAP ต่ำกว่า YOLO26m; forward ช้ากว่า YOLOv8m','เปรียบเทียบ pipeline ที่ใกล้กันเชิงพรรณนา'],
      'YOLOv8m-Seg':['Inference เร็วสุด','mAP ต่ำสุด; pipeline/VRAM สูงสุด','Forward latency เป็นข้อจำกัดหลัก'],
      'YOLO26s-Seg':['นำ mAP/AP75/Recall; pipeline/VRAM ต่ำสุด','Forward ช้าที่สุดใน tier','เน้น accuracy ของ Small และ native-mask pipeline'],
      'YOLO11s-Seg':['mAP near-tied กับ YOLOv8s; pipeline/VRAM ต่ำกว่า','Recall และ forward ด้อยกว่า YOLOv8s','ตรวจ resource trade-off ภายในคู่ near tie'],
      'YOLOv8s-Seg':['Forward เร็วสุด; Recall สูงกว่า YOLO11s','Pipeline/VRAM สูงสุด; AP75 ต่ำกว่า YOLO11s','สนใจ forward/coverage พร้อมยอมรับต้นทุน pipeline'],
      'YOLO26n-Seg':['นำ mAP/AP75/Precision; pipeline เร็วสุด','Recall ต่ำสุด; forward ช้าสุด','เน้น mask AP และ pipeline โดยตรวจ coverage เพิ่ม'],
      'YOLO11n-Seg':['VRAM, parameters และ checkpoint ต่ำสุดใน tier','Pipeline ช้าที่สุด; Recall ต่ำกว่า YOLOv8n','Memory หรือขนาด checkpoint เป็นข้อจำกัด'],
      'YOLOv8n-Seg':['Recall สูงสุดและ inference เร็วสุด','Precision/mAP ต่ำสุด; VRAM สูงสุด','สนใจ forward หรือ Recall พร้อมตรวจ FP'],
    }
    findings={
      'largest':['YOLO11x/YOLOv9e เป็น descriptive near tie ของ mAP; AP75 นำใน YOLO11x แต่ Recall และทรัพยากรนำใน YOLOv9e',
                 'YOLOv8x forward เร็วกว่า YOLO26x แต่ pipeline ใกล้กัน และใช้ VRAM มากกว่า'],
      'second_largest':['YOLO26l นำ accuracy, pipeline และ memory พร้อมกันตามค่าที่วัด แต่ไม่ได้มี forward เร็วสุด',
                        'YOLOv9c/YOLO11l มี forward ใกล้กันมาก และ pipeline ของสามรุ่นนำอยู่ใกล้กัน; ยังไม่มี significance test'],
      'medium':['YOLOv8m forward เร็วสุด แต่ pipeline ช้าสุด; อันดับ forward จึงกลับทิศเมื่อดูงานทั้ง pipeline',
                'YOLO11m pipeline ต่ำสุดตาม mean แต่ใกล้ YOLO26m; ไม่อ้างความเหนือกว่าที่แน่นอนจากช่องว่างเล็ก'],
      'small':['YOLO11s/YOLOv8s เป็น descriptive near tie ของ mAP แต่ AP75, Recall และ resource ranking ต่างกัน',
               'YOLO26s มี forward ช้าที่สุดใน tier แต่ pipeline เร็วที่สุด; postprocessing เป็นส่วนหนึ่งของผลรวม'],
      'nano':['YOLOv8n นำ Recall แต่มี Precision ต่ำสุด; YOLO26n นำ mAP/AP75 แต่ Recall ต่ำสุด',
              'YOLOv8n forward เร็วสุด แต่ YOLO26n pipeline เร็วสุด; YOLO11n ใช้ VRAM ต่ำสุดแต่ pipeline ช้าที่สุด'],
    }
    fields=['mask_map50_95','ap75','recall','inference_ms_mean','pipeline_ms_mean','fps','peak_allocated_vram_mib']
    fmt=lambda key,n:f'{D(n):.{2 if key=="peak_allocated_vram_mib" else 3 if key in fields[3:6] else 6}f}'
    categories=[('Mask mAP50-95','mask_map50_95',max,''),('AP75','ap75',max,''),('Recall','recall',max,''),
                ('Inference speed','inference_ms_mean',min,' ms'),('Pipeline speed','pipeline_ms_mean',min,' ms'),('VRAM','peak_allocated_vram_mib',min,' MiB')]
    for tier,slug,human in TIERS:
        directory=WORKSPACE/f'YOLO_{slug}_Seg_MOTS20_Benchmark';rows=[r for r in ROWS if r['tier']==tier]
        winner=max(rows,key=lambda r:D(r['mask_map50_95']));runner=sorted(rows,key=lambda r:D(r['mask_map50_95']),reverse=True)[1]
        fast=min(rows,key=lambda r:D(r['inference_ms_mean']));pipe=min(rows,key=lambda r:D(r['pipeline_ms_mean']));mem=min(rows,key=lambda r:D(r['peak_allocated_vram_mib']))
        gap=(D(winner['mask_map50_95'])-D(runner['mask_map50_95']))*100
        main='\n'.join('| '+' | '.join([r['model']]+[fmt(k,r[k]) for k in fields])+' |' for r in rows)
        win='\n'.join('| '+title+' | '+fn(rows,key=lambda r:D(r[key]))['model']+' | '+fmt(key,fn(rows,key=lambda r:D(r[key]))[key])+unit+' |' for title,key,fn,unit in categories)
        role='\n'.join('| '+' | '.join([r['model']]+roles[r['model']])+' |' for r in rows)
        pairs=[p for p in read('NEAR_TIES.csv') if p['tier_a']==tier and p['tier_b']==tier]
        tie=('; '.join(f"{p['model_a']}/{p['model_b']}: ต่าง {D(p['map_abs_gap'])*100:.6f} percentage points" for p in pairs)
             if pairs else 'ไม่มีคู่ผ่าน descriptive mAP near-tie screen ≤0.001; ความใกล้ของ latency เป็นคนละประเด็น')
        memory=(f"{winner['model']} ใช้ VRAM มากกว่า {mem['model']} {D(winner['peak_allocated_vram_mib'])-D(mem['peak_allocated_vram_mib']):.2f} MiB เพื่อ mAP สูงกว่า {(D(winner['mask_map50_95'])-D(mem['mask_map50_95']))*100:.3f} percentage points"
                if winner!=mem else f"{winner['model']} นำทั้ง mAP และ allocated VRAM ต่ำสุดใน tier นี้ จึงไม่มีการแลก accuracy ลงเพื่อ memory ที่ต่ำกว่าในคู่ที่วัด")
        body=f'''# สรุปผล {human} YOLO Instance Segmentation

## สรุปใน 1 นาที

- โมเดล: {', '.join(r['model'] for r in rows)}
- MOTS20 2,862 frames / 26,894 Person GT instances รายเฟรม
- Official pretrained checkpoints; ไม่มี training หรือ fine-tuning; สถานะ PASS WITH WARNINGS
- Accuracy สูงสุด: {winner['model']} — Mask mAP50-95 {fmt('mask_map50_95',winner['mask_map50_95'])}
- Inference เร็วสุด: {fast['model']} — {fmt('inference_ms_mean',fast['inference_ms_mean'])} ms
- Pipeline เร็วสุด: {pipe['model']} — {fmt('pipeline_ms_mean',pipe['pipeline_ms_mean'])} ms / {fmt('fps',pipe['fps'])} FPS
- Peak allocated VRAM ต่ำสุด: {mem['model']} — {fmt('peak_allocated_vram_mib',mem['peak_allocated_vram_mib'])} MiB
- Trade-off หลัก: ตัวนำ mAP สูงกว่ารองอันดับสอง {gap:.3f} percentage points; ต้องแยก forward จาก pipeline

## ผลลัพธ์หลัก

| Model | Mask mAP50-95 | AP75 | Recall | Inference ms | Pipeline ms | FPS | Peak VRAM MiB |
|---|---|---|---|---|---|---|---|
{main}

AP/Recall เป็น fraction ช่วง 0–1; latency เป็น ms/frame และ FPS มาจาก mean pipeline

## Winner ของแต่ละด้าน

| ด้าน | Model | Result |
|---|---|---|
{win}

## สิ่งที่ตัวเลขบอกเรา

- {winner['model']} นำ {runner['model']} ด้าน mAP {gap:.3f} percentage points
- {findings[tier][0]}
- {findings[tier][1]}
- {tie}; near tie ไม่ใช่ equivalence หรือ statistical significance

## บทบาทของแต่ละโมเดล

| Model | จุดเด่น | สิ่งที่แลก | เหมาะพิจารณาเมื่อ |
|---|---|---|---|
{role}

## Trade-off หลัก

### Accuracy vs Speed

{winner['model']} มี mAP {fmt('mask_map50_95',winner['mask_map50_95'])}; ตัว forward เร็วสุด {fast['model']} มี mAP {fmt('mask_map50_95',fast['mask_map50_95'])} และ inference ต่ำกว่า {D(winner['inference_ms_mean'])-D(fast['inference_ms_mean']):.3f} ms ส่วน pipeline ต้องดู {pipe['model']} แยก ไม่ถือว่า forward winner เป็น throughput winner

### Accuracy vs Memory

{memory} ไม่ใช้ชื่อขนาดหรือ parameters แทน memory measurement

## ข้อควรระวังในการตีความ

ไม่มี significance test; ภาพวิดีโอสัมพันธ์กัน TP-only quality วัดเฉพาะคู่ที่ match และ Recall เป็น mask matching ไม่ใช่ box Recall Pipeline ไม่รวม decode, RLE preparation และการเขียนผล; VRAM เป็น peak allocated ภายใต้ benchmark นี้ การแบ่ง tier ไม่ทำให้ capacity/pretraining เท่ากัน และยังไม่ยืนยัน CCTV robustness ไม่มี weighted score หรือผู้ชนะทุกข้อจำกัด

## รายละเอียดเพิ่มเติม

[REPORT.md](REPORT.md) · [รายงานวิจัยภาพเชิงคุณภาพ](PRESENTATION_SUMMARY_TH.md) · [TIER_RESULTS.csv](metrics/TIER_RESULTS.csv) · [Master Study](https://github.com/folklazy/YOLO_Instance_Segmentation_MOTS20_Scaling_Study)
'''
        (directory/'RESULTS_SUMMARY_TH.md').write_text(body)
        report=directory/'REPORT.md';s=report.read_text()
        a=s.index('## 12. Relation to Full Scaling Study');b=s.index('## Qualitative Analysis',a)
        s=s[:a]+'''## 12. Relation to Full Scaling Study

All five tiers and the compatible 17-model Master synthesis are complete. See the [Master Study](https://github.com/folklazy/YOLO_Instance_Segmentation_MOTS20_Scaling_Study) for cross-tier scaling, separate Pareto objectives and post-study derived analyses. This tier retains its original measured results and frozen protocol; this editorial revision starts no benchmark.

'''+s[b:];report.write_text(s)


def candidate_table(english=False):
    if english:
        return '''| Priority | Candidate | Evidence / qualification |
|---|---|---|
| Accuracy | YOLO26x-Seg | Highest pooled mAP/AP75/Recall; selected visual counterexamples remain |
| Whole-pipeline efficiency | YOLO26l-Seg | Lowest measured pipeline mean; second-highest mAP; pipeline Pareto |
| Forward latency | YOLOv8n-Seg | Lowest inference mean; not the pipeline winner |
| Allocated VRAM | YOLO26l-Seg | Lowest measured allocator peak; not total deployment memory |
| Small-model accuracy | YOLO26s-Seg | Highest Small mAP; near-equal aggregate mAP to 11x/v9e with lower Recall |
| Later CCTV robustness evaluation | YOLO26x / YOLO26l / YOLO26s | Conditional candidates; no CCTV winner established |'''
    return '''| Priority | Candidate | หลักฐานและข้อแลกเปลี่ยน |
|---|---|---|
| Accuracy | YOLO26x-Seg | นำ pooled mAP/AP75/Recall แต่มี counterexamples ในภาพจริง |
| Whole-pipeline efficiency | YOLO26l-Seg | Pipeline mean ต่ำสุดและ mAP อันดับสอง; อยู่บน pipeline Pareto |
| Forward latency | YOLOv8n-Seg | Inference ต่ำสุด แต่ pipeline ไม่เร็วสุด |
| Memory | YOLO26l-Seg | Peak allocated VRAM ต่ำสุดใน benchmark; ไม่ใช่ total deployment memory |
| Small-model accuracy | YOLO26s-Seg | นำ Small mAP; mAP ใกล้ 11x/v9e แต่ Recall ต่ำกว่า |
| Later CCTV robustness evaluation | YOLO26x / YOLO26l / YOLO26s | Candidates ตามข้อจำกัด ยังไม่ยืนยัน CCTV superiority |'''


def master_reports():
    global CURRENT
    CURRENT='RESEARCH_INSIGHTS_TH.md'
    body=f'''# Research Insights — YOLO Instance Segmentation Scaling on MOTS20

## สรุปใน 1 นาที

- YOLO26 นำ pooled mAP/AP75 ทั้งห้า tiers และนำ sequence mAP ครบ 20 tier × sequence comparisons; เป็นผลของ checkpoints ที่วัด ไม่ใช่ข้อพิสูจน์สาเหตุจาก architecture
- YOLO26x → l ลด mAP {delta('YOLO26','YOLO26l-Seg','mask_map50_95','delta_percentage_points',3)} percentage points แลก inference {delta('YOLO26','YOLO26l-Seg','inference_ms_mean')}% และ pipeline {delta('YOLO26','YOLO26l-Seg','pipeline_ms_mean')}%
- Accuracy–pipeline Pareto เหลือ x/l; objective นี้ให้ข้อสรุปต่างจาก accuracy–inference ที่มีเก้า candidates
- Postprocessing มากกว่าครึ่ง pipeline ใน 12 จาก 17 โมเดล; forward ที่เร็วขึ้นไม่รับประกัน throughput ที่สูงขึ้น
- YOLO26s/YOLO11x/YOLOv9e มี aggregate mAP ใกล้กันมาก แต่ resource และ Recall ต่างกัน
- Ranking ของรุ่นรองเปลี่ยนบาง sequence; ทุกตระกูลมี endpoint accuracy drop มากสุดใน MOTS20-02 แต่ยังไม่ระบุสาเหตุจากฉาก
- s → n เป็น adjacent mAP drop ที่มากสุดในทั้ง YOLO26/YOLO11/YOLOv8 ขณะที่ pipeline และ memory ไม่ได้ดีขึ้นเสมอ

## 1. YOLO26 นำ accuracy อย่างสม่ำเสมอแค่ไหน

**Observation:** YOLO26 นำ mAP/AP75 รวมทั้งห้า tiers และมี mAP สูงสุดในทุก tier × sequence รวม 20 comparisons YOLO26x มี pooled mAP {v('YOLO26x-Seg','mask_map50_95',6)} แต่ Recall winner ใน Nano เป็น YOLOv8n ไม่ใช่ YOLO26n

**Interpretation:** หลักฐานสอดคล้องกันทั้ง pooled และ per-sequence accuracy ภายใต้ protocol นี้ แต่การแบ่ง tier ไม่ทำให้ parameters หรือ pretraining เท่ากัน จึงไม่แยกผลของ architecture ออกจาก capacity/checkpoint/training เดิม และไม่รับรองว่าตัวนำจะเก็บ GT เพิ่มทุกเฟรม

## 2. จุดลดต้นทุนที่เด่น: YOLO26x → YOLO26l

| ด้าน | การเปลี่ยน l − x |
|---|---:|
| Mask mAP50-95 | {delta('YOLO26','YOLO26l-Seg','mask_map50_95','delta_percentage_points',3)} percentage points |
| Recall | {delta('YOLO26','YOLO26l-Seg','recall','delta_percentage_points',3)} percentage points |
| Inference | {delta('YOLO26','YOLO26l-Seg','inference_ms_mean')}% |
| Pipeline | {delta('YOLO26','YOLO26l-Seg','pipeline_ms_mean')}% |
| Peak allocated VRAM | {delta('YOLO26','YOLO26l-Seg','peak_allocated_vram_mib')}% |
| Parameters | {delta('YOLO26','YOLO26l-Seg','parameters')}% |

**Observation:** l มี mAP อันดับสอง และ pipeline/allocated VRAM ต่ำสุดทั้งชุด Accuracy–pipeline และ accuracy–VRAM Pareto ต่างเหลือเพียง YOLO26x/YOLO26l

**Interpretation:** x → l เป็น candidate efficiency point หรือ descriptive knee ภายใต้ objectives ที่วัด ไม่ใช่ universal sweet spot เพราะบางงานให้ความสำคัญกับ forward latency หรือ checkpoint size มากกว่า

อ่าน [รายงานภาพ x/l แบบละเอียด](YOLO26X_VS_YOLO26L_VISUAL_TH.md): same-frame saved predictions แสดงทั้ง coverage advantage, mask overlap ต่างกัน, l counterexample และ common failure โดยไม่อ้างว่า x ดีกว่าทุกคนในทุกเฟรม

## 3. ทำไม model เล็กลงแต่ pipeline ไม่เร็วขึ้นเสมอ

![Timing stage composition](plots/13_timing_stage_composition.png)

**Observation:** YOLO26x มี inference share {timing('YOLO26x-Seg','inference_share_pct')}% และ postprocess share {timing('YOLO26x-Seg','postprocess_share_pct')}% ส่วน YOLO26n มี postprocess share {timing('YOLO26n-Seg','postprocess_share_pct')}% และ YOLOv8n {timing('YOLOv8n-Seg','postprocess_share_pct')}% ของ pipeline Preprocess + inference + postprocess ตรงกับ pipeline ภายใน tolerance 1e-10 ms จากความละเอียดตัวเลขต้นทาง

**Interpretation:** ใน native-mask pipeline นี้ งานหลัง forward เป็นข้อจำกัดสำคัญเมื่อ checkpoint เล็กลง Stage trends สอดคล้องกับข้อจำกัดดังกล่าว แต่ยังไม่พิสูจน์ว่าขนาดโมเดลหรือ architecture ทำให้ postprocessing ช้าลงโดยตรง และไม่รับรองพฤติกรรมเดียวกันใน deployment pipeline อื่น

RLE preparation ถูกวัดแยก ไม่รวมใน pipeline FPS ส่วน ultralytics_postprocess_inclusive เป็น diagnostic subset ไม่บวกซ้ำ ดู [TIMING_COMPOSITION.csv](metrics/TIMING_COMPOSITION.csv)

## 4. Accuracy ใกล้กันมาก แต่ computational cost ต่างกันมาก

| Model | Mask mAP50-95 | Recall | Inference ms | Pipeline ms | VRAM MiB | Checkpoint MB |
|---|---:|---:|---:|---:|---:|---:|
'''
    for model in ['YOLO26s-Seg','YOLO11x-Seg','YOLOv9e-Seg']:
        body+='| '+model+' | '+' | '.join(v(model,f,n) for f,n in [('mask_map50_95',6),('recall',6),('inference_ms_mean',3),('pipeline_ms_mean',3),('peak_allocated_vram_mib',2),('checkpoint_mb',2)])+' |\n'
    body+=f'''
**Observation:** YOLO26s − YOLO11x มี absolute mAP gap เพียง {near('YOLO11x-Seg','YOLO26s-Seg','map_abs_gap_pp',6)} percentage points แต่ inference {near('YOLO11x-Seg','YOLO26s-Seg','inference_relative_pct')}%, pipeline {near('YOLO11x-Seg','YOLO26s-Seg','pipeline_relative_pct')}% และ parameters {near('YOLO11x-Seg','YOLO26s-Seg','parameters_relative_pct')}% ส่วน YOLO26s − YOLOv9e ใช้ allocated VRAM เพิ่ม {near('YOLOv9e-Seg','YOLO26s-Seg','vram_relative_pct')}%

**Interpretation:** Aggregate mAP is descriptively near-equal under this benchmark แต่ resource requirements ต่างกันมาก และ YOLO26s มี Recall ต่ำกว่าสองรุ่นใหญ่ จึงไม่ใช่ equivalence หรือหลักฐาน architecture superiority

คู่ YOLOv9c/YOLO11m มี mAP ใกล้กันและ pipeline delta เพียง {near('YOLOv9c-Seg','YOLO11m-Seg','pipeline_delta_b_minus_a',3)} ms ส่วน YOLO11s/YOLOv8s มี near-tied mAP แต่ YOLOv8s forward เร็วกว่า ขณะที่ pipeline และ VRAM สูงกว่า ดูทุกคู่และทิศทาง B − A ใน [NEAR_TIE_RESOURCE_DELTAS.csv](metrics/NEAR_TIE_RESOURCE_DELTAS.csv)

## 5. Sequence sensitivity และความสม่ำเสมอของ ranking

![Per-sequence mAP heatmap](plots/14_per_sequence_map_heatmap.png)

**Observation:** YOLO26 นำ mAP ทุก tier × sequence แต่ ranking ของรุ่นรองเปลี่ยน: YOLOv9e สูงกว่า YOLO11x ใน 02, YOLOv9c สูงกว่า YOLOv8l ใน 09 และ YOLOv8s สูงกว่า YOLO11s ใน 02/11 เทียบกับ pooled ordering MOTS20-02 มี measured mAP ต่ำสุดของแต่ละโมเดลครบทั้ง 17 ตัว

เมื่อดู endpoint x → n, YOLO26 เสีย mAP ใน 02 ประมาณ 15.057 pp เทียบกับ 11 ประมาณ 11.480 pp; YOLO11/YOLOv8 ก็เสียมากสุดใน 02 และน้อยสุดใน 11 ส่วน YOLOv9 e → c เสียมากสุดใน 02 และน้อยสุดใน 09

**Interpretation:** Sequence sensitivity ไม่เหมือนกัน และ aggregate ranking ไม่แทน ranking ทุก sequence ยังไม่มีหลักฐานให้ระบุว่าเกิดจาก blur, low light, camera angle หรือ occlusion severity AP รวมเป็น pooled AP ไม่ใช่ค่าเฉลี่ย AP ของสี่ sequences ดู [PER_SEQUENCE_SCALING_ANALYSIS.csv](metrics/PER_SEQUENCE_SCALING_ANALYSIS.csv)

## 6. Scaling cliff เมื่อ s → n

| Family | Δ mAP (pp) | Δ inference (%) | Δ pipeline (%) | Δ VRAM (%) |
|---|---:|---:|---:|---:|
'''
    for family in ['YOLO26','YOLO11','YOLOv8']:
        model=family+'n-Seg';body+='| '+family+' | '+delta(family,model,'mask_map50_95','delta_percentage_points',3)+' | '+delta(family,model,'inference_ms_mean')+' | '+delta(family,model,'pipeline_ms_mean')+' | '+delta(family,model,'peak_allocated_vram_mib')+' |\n'
    body+='''
**Observation:** s → n เป็น adjacent mAP drop มากสุดของทั้งสามตระกูล Forward ลดลง แต่ pipeline เพิ่มทั้งสาม ส่วน VRAM เพิ่มใน YOLO26/YOLO11 และลดใน YOLOv8

**Interpretation:** Nano เป็น candidate เมื่อ forward/ขนาด checkpoint เป็นข้อจำกัดเฉพาะ ไม่เลือกจากชื่อขนาดเพียงอย่างเดียว และไม่เรียกการลดขนาดนี้ว่าคุ้มทุก objective

## 7. สิ่งที่ภาพจริงบอก แต่ aggregate metric ไม่บอก

**Observation:** รายงาน tier พบทั้ง TP เท่ากันแต่ match GT คนละชุด, extra masks บน GT ที่มีคู่แล้ว, common FN และกรณีที่ accuracy leader ไม่ได้ match มากสุด รายงาน x/l ยังแยก candidate confidence ต่ำกว่า threshold ออกจาก mask overlap ที่ไม่ผ่านเกณฑ์

**Interpretation:** FP คือ unmatched prediction ตาม policy ไม่ใช่คนที่ไม่มีจริงโดยอัตโนมัติ FN คือ unmatched GT และอาจมี mask อยู่แล้ว Selected frames เป็น diagnostic ไม่ใช่ representative sample หรือหลักฐาน statistical equivalence

'''
    body+='\n'.join(f'- [{human} qualitative research](https://github.com/folklazy/YOLO_{slug}_Seg_MOTS20_Benchmark/blob/main/PRESENTATION_SUMMARY_TH.md)' for _,slug,human in TIERS)
    body+='''

- [YOLO26x → l detailed visual comparison](YOLO26X_VS_YOLO26L_VISUAL_TH.md)
- [YOLO26 scaling ladder](YOLO26_SCALING_VISUAL_TH.md)
- [26s / 11x / v9e near-tie visual comparison](NEAR_TIE_26S_11X_V9E_VISUAL_TH.md)

## 8. Accuracy vs Inference vs Pipeline vs Memory

**Observation:** Accuracy–inference Pareto มีเก้าโมเดล ส่วน accuracy–pipeline และ accuracy–allocated VRAM มีสองโมเดลคือ YOLO26x/YOLO26l

**Interpretation:** การเลือก objective เปลี่ยน candidate set งาน forward-budget อาจเลือก Small/Nano ได้ ขณะที่ native-mask pipeline-budget ให้น้ำหนัก l มากขึ้น ภาพ segmentation วัด latency/VRAM ไม่ได้ และไม่มี weighted overall score

[Inference Pareto](metrics/PARETO_FRONTIER.csv) · [Pipeline Pareto](metrics/PARETO_PIPELINE.csv) · [VRAM Pareto](metrics/PARETO_VRAM.csv)

## 9. Candidate ตามข้อจำกัด

'''+candidate_table()+'''

## 10. สิ่งที่ผลนี้ยังบอกไม่ได้

- Architecture causality: capacity, checkpoint และ pretraining ต่างกัน
- Statistical significance หรือ equivalence: ยังไม่มี valid test และเฟรมวิดีโอสัมพันธ์กัน
- CCTV robustness, deployment readiness หรือ robustness ต่อ blur/low-light/มุมกล้อง/ระดับ occlusion
- Tracking performance: การศึกษานี้เป็น frame-level instance segmentation
- Total deployment GPU memory: วัด peak allocated ภายใต้ benchmark นี้
- End-to-end saved-mask FPS: pipeline ไม่รวม decode, RLE และการเขียนผล
- ความถี่ failure จาก selected cases: ภาพเชิงคุณภาพไม่แทน dataset-level metrics

## 11. ขั้นตอนถัดไป

แยก future robustness/deployment study ออกจากผล MOTS20 ปัจจุบัน หากต้องการ human parsing ให้กำหนด output labels, crop policy และวิธีประเมินผลทั้งระบบก่อน โดยยังถือเป็นแผน ไม่ใช่ผลทดลองที่เสร็จแล้ว เอกสารรอบนี้ไม่เริ่ม inference หรือ experiment ใหม่

[Technical source of truth](MASTER_RESULTS.md) · [Derived analysis provenance](provenance/POST_STUDY_ANALYSIS.json) · [Revision provenance](provenance/POST_STUDY_REVISION.json)
'''
    save(CURRENT,body)
    CURRENT='MEETING_SUMMARY_TH.md'
    meeting=f'''# YOLO Instance Segmentation Scaling Study — MOTS20

## 1. เราทดสอบอะไร

| รายการ | ขอบเขต |
|---|---|
| Models | 17 checkpoints: YOLO26/YOLO11/YOLOv8 อย่างละ 5, YOLOv9 2 |
| Tiers | Largest, Second-largest, Medium, Small, Nano |
| Dataset | MOTS20 2,862 frames / 26,894 Person GT instances รายเฟรม |
| เงื่อนไขร่วม | Pretrained, ไม่มี fine-tuning; protocol เดียวกัน; Tesla T4, FP32, imgsz640, batch1 |
| Status | ทั้งห้า tiers และ Master COMPLETE; post-study ใช้ artifacts เดิม |

## 2. ภาพรวมผล 17 โมเดล

![Master accuracy](plots/01_master_map_by_model.png)

ภาพรวมต้องดูหลาย objectives: YOLO26x นำ mAP, YOLOv8n forward เร็วสุด และ YOLO26l มี pipeline/allocated VRAM ต่ำสุดในค่าที่วัด ไม่มีโมเดลเดียวชนะทุกข้อจำกัด

## 3. ใครนำด้าน Accuracy

**Observation:** YOLO26 นำ pooled mAP/AP75 ทั้งห้า tiers และ sequence mAP ครบ 20 comparisons แต่ Nano Recall winner เป็น YOLOv8n

**Interpretation:** ผล accuracy สม่ำเสมอในข้อมูลชุดนี้ แต่ไม่ได้ควบคุม parameters/pretraining ให้เท่ากัน และไม่รับรองผู้ชนะทุกเฟรม

## 4. ลดขนาดแล้วได้อะไร เสียอะไร

![Family accuracy scaling](plots/02_family_accuracy_scaling.png)

**Observation:** ทุก adjacent size reduction ลด mAP และ inference ส่วน pipeline เพิ่มใน 8 จาก 13 คู่ และ allocated VRAM เพิ่มใน 7 คู่ s → n เป็น adjacent accuracy drop มากสุดของทั้ง YOLO26/YOLO11/YOLOv8

**Interpretation:** ต้องถามว่าจะลดต้นทุนด้านใด ชื่อขนาดหรือ parameters อย่างเดียวตอบเรื่อง throughput/memory ไม่ได้

## 5. จุดที่น่าสนใจที่สุด: x → l

**Observation:** YOLO26x → l ลด mAP {delta('YOLO26','YOLO26l-Seg','mask_map50_95','delta_percentage_points',3)} pp แต่ inference {delta('YOLO26','YOLO26l-Seg','inference_ms_mean')}%, pipeline {delta('YOLO26','YOLO26l-Seg','pipeline_ms_mean')}% และ VRAM {delta('YOLO26','YOLO26l-Seg','peak_allocated_vram_mib')}% Pipeline Pareto เหลือ x/l

**Interpretation:** l เป็น candidate efficiency point ภายใต้ objectives ที่วัด ส่วน x ยังคงเป็นตัวเลือกเมื่อ accuracy มีน้ำหนักมากกว่า

ใน [รายงานภาพ x/l](YOLO26X_VS_YOLO26L_VISUAL_TH.md) มีทั้งภาพที่ x เก็บเพิ่ม ภาพที่ l เก็บเพิ่ม และกรณี mask คุณภาพต่างกันแม้ coverage เท่ากัน ควรใช้ประกอบการตัดสินโดยไม่ยกภาพเดียวเป็นข้อสรุปทั้ง dataset

## 6. Inference เร็ว ไม่ได้แปลว่า Pipeline เร็ว

![Timing composition](plots/13_timing_stage_composition.png)

**Observation:** Postprocessing มากกว่าครึ่ง pipeline ใน 12 โมเดล YOLOv8n ใช้ inference {v('YOLOv8n-Seg','inference_ms_mean')} ms แต่ pipeline {v('YOLOv8n-Seg','pipeline_ms_mean')} ms; YOLO26l pipeline {v('YOLO26l-Seg','pipeline_ms_mean')} ms

**Interpretation:** Forward เร็วขึ้นยังติดต้นทุน native-mask postprocessing ได้ แต่ stage trends ไม่พิสูจน์สาเหตุเชิง architecture RLE preparation อยู่แยกและไม่รวมใน FPS; diagnostic inclusive stage ไม่บวกซ้ำ

## 7. Accuracy แทบเท่ากัน แต่ resource ต่างกันมาก

YOLO26s/YOLO11x/YOLOv9e มี aggregate mAP near-equal ภายใต้ screen ≤0.001 โดย YOLO26s − YOLO11x มี inference {near('YOLO11x-Seg','YOLO26s-Seg','inference_relative_pct')}% และ pipeline {near('YOLO11x-Seg','YOLO26s-Seg','pipeline_relative_pct')}% แต่ Recall ของ Small ต่ำกว่าสองรุ่นใหญ่ และ VRAM ของ 26s สูงกว่า v9e เล็กน้อย

**Interpretation:** Near tie ไม่ใช่ equivalence หรือ prediction เหมือนกัน ดู [same-frame near-tie visual research](NEAR_TIE_26S_11X_V9E_VISUAL_TH.md) เพื่อเห็น GT coverage และ unmatched outputs ต่างกัน และ [resource deltas](metrics/NEAR_TIE_RESOURCE_DELTAS.csv) เพื่อดูทิศทาง B − A

## 8. ผลเหมือนกันทุก sequence หรือไม่

![Sequence heatmap](plots/14_per_sequence_map_heatmap.png)

**Observation:** ตัวนำ YOLO26 คงที่ทุก tier × sequence แต่รุ่นรองมี ranking inversion ใน Largest/Second-largest/Small MOTS20-02 มี mAP ต่ำสุดของทุกโมเดล

**Interpretation:** ค่า pooled ไม่แทนทุก sequence และยังไม่ระบุสาเหตุจาก blur/แสง/มุมกล้อง AP รวมไม่ใช่ค่าเฉลี่ยของ sequence AP

## 9. ถ้าต้องเลือกโมเดล

'''+candidate_table()+'''

## 10. ข้อจำกัด

ไม่มี statistical significance test; เฟรมต่อเนื่องสัมพันธ์กันและภาพ diagnostic ไม่ใช่ representative sample TP-only quality ใช้ matched subsets; FP/FN เป็นผลตาม evaluator ไม่ใช่ absence โดยอัตโนมัติ Pipeline FPS ไม่รวม decode/RLE/เขียนผล และ VRAM เป็น allocated peak ไม่ใช่ total deployment memory ผลนี้ยังไม่วัด tracking หรือยืนยัน CCTV robustness และไม่แยก architecture causality

## 11. ขั้นตอนถัดไป

กำหนดข้อจำกัดของงานจริง แล้วทดสอบ candidates บน future robustness/deployment study แยกต่างหาก หากจะใช้ parsing ให้กำหนด output labels, crop policy และผลทั้งระบบก่อน รอบนี้เป็นการสังเคราะห์ผลเดิมและ STOP

อ่านบทวิเคราะห์เต็มที่ [RESEARCH_INSIGHTS_TH.md](RESEARCH_INSIGHTS_TH.md) และ [MASTER_RESULTS.md](MASTER_RESULTS.md)
'''
    save(CURRENT,meeting)
    CURRENT='MASTER_RESULTS.md';technical=original(CURRENT)
    technical+='\n\n'+f'''## 13. Accuracy / Pipeline Pareto

DERIVED FROM EXISTING CANONICAL RESULTS. Maximize unrounded mask_map50_95 and minimize pipeline_ms_mean. Strict two-objective dominance retains equal points. [PARETO_PIPELINE.csv](metrics/PARETO_PIPELINE.csv) includes all 17 models with explicit membership flags; existing Pareto CSVs are unchanged.

**Observation:** Only YOLO26x-Seg and YOLO26l-Seg are nondominated. **Interpretation:** l is a candidate efficiency point under the measured objectives, not a universal optimum. Inference Pareto retains nine candidates and allocated-VRAM Pareto retains two; no weighted score is used.

![Accuracy versus pipeline](plots/11_accuracy_vs_pipeline.png)
![Pipeline Pareto](plots/12_pareto_accuracy_pipeline.png)

See the [detailed saved-prediction same-frame x/l visual comparison](YOLO26X_VS_YOLO26L_VISUAL_TH.md), which supports interpretation of this trade-off and retains counterexamples/common failures.

## 14. Timing Stage Composition

[TIMING_COMPOSITION.csv](metrics/TIMING_COMPOSITION.csv) retains original means for preprocessing, inference, postprocessing, pipeline, RLE preparation and ultralytics_postprocess_inclusive; stage shares are derived. The three included component means sum to pipeline within 1e-10 ms (decimal-string rounding tolerance).

**Observation:** Postprocessing exceeds half of pipeline in 12 of 17 models. YOLO26x postprocessing share is {timing('YOLO26x-Seg','postprocess_share_pct')}%; YOLO26n is {timing('YOLO26n-Seg','postprocess_share_pct')}%; YOLOv8n is {timing('YOLOv8n-Seg','postprocess_share_pct')}%. **Interpretation:** forward savings can be outweighed by the measured native-mask postprocessing workload. These trends do not establish architecture causality or generalize to every deployment pipeline.

![Stage composition](plots/13_timing_stage_composition.png)

RLE preparation is separate and excluded from pipeline FPS. Ultralytics-inclusive postprocess is a diagnostic subset, not an extra component. No timing was rerun.

## 15. Per-Sequence Scaling and Consistency

[PER_SEQUENCE_SCALING_ANALYSIS.csv](metrics/PER_SEQUENCE_SCALING_ANALYSIS.csv) preserves source mAP/Recall, ranks models within each tier/sequence, compares these ranks with pooled tier ranks, and records within-family endpoint and adjacent deltas. Endpoint deltas use smaller minus larger and are repeated as family/sequence context, not independent observations.

**Observation:** YOLO26 wins mAP in all 20 tier-by-sequence comparisons. Secondary rankings invert for 11x/v9e on 02, v9c/v8l on 09, and 11s/v8s on 02/11. Sequence 02 has the lowest mAP for every tested model and the largest endpoint mAP loss in all four families. x-to-n loss is smallest on 11 for YOLO26/YOLO11/YOLOv8; e-to-c loss is smallest on 09 for YOLOv9.

**Interpretation:** aggregate rankings can hide sequence sensitivity. No causal scene label is assigned. Pooled AP is not a mean of sequence AP; no predictions were recomputed.

![Per-sequence accuracy](plots/14_per_sequence_map_heatmap.png)

## 16. Near-Tie Resource Deltas

[NEAR_TIE_RESOURCE_DELTAS.csv](metrics/NEAR_TIE_RESOURCE_DELTAS.csv) extends the unchanged descriptive screen |delta mAP| <= 0.001. Direction is B minus A and relative cost change uses A as denominator. It reports absolute mAP gap/percentage points and inference, pipeline, allocated VRAM, parameters and checkpoint deltas.

**Observation:** YOLO26s minus YOLO11x changes inference by {near('YOLO11x-Seg','YOLO26s-Seg','inference_relative_pct')}%, pipeline by {near('YOLO11x-Seg','YOLO26s-Seg','pipeline_relative_pct')}%, parameters by {near('YOLO11x-Seg','YOLO26s-Seg','parameters_relative_pct')}% and allocated VRAM by {near('YOLO11x-Seg','YOLO26s-Seg','vram_relative_pct')}%. Relative to YOLOv9e, 26s allocated VRAM changes by {near('YOLOv9e-Seg','YOLO26s-Seg','vram_relative_pct')}%, despite much lower forward latency.

**Interpretation:** aggregate mAP is descriptively near-equal under this benchmark while resource requirements differ substantially. Recall and per-frame errors differ; this is not equivalence or architecture superiority. See the [same-frame near-tie diagnostics](NEAR_TIE_26S_11X_V9E_VISUAL_TH.md) and [YOLO26 scaling ladder](YOLO26_SCALING_VISUAL_TH.md).

[Post-study revision provenance](provenance/POST_STUDY_REVISION.json) records archives and immutable before/after measurements. Original synthesis provenance remains preserved.
'''
    save(CURRENT,technical)
    CURRENT='EXECUTIVE_SUMMARY_TH.md';executive=original(CURRENT)
    executive+='\n\n'+f'''## ข้อมูลเสริมจาก post-study analysis

Accuracy–pipeline Pareto เหลือ YOLO26x/YOLO26l โดย l เป็น candidate efficiency point ภายใต้ objectives ที่วัด Postprocessing มากกว่าครึ่ง pipeline ใน 12 จาก 17 โมเดล จึงไม่ใช้ forward latency แทน throughput; RLE ยังแยกจาก pipeline FPS รายงาน [ภาพ x/l](YOLO26X_VS_YOLO26L_VISUAL_TH.md) เก็บทั้งข้อได้เปรียบและ counterexamples ส่วน [Research Insights](RESEARCH_INSIGHTS_TH.md) อธิบาย sequence sensitivity และ near-tie resource trade-offs ตัวเลข benchmark เดิมไม่เปลี่ยนและยังไม่เริ่มการทดลองใหม่
'''
    save(CURRENT,executive)
    CURRENT='README.md';landing=original(CURRENT)
    landing=landing.replace('## Ten Master plots','## Original ten Master plots')
    landing=landing.replace('## Reproduce and validate','''## Visual Research Analyses

- [YOLO26x versus YOLO26l](YOLO26X_VS_YOLO26L_VISUAL_TH.md): detailed saved-prediction same-frame evidence for the x-to-l trade-off.
- [YOLO26 scaling ladder](YOLO26_SCALING_VISUAL_TH.md): x/l/m/s/n diagnostic comparisons, including a control and common failure.
- [Near-tied 26s / 11x / v9e](NEAR_TIE_26S_11X_V9E_VISUAL_TH.md): same-frame coverage and extra-output differences despite near-equal aggregate mAP.
'''+ '\n'.join(f'- [{human} full qualitative research report](https://github.com/folklazy/YOLO_{slug}_Seg_MOTS20_Benchmark/blob/main/PRESENTATION_SUMMARY_TH.md)' for _,slug,human in TIERS)+'''

## Post-study derived analyses

Derived from existing canonical results; no inference or timing rerun. Original measured CSVs, synthesis provenance and ten plots remain unchanged.

- [Pipeline Pareto](metrics/PARETO_PIPELINE.csv): all models with membership flags; [scatter](plots/11_accuracy_vs_pipeline.png), [frontier](plots/12_pareto_accuracy_pipeline.png).
- [Timing composition](metrics/TIMING_COMPOSITION.csv): original stage means and derived shares; [stacked plot](plots/13_timing_stage_composition.png).
- [Per-sequence scaling](metrics/PER_SEQUENCE_SCALING_ANALYSIS.csv): source values, ranks and family deltas; [heatmap](plots/14_per_sequence_map_heatmap.png).
- [Near-tie resource deltas](metrics/NEAR_TIE_RESOURCE_DELTAS.csv): explicit B-minus-A direction, absolute and relative costs.
- [Derived data interface](POST_STUDY_DATA_SCHEMA.md): fields, units, ranks and direction conventions.
- [Revision provenance](provenance/POST_STUDY_REVISION.json) and [revision validation](provenance/POST_STUDY_VALIDATION.json): archives, protected hashes and checks.

## Reproduce and validate''')
    landing=landing.replace('scripts/validate_master.py\n','scripts/validate_master.py\n.venv/bin/python -B YOLO_Instance_Segmentation_MOTS20_Scaling_Study/scripts/validate_post_study.py\n')
    landing=landing.replace('Existing tier quantitative summaries and visual qualitative analyses retain their distinct roles.','Tier RESULTS summaries are compact numerical quick summaries; PRESENTATION documents remain full visual qualitative research reports.')
    landing=landing.replace('this synthesis does not reselect frames or inspect raw predictions.','the original CSV-only synthesis did not inspect raw predictions. The separately authorized post-study visual extension reads only selected saved artifacts, as declared in revision provenance.')
    landing=landing.replace('The initial build uses scripts/build_master.py, plots use scripts/plot_master.py, and documents use scripts/report_master.py.','The original synthesis used scripts/build_master.py, scripts/plot_master.py and scripts/report_master.py. Preserve its existing outputs. Post-study analysis/report generation uses scripts/post_study_analysis.py and scripts/report_post_study.py; do not use the original report generator to replace the revised layout.')
    save(CURRENT,landing)
    (MASTER/'provenance/POST_STUDY_DOCUMENT_VALUE_CLAIMS.json').write_text(json.dumps({'scope':'Derived post-study report display claims; original synthesis claims remain archived in their original context','claims':CLAIMS},ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    with localcontext() as ctx:
        ctx.prec=50;tier_summaries();master_reports()
    print('[POST-STUDY] five compact tier summaries and Master insight/narrative updates generated')
