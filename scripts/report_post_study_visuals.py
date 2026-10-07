"""Reports for visually inspected saved-prediction cases; no inference."""
from pathlib import Path
import json
MASTER=Path(__file__).resolve().parents[1]

LADDER=[
 ('Coverage คล้ายกัน ไม่ได้แปลว่า mask ทุกขนาดเหมือนกัน',
  ['ทุกขนาด match valid GT ครบ 6 instances; x/l/m/s ไม่มี FP ส่วน n มี FP หนึ่ง mask ที่ทับบางส่วนของ GT 2001 ซึ่งมีคู่แล้ว',
   'ROI ของคนเสื้อเขียว GT 2019 มี matched mask ทุกโมเดล รูปทรงหลักคล้ายกัน; IoU x/l/m/s/n เป็น 0.895/0.880/0.886/0.877/0.821'],
  'เป็น control ที่แสดงว่าการลดขนาดไม่จำเป็นต้องเสีย coverage ในทุกภาพ และ mask ของคนหลักยังใกล้กัน ขณะเดียวกัน IoU ของ m สูงกว่า l ใน GT นี้ จึงไม่ใช่คุณภาพต่อคนที่ลดลงแบบ monotonic ทุกครั้ง Extra mask ของ n อยู่คนละบริเวณกับ ROI ต้องตรวจภาพเต็มควบคู่',
  'สอดคล้องกับ TP-only overlap ที่ยังค่อนข้างใกล้กันในหลายขนาด แต่ TP-only means ใช้ matched subsets ต่างกัน; ภาพนี้ไม่ยืนยันว่าขนาดต่างกันให้ output เหมือนกันทั้ง dataset'),
 ('Nano พลาดคนเพิ่ม แต่ l/s มี coverage มากกว่า x ในเฟรมเดียวกัน',
  ['x/l/m/s/n มี TP 9/10/9/10/6 และ FP 0/0/1/1/1 ตามลำดับ',
   'GT 2026 ฝั่งซ้ายและ GT 2040 หลังคนใหญ่กลางภาพ match ใน x/l/m/s แต่ไม่ match ใน n ที่ confidence เดิม',
   'l เก็บ GT 2028 ซึ่ง x/m พลาด; s เก็บ GT 2003 ซึ่ง x/l/m พลาด แต่ s ยังพลาด 2028 และมี FP ทับบางส่วนของ GT 2040 ที่มีคู่แล้ว'],
  'ตัวอย่างแสดงทั้ง degradation ของ n และ counterexample ต่อการใช้ขนาดตัดสินทุกเฟรม Coverage มากขึ้นของ s ไม่ได้มาพร้อม output ส่วนเกินที่น้อยกว่าเสมอ FP ของ m ทับ GT 2037 ที่มีคู่แล้ว ส่วน n มี unmatched mask ทับ GT 2027 โดย IoU ประมาณ 0.481 ต่ำกว่าเกณฑ์ ไม่เรียกว่า nonexistent person โดยอัตโนมัติ',
  'สอดคล้องในทิศทางกับ Recall ที่ลดเมื่อ s → n แต่ไม่อธิบายช่องว่างรวมจากเฟรมเดียว การที่ l/s มี TP มากกว่า x ในเฟรมนี้ไม่ขัดกับ pooled mAP/Recall ของ x ที่สูงกว่า'),
 ('ข้อผิดพลาดร่วม และ TP เท่ากันแต่เก็บ GT ต่างชุด',
  ['x/l/m/s มี TP 9 และ FN 4 ส่วน n มี TP 7 และ FN 6; FP เป็น 2/2/1/2/3',
   'GT 2011 ไม่ match ทุกขนาด แม้มี prediction ในบริเวณคนจริง; best candidate IoU ที่ confidence ≥0.25 ยังต่ำกว่า 0.50',
   'GT 2007 match ใน x/l/m แต่ไม่ match ใน s/n; s เก็บ GT 2001 ที่ x/l/m ไม่ match จึงได้ TP เท่ากันจาก GT คนละชุด',
   'GT 2018 match ใน x/l/m/s แต่ n มี best candidate IoU ประมาณ 0.452 และยังเป็น FN'],
  'การดูจำนวน TP รวมปิดบังว่าใครถูกเก็บหรือพลาด ภาพแสดง mask บนบริเวณคนจริงที่ไม่ผ่าน matching พร้อม common failure ของทุกขนาด จึงไม่ใช่เพียงโมเดลใหญ่แก้ปัญหาได้ทั้งหมด และไม่อนุมาน severity หรือสาเหตุเชิง architecture',
  'ตัวอย่างนี้ช่วยตีความ coverage และ overlap แยกกัน แต่ไม่ได้อธิบาย dataset-wide AP75/Recall หรือความถี่ของ failure และไม่ได้ใช้ภาพอนุมาน latency/VRAM')]

NEAR=[
 ('ผลหลักคล้ายกันใน control',
  ['ทั้งสาม match valid GT ครบ 6 instances ไม่มี FP/FN',
   'GT 2019 คนเสื้อเขียวด้านขวาถูก match ทุกโมเดลและมีรูปทรง mask ใกล้กัน; IoU ของ 26s/11x/v9e เป็น 0.877/0.865/0.863'],
  'มีภาพที่ near-tied checkpoints ให้ coverage เหมือนกันและ mask ของคนหลักคล้ายกัน แต่ IoU ของ 26s สูงกว่าสองรุ่นใหญ่สำหรับ GT นี้ ไม่ใช้ control นี้พิสูจน์ว่า output เหมือนกันทั้งหมด',
  'Aggregate mAP ใกล้กันอาจอยู่ร่วมกับผลคล้ายกันบางเฟรมได้ แต่ไม่ใช่ equivalence หรือการทดสอบนัยสำคัญ'),
 ('รุ่นใหญ่เก็บ GT เพิ่ม และมี failure ร่วมบนคนจริง',
  ['26s/11x/v9e มี TP 5/6/7, FP 2/1/1 และ FN 5/4/3',
   'GT 2004 หลังคนเสื้อฟ้าทางซ้าย match เฉพาะ v9e; 11x มี saved candidate IoU ประมาณ 0.564 แต่ confidence 0.217 ต่ำกว่า 0.25 ส่วน v9e ผ่านด้วย confidence ประมาณ 0.254',
   'GT 2003 ทางขวาไม่ match ทั้งสาม แต่มี red unmatched mask ในบริเวณนั้น; best IoU ที่ confidence เดิมเป็น 0.350/0.342/0.437'],
  'v9e มี coverage มากกว่าในเฟรมนี้ แต่ GT 2003 เป็น failure ร่วม แม้สร้าง mask ได้ การพลาด GT 2004 ของ 11x เกี่ยวข้องกับ operating point ด้วย ไม่ใช่หลักฐานว่าไม่มี candidate mask เลย และไม่รับประกันว่าลด threshold แล้วจะเป็น TP โดยไม่มี FP เพิ่ม',
  'สอดคล้องในทิศทางกับ Recall รวมของ 26s ที่ต่ำกว่าสองรุ่นใหญ่ แต่ภาพเดียวไม่อธิบายความต่างทั้งหมด และ mAP ใช้ confidence ranking ต่างจาก fixed-confidence counts ในภาพ'),
 ('Coverage เท่ากัน แต่ extra masks ต่างกันชัด',
  ['ทั้งสาม match valid GT ครบ 9 instances ไม่มี FN; 26s/11x ไม่มี FP ส่วน v9e มี FP สอง masks',
   'ROI ของ GT 2010 แสดง red mask เพิ่มใน v9e บนคนที่มี matched prediction อยู่แล้ว FP นี้มี IoU ประมาณ 0.648 กับ GT แต่ one-to-one matching ไม่ให้เป็น TP เพิ่ม',
   'ภาพเต็มแสดง FP อีก mask ของ v9e ที่ราวด้านขวา ซึ่งไม่ทับ valid GT; ROI ไม่ได้แสดงข้อผิดพลาดนอกบริเวณทั้งหมด'],
  'Near-tied aggregate score ซ่อนภาระ extra-output ได้ GT 2010 ของ v9e มี matched IoU ประมาณ 0.576 ซึ่งต่ำกว่า IoU ของ extra mask เพราะ evaluator จับตาม confidence ก่อน ต้องรักษา policy เดิม ไม่สลับคู่ย้อนหลังเพื่อทำให้ผลดูดี FP บนราวไม่ยืนยันว่าไม่มีคนจริงจากภาพเดียว',
  'เป็นตัวอย่างที่ 26s/11x มี output ส่วนเกินน้อยกว่า v9e ขณะที่ Case 2 ให้ v9e มี coverage มากกว่า จึงไม่ถือว่าใครชนะทุกกรณี หรือสรุป frequency จาก case selection'),
 ('TP เท่ากัน แต่ GT คนละชุด และ v9e มี FP น้อยกว่า',
  ['26s/11x มี TP 9, FP 2, FN 4 เท่ากัน แต่ 26s เก็บ GT 2001 ที่ 11x พลาด และ 11x เก็บ GT 2007 ที่ 26s พลาด',
   'v9e มี TP 8, FP 1, FN 5; GT 2007 ไม่ match แต่มี saved candidate IoU ประมาณ 0.707 ที่ confidence 0.243 ต่ำกว่า 0.25',
   'GT 2011 ไม่ match ทั้งสาม และมี unmatched prediction บนบริเวณคนจริง โดย best IoU ยังต่ำกว่า 0.50'],
  'เป็น counterexample ต่อ Case 2: v9e มี matched coverage น้อยกว่า แต่ FP น้อยกว่า อีกสองโมเดลมี counts เท่ากันโดยไม่ได้เก็บคนชุดเดียวกัน ส่วนของ 26s บน GT 2007 มี candidate ที่ confidence ต่ำมากใน saved predictions จึงไม่ควรตีความ FN ว่าไม่มี mask เลย',
  'Near-equal mAP ไม่ระบุว่า error เกิดที่คนใด ผลนี้จึงใช้สนับสนุนการดู GT-level diagnostics ร่วมกับ Recall/Precision รวม โดยไม่พิสูจน์ statistical equivalence')]


def report(slug,name,title,texts,ladder):
    root=MASTER/f'outputs/visualizations/{slug}/v1';e=json.loads((root/'EVIDENCE.json').read_text())
    lines=[f'# {title}','', '**สถานะ: diagnostic qualitative analysis จาก saved predictions เท่านั้น; ไม่มี inference หรือ benchmark ใหม่**','',
           '## ภาพรวมและวิธีอ่านหลักฐาน','',
           'ใช้ original MOTS20/GT และ saved lossless RLE จาก benchmark ที่เสร็จแล้ว ทุก panel ใน case ใช้เฟรมเดียวกัน confidence ≥0.25, one-to-one mask matching IoU ≥0.50 และ ignore IoA ≥0.50 ตาม policy เดิม สีเขียวเป็น GT, สีชมพูเป็น matched prediction, สีแดงเป็น FP และสีเทาเป็น ignored prediction ซึ่งไม่ถูกนับ FP','',
           ('Panel เรียงซ้ายไปขวา: Original/GT, x, l ในแถวแรก; m, s, n ในแถวที่สอง' if ladder else 'Panel เรียงซ้ายไปขวา: Original/GT, YOLO26s ในแถวแรก; YOLO11x, YOLOv9e ในแถวที่สอง'),' ',
           'Full-frame อยู่ทุก case และ ROI เป็นภาพเสริมที่ใช้พิกัด/scale เดียวกันทุกโมเดล การขยายไม่เพิ่มรายละเอียดต้นฉบับ TP/FP/FN ด้านล่างเป็น counts ของเต็มเฟรม ไม่ใช่เฉพาะ ROI FN อาจมี prediction อยู่แล้วแต่ไม่ผ่าน confidence/matching; FP เป็น unmatched prediction ไม่จำเป็นต้องเป็นคนที่ไม่มีจริง','',
           'เลือกจาก existing diagnostic pool และ per-frame CSV ให้มี control, ความต่าง, counterexample และ common failure ไม่ใช่ representative sample หรือ full-dataset extrema เฟรมที่ซ้ำกับรายงาน tier/รายงานอื่นไม่เพิ่มจำนวน independent samples','',
           f'[Case selection](outputs/visualizations/{slug}/v1/CASE_SELECTION.md) · [Evidence, source hashes and ROI](outputs/visualizations/{slug}/v1/EVIDENCE.json)','']
    for c,(case_title,observations,analysis,connection) in zip(e['cases'],texts):
        lines += [f"## Case {c['case']} — {case_title}",'',f"**เฟรม:** {c['sequence']} / {c['frame']:06d}",'',f"**เหตุผลที่เลือก:** {c['selection_reason']}",'',
                  '### ภาพเปรียบเทียบ','',f"![Case {c['case']} full frame]({c['comparison_path']})",'',
                  '### ภาพขยาย','',f"![Case {c['case']} identical ROI]({c['focus_path']})",'',f"ROI xyxy: `{c['roi_xyxy']}`; ภาพเต็มด้านบนยังคงแสดง error นอก ROI",'',
                  '### สิ่งที่เห็นจากภาพ','']+['- '+x for x in observations]+['',
                  '| Model | TP | FP | FN |','|---|---:|---:|---:|']
        lines += [f"| {m['model']} | {m['counts']['tp']} | {m['counts']['fp']} | {m['counts']['fn']} |" for m in c['models']]
        lines += ['', 'GT-level diagnostics จาก saved masks; TP เป็น matched IoU ส่วน FN เป็น best candidate IoU ที่ confidence เดิม:', '',
                  '| GT | '+' | '.join(m['model'] for m in c['models'])+' |','|---|'+'---|'*len(c['models'])]
        for gid in c['gt_ids']:
            values=[]
            for m in c['models']:
                t=next(t for t in m['targets'] if t['gt_id']==gid)
                values.append(f"TP IoU {t['matched_iou']:.3f}" if t['matched'] else f"FN; best IoU {t['best_at_conf025']['iou']:.3f}")
            lines.append('| '+str(gid)+' | '+' | '.join(values)+' |')
        lines += ['', '### วิเคราะห์','',analysis,'','### เชื่อมกับผลเชิงตัวเลข','',connection,'',
                  '### ขอบเขตของหลักฐาน','','ผลเฉพาะเฟรมนี้ไม่แทน dataset-level metrics หรือความถี่ของ failure Candidate IoU ต่ำกว่า threshold เป็น diagnostic overlap ไม่รับประกันว่าจะกลายเป็น TP หากเปลี่ยน confidence และยังต้องใช้ one-to-one/ignore policy เดิม','']
    failures=([('Extra mask บน GT ที่มีคู่แล้ว','n; m/s','1/2','counts และ matched GT ไม่แทนจำนวน outputs'),
               ('GT ไม่ผ่าน matching เมื่อขนาดเล็กลง','n; s/n','2/3','ข้อได้เปรียบของขนาดใหญ่มี counterexample ด้วย'),
               ('Mask บนคนจริงแต่ IoU ไม่ผ่าน','ทุกขนาด','3','FP/FN อาจเกิดบริเวณเดียวกัน')]
              if ladder else [('GT coverage ต่างกัน','26s/11x/v9e','2/4','near-equal mAP ไม่ใช่ GT ชุดเดียวกัน'),
                              ('Extra masks บน GT ที่มีคู่แล้ว/ไม่ทับ valid GT','v9e','3','one-to-one matching; ไม่ยืนยัน nonexistent person'),
                              ('Common unmatched GT มี prediction ในบริเวณนั้น','ทุกโมเดล','2/4','mask presence ไม่รับประกัน IoU ผ่าน')])
    lines += ['## Failure Analysis','','| Failure pattern | Models observed | Case | Interpretation |','|---|---|---|---|']+['| '+' | '.join(x)+' |' for x in failures]
    lines += ['', '## สิ่งที่เรียนรู้จากภาพจริง','',
              '**Observation:** มี control ที่ coverage เท่ากัน แต่กรณีอื่นมีทั้ง GT ชุดต่างกันและ extra outputs ต่างกัน','',
              '**Interpretation:** การเลือกต้องดูชนิดและตำแหน่งของ error ร่วมกับ pooled metrics ไม่สรุปจาก counts หรือภาพคนใหญ่เพียงอย่างเดียว','',
              '**Observation:** ตัวอย่างมีทั้งผลที่สอดคล้องและสวนอันดับรวม รวมถึง common failures','',
              '**Interpretation:** ขนาด/คะแนนรวมไม่รับรองคุณภาพต่อ GT ทุกคน และกรณีที่เลือกไม่วัดความถี่หรือ causal architecture effect','',
              '## เมื่อดูทั้งตัวเลขและภาพร่วมกัน','',
              ('YOLO26 pooled mAP/Recall ลดลงเมื่อ x → n แต่ Case 2 ให้ l/s มี TP มากกว่า x และ Case 3 ให้ TP เท่ากันจาก GT คนละชุด จึงไม่ใช้คำว่า mask ทุกคนลดคุณภาพตามขนาดแบบ monotonic TP-only IoU/Dice ใช้ matched subsets ที่อาจต่างกัน' if ladder else
               'YOLO26s/YOLO11x/YOLOv9e ผ่าน descriptive near-tie screen |ΔmAP| ≤0.001 แต่ Recall ของ 26s ต่ำกว่า และภาพมี error trade-offs ต่างกัน Case 2 ให้ v9e มี coverage มากกว่า ส่วน Case 3 ให้ 26s/11x มี FP น้อยกว่า ไม่เรียกว่า equivalence'),'',
              'Latency และ VRAM เป็น system measurements จาก benchmark ภาพ segmentation ไม่วัดสองค่านี้ Pipeline FPS ไม่รวม decode, RLE preparation และการเขียนผล','',
              '## ใช้ประกอบการเลือกอย่างไร','',
              'หากเน้นความครบถ้วน ให้ตรวจ GT coverage และ Recall รวม หากเน้น mask extraction ให้ตรวจ foreground loss/พื้นที่ติดเกินและ diagnostics ของคนเดียวกัน ส่วนข้อจำกัด speed/memory อ่านจาก CSV ภาพเหล่านี้ไม่ประกาศผู้ชนะทุกด้านหรือ final CCTV superiority','',
              '## ข้อจำกัด','',
              '- เฟรมที่เลือกเป็น diagnostic examples ไม่ใช่ random representative sample และเฟรมวิดีโอสัมพันธ์กัน',
              '- Overlay และภาพย่ออาจซ่อนรายละเอียดพิกเซล; ค่าราย GT ช่วยตรวจ overlap ไม่ใช่ boundary metric โดยตรง',
              '- ไม่มี statistical significance/equivalence test หรือการวัด tracking',
              '- ไม่อนุมาน blur/low-light/มุมกล้อง/occlusion robustness หรือ deployment readiness',
              '- Capacity/checkpoints/pretraining ต่างกันใน cross-generation comparison; ไม่พิสูจน์ architecture causality','',
              '## รายละเอียดเพิ่มเติม','',
              '[Canonical results](metrics/MASTER_17_MODELS.csv) · [Research insights](RESEARCH_INSIGHTS_TH.md) · [Technical record](MASTER_RESULTS.md) · [x/l visual study](YOLO26X_VS_YOLO26L_VISUAL_TH.md)','']
    path=MASTER/name;assert not path.exists();path.write_text('\n'.join(line.rstrip() for line in lines))


if __name__=='__main__':
    report('yolo26_scaling_ladder','YOLO26_SCALING_VISUAL_TH.md','YOLO26 x → n — Same-Frame Scaling Visual Analysis',LADDER,True)
    report('near_tie_26s_11x_v9e','NEAR_TIE_26S_11X_V9E_VISUAL_TH.md','YOLO26s / YOLO11x / YOLOv9e — Near-Tie Visual Analysis',NEAR,False)
    print('[POST-STUDY] visually inspected ladder (3) and near-tie (4) cases documented')
