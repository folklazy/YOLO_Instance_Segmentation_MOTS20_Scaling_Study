# สรุปผล {{TIER}} YOLO Instance Segmentation

## สรุปใน 1 นาที

<!-- ใช้ 5–8 ข้อ: สมาชิกโมเดล, 2,862 เฟรม, GT 26,894 รายเฟรม, pretrained ไม่ปรับจูน, ผู้ชนะและข้อแลกเปลี่ยนหลัก -->
{{OVERVIEW_BULLETS}}

## ผลลัพธ์หลัก

| โมเดล | Mask mAP50-95 | AP75 | Recall | Inference (ms) | Pipeline (ms) | FPS | Peak allocated VRAM (MiB) |
| --- | --- | --- | --- | --- | --- | --- | --- |

## สรุปผลจากตาราง

<!-- หนึ่งหัวข้อย่อยต่อโมเดลเรียงตามตระกูล ใช้ 1–2 ย่อหน้าสั้นอธิบายจุดเด่น ข้อจำกัดข้อแลกเปลี่ยนจริงและตัวเลือกแบบมีเงื่อนไข AP50/TP-only ที่ไม่มีในตารางย่อต้องอ้าง CSV/REPORT; TP-only ใช้คู่ GT คนละชุดได้ ห้ามใส่กรณีภาพหรือคะแนนถ่วงน้ำหนัก -->

### {{MODEL}}

{{MEASURED_STRENGTH_AND_LIMITATION}}

{{RESOURCE_TRADE_OFF_AND_CONDITIONAL_CANDIDATE}}

## ผู้ชนะในแต่ละด้าน

<!-- 6 ด้าน: Mask mAP50-95, AP75, Recall, inference, pipeline, VRAM ใช้หน่วยชัดเจนและผลครบทุกโมเดล -->

| ด้าน | โมเดล | ผลลัพธ์ |
| --- | --- | --- |

## สิ่งที่ตัวเลขบอกเรา

{{THREE_TO_FIVE_MEASURED_FINDINGS_INCLUDING_DESCRIPTIVE_NEAR_TIES}}

## ข้อแลกเปลี่ยนหลัก

### ความแม่นยำกับความเร็ว

{{CANONICAL_ACCURACY_GAP_AND_LATENCY_DIFFERENCE}}

### ความแม่นยำกับหน่วยความจำ

{{CANONICAL_ACCURACY_GAP_AND_ALLOCATED_VRAM_DIFFERENCE}}

## ข้อควรระวังในการตีความ

ไม่มีการทดสอบนัยสำคัญทางสถิติ; MOTS20 ไม่ใช่ผลทดสอบความทนทานต่อ CCTV ขั้นสุดท้าย
Pipeline ไม่รวม RLE preparation และการอ่านเขียนดิสก์; VRAM เป็น peak allocated ของ benchmark

## ข้อมูลสำหรับนำไปรวมต่อ

{{CROSS_TIER_CANDIDATES_WITHOUT_WEIGHTED_SCORE_OR_FINAL_17_MODEL_CONCLUSION}}

[TIER_RESULTS.csv](metrics/TIER_RESULTS.csv) · [REPORT.md](REPORT.md) ·
[การวิเคราะห์ภาพ](PRESENTATION_SUMMARY_TH.md) ·
[การศึกษาหลัก](https://github.com/folklazy/YOLO_Instance_Segmentation_MOTS20_Scaling_Study)
