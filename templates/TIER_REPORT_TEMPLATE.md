# {{TIER}} การทดสอบ YOLO Instance Segmentation — MOTS20

## 1. สถานะการทดลอง

{{1._EXPERIMENT_STATUS}}

## 2. โมเดลที่ทดสอบ

{{2._MODELS_TESTED}}

| ตระกูล | โมเดล | จำนวนพารามิเตอร์ | GFLOPs | Checkpoint (MB) |
| --- | --- | --- | --- | --- |

## 3. ความสอดคล้องกับโพรโทคอล

<!-- ตารางกลาง 9 รายการ: ข้อมูล, ตัวประเมิน, การเตรียมภาพ, ขนาดภาพเข้า, ความละเอียดเชิงตัวเลข, เกณฑ์, maxDet, วิธีวัดเวลา, environment รายการตรวจละเอียดเชื่อมไปยัง manifest -->
{{3._PROTOCOL_COMPATIBILITY}}

| รายการ | สถานะ |
| --- | --- |

## 4. ผลลัพธ์รวม

{{4._OVERALL_RESULTS}}

| โมเดล | Mask mAP50-95 | AP50 | AP75 | Precision | Recall | F1 | TP-only IoU | TP-only Dice | Inference (ms) | Pipeline (ms) | FPS | Peak allocated VRAM (MiB) | จำนวนพารามิเตอร์ | GFLOPs | Checkpoint (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

## 5. ผู้ชนะในแต่ละด้าน

<!-- 7 แถวตามลำดับ: mAP, AP75, Recall, inference, pipeline, FPS, VRAM ค่าจาก CSV และระบุหน่วย -->
{{5._TIER_WINNERS}}

| ด้าน | โมเดล | ผลลัพธ์ |
| --- | --- | --- |

## 6. ข้อค้นพบสำคัญ

{{6._KEY_FINDINGS}}

## 7. ข้อสังเกตรายลำดับภาพ

<!-- สูงสุด/ต่ำสุดรายโมเดลจาก PER_SEQUENCE_RESULTS พร้อมลำดับที่ต่างจาก pooled AP ไม่คัดลอกทุก cell มาเล่าซ้ำ -->
{{7._PER-SEQUENCE_OBSERVATIONS}}

## 8. ประสิทธิภาพและการใช้ทรัพยากร

{{8._EFFICIENCY_AND_RESOURCE_OBSERVATIONS}}

## 9. คำเตือนและข้อสังเกตผิดปกติ

{{9._WARNINGS_AND_ANOMALIES}}

## 10. ข้อจำกัด

{{10._LIMITATIONS}}

## 11. หลักฐานสำหรับตรวจสอบซ้ำ

{{11._REPRODUCIBILITY_AND_SOURCE_ARTIFACTS}}

## 12. ความเชื่อมโยงกับการศึกษาทุกขนาด

{{12._RELATION_TO_FULL_SCALING_STUDY}}

## การวิเคราะห์เชิงคุณภาพ

เมื่อทุกโมเดลครบ ให้เชื่อมไปยัง [PRESENTATION_SUMMARY_TH.md](PRESENTATION_SUMMARY_TH.md)
และกรณี selection เวอร์ชันปัจจุบัน เก็บการอภิปรายภาพไว้ที่นั่น เมื่อไม่ครบให้ระบุว่ารอผล
