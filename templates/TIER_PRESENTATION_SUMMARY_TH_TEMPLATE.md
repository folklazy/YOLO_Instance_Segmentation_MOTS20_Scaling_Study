# {{TIER}} — การวิเคราะห์ภาพและพฤติกรรมเชิงคุณภาพ

## 1. ภาพรวมผลการทดลอง

{{ONE_SHORT_PARAGRAPH_WITH_STATUS_AND_LINK_TO_RESULTS_SUMMARY}}

| โมเดล | Mask mAP50-95 | AP75 | Recall |
| --- | --- | --- | --- |

## การเลือกกรณีและการอ่านภาพ

<!-- วิเคราะห์เมื่อทุกโมเดลครบแล้วเท่านั้น เลือกประมาณ 3–5 กรณีจากภาพจริง: ข้อได้เปรียบข้อผิดพลาดร่วมกรณีสวนอันดับผลคล้ายกัน/คะแนนใกล้ ใช้กรณีร่วมหนึ่งกรณีและกรณีแยกพฤติกรรมตามขนาดเมื่อมีประโยชน์ เหตุผลภาพซ้ำต้องชัด ไม่ถือว่าเป็นตัวอย่างอิสระเพิ่ม ใช้ภาพเดิมก่อน แล้วจึงสร้างจาก saved RLE; ไม่ inference เพื่อเอกสาร เก็บภาพเต็มพร้อม ROI พิกัดเดียวกันทุกโมเดลและเปิดใช้เวอร์ชันผ่าน QUALITATIVE_SELECTION.json -->
{{SELECTION_POOL_AND_REASONS}}

## กรณี {{N}} — {{SHORT_DIAGNOSTIC_TITLE}}

เหตุผลที่เลือก: {{REASON}} · ลำดับภาพ: {{SEQUENCE}} · เฟรม: {{FRAME}}

### ภาพเปรียบเทียบ

<!-- ทำบล็อกกรณีซ้ำ 3–5 ครั้ง เฉพาะภาพ repository ที่มีจริง ใช้เฟรม/พื้นที่/สัดส่วนภาพ/policy เดียวกัน เรียงต้นฉบับ/GT, YOLO26, YOLO11, YOLOv9 e/c เฉพาะสองขนาดใหญ่, YOLOv8 ตรวจ PNG, Git tracking และการโหลดจริงหลัง push; เพิ่มข้อยกเว้นชื่อไฟล์ที่เลือกใน .gitignore -->
{{ACTUAL_COMPARISON_IMAGE_EMBED}}

### สิ่งที่เห็นจากภาพ

{{DIRECT_OBSERVATIONS_BY_MODEL_AND_REGION_OR_GT_ID}}

### วิเคราะห์

{{INTERPRETATION_OF_BEHAVIOR_AND_ALTERNATIVE_EXPLANATIONS}}

### เชื่อมกับผลเชิงตัวเลข

{{DIRECTIONAL_CONSISTENCY_OR_COUNTEREXAMPLE_TO_CANONICAL_MAP_AP75_RECALL}}

### ใช้ประกอบการเลือกอย่างไร

**ประเด็นปัญหา / บทบาท:** {{DISCRIMINATING_SHARED_FAILURE_TRADE_OFF_OR_CONTROL}}

**ใช้ประกอบการเลือก:** {{SPECIFIC_PRIORITY_AND_SUPPORTED_MODEL_COMPARISON}}

<!-- FN อาจเป็น mask ที่ไม่ผ่าน IoU; FP อาจทับ GT ที่มีคู่แล้ว ตรวจ saved masks ก่อนเรียกว่าคนปลอมในฉากหลังภาพเดียวไม่ยืนยันความถี่ทั้งชุดข้อมูล/นัยสำคัญ/CCTV; เวลาและ VRAM อ่านจาก benchmark -->
**ขอบเขตหลักฐาน:** {{WHAT_THIS_CASE_CANNOT_ESTABLISH}}

## วิเคราะห์ข้อผิดพลาด

| รูปแบบข้อผิดพลาด | โมเดลที่พบ | กรณีที่พบ | การตีความ |
| --- | --- | --- | --- |

## ตรวจภาพของคู่ที่คะแนนใกล้กัน

{{DESCRIPTIVE_NEAR_TIE_PAIR_AND_SAME_FRAME_COMPARISON_WHERE_PRACTICAL}}

## สิ่งที่เรียนรู้จากภาพจริง

### ข้อสังเกต {{N}}

{{DIRECT_OBSERVATION}}

**การตีความ:** {{CAUTIOUS_INTERPRETATION}}

## เมื่อดูทั้งตัวเลขและภาพร่วมกัน

{{SYNTHESIS_OF_MAP_RECALL_AP75_AND_VISIBLE_BEHAVIOR}}
เวลาแฝงและ VRAM เป็นการวัดระดับระบบอ่านจาก benchmark;
ภาพ segmentation ไม่สามารถอธิบายหรือวัดสองค่านี้ได้

## ถ้าพิจารณาทั้งผลเชิงตัวเลขและภาพ

| สิ่งที่ให้ความสำคัญ | โมเดลที่พิจารณา | หลักฐาน |
| --- | --- | --- |

## ข้อจำกัด

- เฟรมที่เลือกเป็นตัวอย่างเชิงคุณภาพ ไม่แทนตัวชี้วัดระดับชุดข้อมูล
- เลือกทั้งข้อได้เปรียบข้อผิดพลาดกรณีสวนอันดับและผลคล้ายกัน เพื่อลด cherry-picking
- MOTS20 ไม่ใช่ผลทดสอบความทนทานต่อ CCTV ขั้นสุดท้าย
- Qualitative observations และ numerical near ties ไม่ใช่นัยสำคัญทางสถิติ

## รายละเอียดเต็ม

[บทสรุปเชิงตัวเลข](RESULTS_SUMMARY_TH.md) · [REPORT.md](REPORT.md) ·
[TIER_RESULTS.csv](metrics/TIER_RESULTS.csv) ·
[การศึกษาหลัก](https://github.com/folklazy/YOLO_Instance_Segmentation_MOTS20_Scaling_Study)
