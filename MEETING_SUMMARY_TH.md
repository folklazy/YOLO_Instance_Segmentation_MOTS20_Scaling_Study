# YOLO Instance Segmentation Scaling Study — MOTS20

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

**Observation:** YOLO26x → l ลด mAP -1.749 pp แต่ inference -49.95%, pipeline -30.31% และ VRAM -18.34% Pipeline Pareto เหลือ x/l

**Interpretation:** l เป็น candidate efficiency point ภายใต้ objectives ที่วัด ส่วน x ยังคงเป็นตัวเลือกเมื่อ accuracy มีน้ำหนักมากกว่า

ใน [รายงานภาพ x/l](YOLO26X_VS_YOLO26L_VISUAL_TH.md) มีทั้งภาพที่ x เก็บเพิ่ม ภาพที่ l เก็บเพิ่ม และกรณี mask คุณภาพต่างกันแม้ coverage เท่ากัน ควรใช้ประกอบการตัดสินโดยไม่ยกภาพเดียวเป็นข้อสรุปทั้ง dataset

## 6. Inference เร็ว ไม่ได้แปลว่า Pipeline เร็ว

![Timing composition](plots/13_timing_stage_composition.png)

**Observation:** Postprocessing มากกว่าครึ่ง pipeline ใน 12 โมเดล YOLOv8n ใช้ inference 9.045 ms แต่ pipeline 95.683 ms; YOLO26l pipeline 76.403 ms

**Interpretation:** Forward เร็วขึ้นยังติดต้นทุน native-mask postprocessing ได้ แต่ stage trends ไม่พิสูจน์สาเหตุเชิง architecture RLE preparation อยู่แยกและไม่รวมใน FPS; diagnostic inclusive stage ไม่บวกซ้ำ

## 7. Accuracy แทบเท่ากัน แต่ resource ต่างกันมาก

YOLO26s/YOLO11x/YOLOv9e มี aggregate mAP near-equal ภายใต้ screen ≤0.001 โดย YOLO26s − YOLO11x มี inference -75.06% และ pipeline -25.55% แต่ Recall ของ Small ต่ำกว่าสองรุ่นใหญ่ และ VRAM ของ 26s สูงกว่า v9e เล็กน้อย

**Interpretation:** Near tie ไม่ใช่ equivalence หรือ prediction เหมือนกัน ดู [same-frame near-tie visual research](NEAR_TIE_26S_11X_V9E_VISUAL_TH.md) เพื่อเห็น GT coverage และ unmatched outputs ต่างกัน และ [resource deltas](metrics/NEAR_TIE_RESOURCE_DELTAS.csv) เพื่อดูทิศทาง B − A

## 8. ผลเหมือนกันทุก sequence หรือไม่

![Sequence heatmap](plots/14_per_sequence_map_heatmap.png)

**Observation:** ตัวนำ YOLO26 คงที่ทุก tier × sequence แต่รุ่นรองมี ranking inversion ใน Largest/Second-largest/Small MOTS20-02 มี mAP ต่ำสุดของทุกโมเดล

**Interpretation:** ค่า pooled ไม่แทนทุก sequence และยังไม่ระบุสาเหตุจาก blur/แสง/มุมกล้อง AP รวมไม่ใช่ค่าเฉลี่ยของ sequence AP

## 9. ถ้าต้องเลือกโมเดล

| Priority | Candidate | หลักฐานและข้อแลกเปลี่ยน |
|---|---|---|
| Accuracy | YOLO26x-Seg | นำ pooled mAP/AP75/Recall แต่มี counterexamples ในภาพจริง |
| Whole-pipeline efficiency | YOLO26l-Seg | Pipeline mean ต่ำสุดและ mAP อันดับสอง; อยู่บน pipeline Pareto |
| Forward latency | YOLOv8n-Seg | Inference ต่ำสุด แต่ pipeline ไม่เร็วสุด |
| Memory | YOLO26l-Seg | Peak allocated VRAM ต่ำสุดใน benchmark; ไม่ใช่ total deployment memory |
| Small-model accuracy | YOLO26s-Seg | นำ Small mAP; mAP ใกล้ 11x/v9e แต่ Recall ต่ำกว่า |
| Later CCTV robustness evaluation | YOLO26x / YOLO26l / YOLO26s | Candidates ตามข้อจำกัด ยังไม่ยืนยัน CCTV superiority |

## 10. ข้อจำกัด

ไม่มี statistical significance test; เฟรมต่อเนื่องสัมพันธ์กันและภาพ diagnostic ไม่ใช่ representative sample TP-only quality ใช้ matched subsets; FP/FN เป็นผลตาม evaluator ไม่ใช่ absence โดยอัตโนมัติ Pipeline FPS ไม่รวม decode/RLE/เขียนผล และ VRAM เป็น allocated peak ไม่ใช่ total deployment memory ผลนี้ยังไม่วัด tracking หรือยืนยัน CCTV robustness และไม่แยก architecture causality

## 11. ขั้นตอนถัดไป

กำหนดข้อจำกัดของงานจริง แล้วทดสอบ candidates บน future robustness/deployment study แยกต่างหาก หากจะใช้ parsing ให้กำหนด output labels, crop policy และผลทั้งระบบก่อน รอบนี้เป็นการสังเคราะห์ผลเดิมและ STOP

อ่านบทวิเคราะห์เต็มที่ [RESEARCH_INSIGHTS_TH.md](RESEARCH_INSIGHTS_TH.md) และ [MASTER_RESULTS.md](MASTER_RESULTS.md)
