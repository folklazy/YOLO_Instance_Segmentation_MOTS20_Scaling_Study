# คู่มืออ่าน Metric — YOLO MOTS20 Scaling Study

## Accuracy และความครบถ้วน

| Metric | ความหมายในชุดนี้ | ข้อควรระวัง |
|---|---|---|
| Mask mAP50-95 | AP ของ Person masks เฉลี่ยที่ IoU 0.50:0.05:0.95 และ 101 recall points | เป็น segmentation AP แบบ pooled ไม่ใช่ box AP หรือ mean ของ sequence AP |
| AP50 / AP75 | AP ที่ mask IoU 0.50 / 0.75 | AP75 เข้มเรื่อง overlap มากขึ้น แต่ไม่แยก boundary quality ออกจากการหา instance ได้ |
| Precision | TP / (TP + FP) ที่ confidence ≥0.25 | FP คือ unmatched prediction; ไม่จำเป็นต้องเป็นคนที่ไม่มีจริง |
| Recall | TP / (TP + FN) ที่ matching mask IoU ≥0.50 | FN อาจมี mask อยู่แล้วแต่ overlap ไม่ผ่าน ไม่ได้แปลว่าขาด detection เสมอ |
| F1 | 2TP / (2TP + FP + FN) | เป็น fixed operating point ไม่ใช่ค่าแทน mAP |
| TP-only IoU / Dice | คุณภาพ mask เฉพาะคู่ที่ match ผ่าน | ไม่รวมคนที่พลาด และแต่ละโมเดลอาจ match GT คนละ subset |

GT Person เป็น class 2; model Person เป็น class 0 valid GT ได้สิทธิ์ match ก่อน ignore region จาก class 10 unmatched predictions ที่ overlap ignore union / prediction area ≥0.50 ถูก ignore หลัง matching ไม่ถูกนับ FP และไม่ clip Person masks

AP confidence floor 0.001; fixed metrics ใช้ 0.25; NMS box IoU 0.70 ไม่ใช่ mask matching IoU Model max_det=1000 ต่างจาก evaluator AP maxDet=200 ซึ่งผ่าน sensitivity เทียบ reference1000 ด้วย tolerance แบบ strict <0.0001

## เวลาและ throughput

| Metric | หน่วย / ขอบเขต | วิธีอ่าน |
|---|---|---|
| Inference | ms/frame | forward ของโมเดลหลัง preprocessing และ CUDA synchronization |
| Preprocessing | ms/frame | resize/letterbox/normalization และ H2D ตาม adapter ที่ freeze |
| Postprocessing | ms/frame | NMS, native masks, bit packing/transfer และ prediction objects |
| Pipeline | ms/frame | preprocessing + inference + postprocessing |
| FPS | frames/s | 1000 / mean pipeline ms; ไม่ใช่ mean ของ FPS รายเฟรม |
| RLE preparation | ms/frame แยกต่างหาก | ไม่ได้รวมใน pipeline หรือ FPS |
| Model loading | seconds แยกต่างหาก | ไม่นับใน steady-state pipeline |

ใช้ timing frames ที่ freeze 100 เฟรม, warmup10 และ clean measured3รอบต่อโมเดล รวม300 observations; mean/median/P50/P95/std คงค่าจาก source โดย std เป็น population std ไม่เฉลี่ย percentiles จากรายรอบ Largest timing รอบแรกที่ปนเปื้อนถูกเก็บไว้แต่ตัดออกทั้งหมด ใช้เฉพาะ clean_repetition

Pipeline ไม่รวม image decode/disk I/O, GT, evaluator, visualization, RLE preparation และการเขียนผล จึงไม่ใช่ end-to-end CCTV หรือ saved-mask throughput ค่า ultralytics_postprocess_inclusive เป็น diagnostic subset ห้ามนำมาบวก pipeline ซ้ำ

## หน่วยความจำและความซับซ้อน

| Metric | หน่วย | ข้อจำกัด |
|---|---|---|
| Peak allocated VRAM | MiB = 2^20 bytes | peak สูงสุดของ accepted rounds หลัง warmup รวม resident model; ไม่ใช่ memory GPU ทั้งหมดหรือ OOM threshold |
| Peak reserved VRAM | MiB | allocator reservation มีใน source timing; ไม่ใช้แทน allocated ใน Pareto นี้ |
| Parameters | จำนวนเต็ม | loaded checkpoint parameters; fused runtime count มีแยกใน complexity CSV |
| GFLOPs | estimate ที่ imgsz640 | ไม่ใช่ measured operations และไม่รับรอง latency |
| Checkpoint size | MB = 10^6 bytes | ขนาดไฟล์ไม่ใช่ VRAM ระหว่างประมวลผล |

## Scaling delta และ Pareto

Delta = smaller − larger; relative % = delta / larger ×100 ใช้ฐานรุ่นใหญ่เสมอ เช่น fraction ที่ลด0.01 คือ −1 percentage point ไม่ใช่ −1% ของค่ารุ่นใหญ่โดยอัตโนมัติ เครื่องหมายลบของเวลา/VRAMหมายถึงค่าลด ส่วนเครื่องหมายลบของ mAP หมายถึง accuracy ลด

Pareto หลัก maximize mAP / minimize inference; Pareto รอง maximize mAP / minimize allocated VRAM จุดถูก dominate เมื่ออีกโมเดลไม่แย่กว่าทั้งสองด้านและดีกว่าอย่างน้อยหนึ่งด้าน คำนวณจากค่าไม่ปัดเศษ จุดที่เท่ากันยังอยู่ทั้งคู่ ไม่มี weighted score และไม่ใช้ Pareto สองด้านแทนทุกข้อจำกัดของงาน

Near-tie screen ใช้ |ΔmAP| ≤0.001 หรือ0.1 pp เพื่อคัดคู่เชิงพรรณนา ไม่ใช่ significance/equivalence test หรือเกณฑ์เลือกผู้ชนะโดยทศนิยมท้าย

## ขอบเขตข้อสรุป

26,894 instances เป็น annotations รายเฟรม ไม่ใช่คนไม่ซ้ำ 2,862 เฟรมเป็นข้อมูลเดียวกันทุกโมเดล ไม่ใช่17ชุดอิสระ Frames วิดีโอสัมพันธ์กัน รุ่น E/X และ C/L ไม่ได้ capacity เท่ากัน และ pretrained checkpoints ต่าง pretraining ผลนี้ไม่ใช่ tracking หรือหลักฐาน CCTV robustness

[Master results](MASTER_RESULTS.md) · [Canonical schema](DATA_SCHEMA.md) · [Methodology](METHODOLOGY_REFERENCE.md) · [Source provenance](provenance/SOURCE_MANIFEST.json)
