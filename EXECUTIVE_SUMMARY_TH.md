# สรุปผลสำหรับตัดสินใจ — YOLO MOTS20 Scaling Study

## วัตถุประสงค์และขอบเขต

เปรียบเทียบ pretrained YOLO Instance Segmentation ตั้งแต่ขนาดใหญ่สุดถึง Nano เพื่อเลือก candidate ตาม accuracy, speed และ memory ทดสอบครบ 17 โมเดล: YOLO26/YOLO11/YOLOv8 family ละ 5 และ YOLOv9 อีก 2 บน MOTS20 2,862 เฟรมกับ Person GT 26,894 instances ภายใต้ protocol เดียวกัน ไม่มี training/fine-tuning ขั้น Master ใช้ CSV เดิมโดยไม่รัน inference ซ้ำ สถานะ **PASS WITH WARNINGS** จากคำเตือนของ source tiers

## ผลที่ควรรู้

| ด้าน | Model | ค่าที่วัด |
|---|---|---|
| Accuracy | YOLO26x-Seg | Mask mAP50-95 0.603730 |
| Inference | YOLOv8n-Seg | 9.045 ms/frame |
| Pipeline / FPS | YOLO26l-Seg | 76.403 ms/frame / 13.089 FPS |
| Low VRAM | YOLO26l-Seg | 777.52 MiB peak allocated |

**Observation:** YOLO26 นำ mAP/AP75 ทุก tier แต่ใน Nano ผู้ชนะ Recall คือ YOLOv8n จึงไม่มีรุ่นที่ชนะทุก metric

## ขนาดที่ลดลงแลกอะไร

- YOLO26x → l: ΔmAP -1.749 pp แต่ Δinference -49.95%; เป็นจุดที่น่าสนใจเมื่อยอมรับ accuracy loss นี้ได้
- การลดขนาดทั้ง 13 คู่ทำให้ inference เร็วขึ้น แต่ pipeline เพิ่มใน 8 คู่และ allocated VRAM เพิ่มใน 7 คู่ ชื่อขนาดจึงไม่พอสำหรับตัดสินระบบ
- s → n เป็น accuracy drop มากที่สุดของ YOLO26/YOLO11/YOLOv8: -6.394, -5.134 และ -5.970 pp ตามลำดับ
- YOLO26s มี mAP ใกล้ YOLO11x/YOLOv9e มาก แม้เป็น Small; score ใกล้กันไม่แปลว่า output หรือ capacity เท่ากัน

## เลือก candidate ตามข้อจำกัด

| Priority | Candidate | หลักฐานและเงื่อนไข |
| --- | --- | --- |
| Accuracy | YOLO26x-Seg | mAP 0.603730; inference 72.458 ms |
| Pipeline / Low VRAM | YOLO26l-Seg | pipeline 76.403 ms; VRAM 777.52 MiB; mAP 0.586237 |
| ลดเวลาของ forward โดยยังรักษา mAP | YOLO26s-Seg | inference 17.308 ms; mAP 0.536640; อยู่บน Pareto ด้าน inference |
| Nano ที่เน้น mask accuracy | YOLO26n-Seg | mAP 0.472698; Recall 0.687068 ต่ำสุดใน Nano |
| Inference ต่ำสุด | YOLOv8n-Seg | inference 9.045 ms; mAP 0.423062; pipeline ไม่เร็วสุด |

**Interpretation:** YOLO26l เป็นจุดเริ่มต้นที่มีเหตุผลเมื่อ pipeline และ VRAM สำคัญร่วมกัน ส่วน YOLO26x ใช้เมื่อ accuracy สำคัญที่สุด และ YOLOv8n ใช้ทดสอบกรณีที่ forward latency เป็นข้อจำกัดหลัก ไม่ใช้ weighted score เพื่อประกาศผู้ชนะรวม

## ข้อจำกัดและขั้นตอนถัดไป

ผลนี้เป็น frame-level segmentation ไม่ใช่ tracking หรือหลักฐานว่าโมเดลใดเหมาะที่สุดสำหรับ CCTV ภาพวิดีโอสัมพันธ์กันและไม่มี significance test; near ties เป็นเชิงพรรณนา รุ่น E/X และ C/L ไม่ได้มี capacity เท่ากัน และ TP-only quality ไม่นับคนที่ไม่ match

Pipeline/FPS ไม่รวม decode, RLE preparation และเขียนผล หน่วย VRAM เป็น allocator peak ไม่ใช่ GPU memory ทั้งหมด ขั้นต่อไปควรกำหนด robustness dataset และ deployment pipeline ให้ตรงงานจริง แล้วประเมิน candidates ด้วยเกณฑ์ที่อนุมัติแยกต่างหาก ขั้นนี้ไม่เริ่ม benchmark ใหม่

[ผลเต็ม](MASTER_RESULTS.md) · [Research insights](RESEARCH_INSIGHTS_TH.md) · [ข้อมูลหลัก](metrics/MASTER_17_MODELS.csv)
