# ทดลองประสิทธิภาพ YOLO Instance Segmentation บน MOTS20
## ตั้งแต่รุ่นขนาดใหญ่สุดจนถึง Nano

## ข้อมูลร่วมของการทดลอง

ครบ 17 official pretrained segmentation checkpoints: YOLO26/YOLO11/YOLOv8 family ละ 5 และ YOLOv9 อีก 2 ทุกโมเดลใช้ MOTS20 2,862 เฟรมกับ Person GT 26,894 instances, imgsz640, batch1, FP32, Tesla T4, preprocessing/evaluator/thresholds เดียวกัน ไม่มี training/fine-tuning ขั้น Master สังเคราะห์ CSV เท่านั้น สถานะ PASS WITH WARNINGS; ไม่รัน inference ซ้ำ

ค่าตารางเป็น mAP/Recall แบบ fraction, inference/pipeline เป็น ms/frame, FPS คำนวณจาก pipeline mean และ VRAM เป็น peak allocated MiB เวลา pipeline ไม่รวม decode, RLE และเขียนไฟล์ วิธีทดลองเต็มอยู่ใน [methodology](METHODOLOGY_REFERENCE.md)

# 1. Largest (X/E)

| Model | Tier | Mask mAP50-95 | AP75 | Recall | Inference ms | Pipeline ms | FPS | VRAM MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO26x-Seg | Largest (X/E) | 0.603730 | 0.664426 | 0.846285 | 72.458 | 109.639 | 9.121 | 952.18 |
| YOLO11x-Seg | Largest (X/E) | 0.536674 | 0.575845 | 0.822228 | 69.406 | 105.885 | 9.444 | 942.48 |
| YOLOv9e-Seg | Largest (X/E) | 0.536642 | 0.572100 | 0.826578 | 65.888 | 103.376 | 9.673 | 855.41 |
| YOLOv8x-Seg | Largest (X/E) | 0.524758 | 0.560452 | 0.814271 | 67.843 | 109.441 | 9.137 | 1001.13 |

**Observation:** YOLO26x นำ mAP/AP75/Recall ส่วน YOLOv9e เร็วสุดทั้ง inference/pipeline และ VRAM ต่ำสุดใน tier นี้ YOLO11x/YOLOv9e มี mAP near-tied เชิงพรรณนา จึงควรเทียบข้อจำกัดด้านเวลาและความครบถ้วนร่วมด้วย

ภาพและ failure analysis ของ tier: [PRESENTATION_SUMMARY_TH.md](https://github.com/folklazy/YOLO_Large_Seg_MOTS20_Benchmark/blob/main/PRESENTATION_SUMMARY_TH.md)

# 2. Second-largest (L/C)

| Model | Tier | Mask mAP50-95 | AP75 | Recall | Inference ms | Pipeline ms | FPS | VRAM MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO26l-Seg | Second-largest (L/C) | 0.586237 | 0.643537 | 0.834387 | 36.266 | 76.403 | 13.089 | 777.52 |
| YOLO11l-Seg | Second-largest (L/C) | 0.528228 | 0.565728 | 0.809772 | 35.387 | 76.810 | 13.019 | 794.47 |
| YOLOv9c-Seg | Second-largest (L/C) | 0.517646 | 0.559180 | 0.802930 | 35.293 | 76.921 | 13.000 | 838.49 |
| YOLOv8l-Seg | Second-largest (L/C) | 0.520145 | 0.558149 | 0.809400 | 40.090 | 83.558 | 11.968 | 855.27 |

**Observation:** YOLO26l นำ mAP/AP75/Recall พร้อม pipeline และ VRAM ที่ดีที่สุดใน tier; YOLOv9c มี inference ต่ำสุด ขณะที่ผล pipeline ของสามรุ่นแรกใกล้กันและยังไม่มี significance test

ภาพและ failure analysis ของ tier: [PRESENTATION_SUMMARY_TH.md](https://github.com/folklazy/YOLO_Second_Largest_Seg_MOTS20_Benchmark/blob/main/PRESENTATION_SUMMARY_TH.md)

# 3. Medium (M)

| Model | Tier | Mask mAP50-95 | AP75 | Recall | Inference ms | Pipeline ms | FPS | VRAM MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO26m-Seg | Medium (M) | 0.574222 | 0.629038 | 0.815312 | 32.043 | 77.309 | 12.935 | 912.16 |
| YOLO11m-Seg | Medium (M) | 0.518307 | 0.557205 | 0.805049 | 30.574 | 76.971 | 12.992 | 889.38 |
| YOLOv8m-Seg | Medium (M) | 0.506971 | 0.542131 | 0.793857 | 27.232 | 78.112 | 12.802 | 1018.76 |

**Observation:** YOLO26m นำ mAP/AP75/Recall; YOLOv8m forward เร็วสุด แต่ YOLO11m นำ pipeline/FPS/VRAM ตัวเลขนี้แสดงว่าการเลือกจาก inference เพียงคอลัมน์เดียวอาจให้คนละ candidate กับการเลือก pipeline

ภาพและ failure analysis ของ tier: [PRESENTATION_SUMMARY_TH.md](https://github.com/folklazy/YOLO_Medium_Seg_MOTS20_Benchmark/blob/main/PRESENTATION_SUMMARY_TH.md)

# 4. Small (S)

| Model | Tier | Mask mAP50-95 | AP75 | Recall | Inference ms | Pipeline ms | FPS | VRAM MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO26s-Seg | Small (S) | 0.536640 | 0.586796 | 0.789581 | 17.308 | 78.828 | 12.686 | 872.67 |
| YOLO11s-Seg | Small (S) | 0.483269 | 0.513823 | 0.764148 | 15.693 | 88.836 | 11.257 | 1007.50 |
| YOLOv8s-Seg | Small (S) | 0.482763 | 0.506411 | 0.767309 | 15.220 | 92.173 | 10.849 | 1139.01 |

**Observation:** YOLO26s นำ mAP/AP75/Recall และ pipeline/VRAM ใน tier; YOLOv8s forward เร็วสุด ส่วน YOLO11s/YOLOv8s near-tied ด้าน mAP แต่ไม่ควรเรียกว่า error behavior เหมือนกัน

ภาพและ failure analysis ของ tier: [PRESENTATION_SUMMARY_TH.md](https://github.com/folklazy/YOLO_Small_Seg_MOTS20_Benchmark/blob/main/PRESENTATION_SUMMARY_TH.md)

# 5. Nano (N)

| Model | Tier | Mask mAP50-95 | AP75 | Recall | Inference ms | Pipeline ms | FPS | VRAM MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO26n-Seg | Nano (N) | 0.472698 | 0.506766 | 0.687068 | 13.042 | 82.390 | 12.137 | 1043.13 |
| YOLO11n-Seg | Nano (N) | 0.431929 | 0.450826 | 0.693798 | 11.745 | 97.441 | 10.263 | 1030.00 |
| YOLOv8n-Seg | Nano (N) | 0.423062 | 0.439115 | 0.696921 | 9.045 | 95.683 | 10.451 | 1117.09 |

**Observation:** YOLO26n นำ mAP/AP75 และ pipeline/FPS; YOLOv8n นำ Recall/inference และ YOLO11n ใช้ VRAM ต่ำสุดใน tier ผู้ชนะ mask accuracy ยังมี Recall ต่ำสุด จึงต้องระบุ priority ก่อนเลือก

ภาพและ failure analysis ของ tier: [PRESENTATION_SUMMARY_TH.md](https://github.com/folklazy/YOLO_Nano_Seg_MOTS20_Benchmark/blob/main/PRESENTATION_SUMMARY_TH.md)

# 6. สรุปรวมทุกขนาด

## ภาพรวม 17 โมเดล

| Family | Models | Largest → smallest available | mAP endpoints | Inference endpoints (ms) |
| --- | --- | --- | --- | --- |
| YOLO26 | 5 | YOLO26x-Seg → YOLO26n-Seg | 0.603730 → 0.472698 | 72.458 → 13.042 |
| YOLO11 | 5 | YOLO11x-Seg → YOLO11n-Seg | 0.536674 → 0.431929 | 69.406 → 11.745 |
| YOLOv8 | 5 | YOLOv8x-Seg → YOLOv8n-Seg | 0.524758 → 0.423062 | 67.843 → 9.045 |
| YOLOv9 | 2 | YOLOv9e-Seg → YOLOv9c-Seg | 0.536642 → 0.517646 | 65.888 → 35.293 |

## ผู้ชนะและข้อแลกเปลี่ยนรวม

- Accuracy: YOLO26x-Seg, mAP 0.603730
- Inference: YOLOv8n-Seg, 9.045 ms
- Pipeline: YOLO26l-Seg, 76.403 ms / 13.089 FPS
- VRAM: YOLO26l-Seg, 777.52 MiB

## Family scaling และจุดลดขนาด

ทุก family มี mAP/forward ลดลงเมื่อ checkpoint เล็กลง x → l และ e → c เป็นจุดที่ควรพิจารณาเมื่อยอมรับ accuracy loss ได้ โดย YOLO26x → l เปลี่ยน mAP -1.749 pp และ inference -49.95%

s → n เป็น adjacent mAP drop มากที่สุดของ YOLO26, YOLO11 และ YOLOv8: -6.394, -5.134 และ -5.970 pp การเรียกจุดใดว่า “คุ้ม” ต้องมี accuracy/latency budget ไม่ใช่คะแนนรวม

## Near ties

| Model A | Model B | Absolute mAP gap |
| --- | --- | --- |
| YOLO11x-Seg | YOLOv9e-Seg | 0.000031903 |
| YOLO11x-Seg | YOLO26s-Seg | 0.000033907 |
| YOLOv9e-Seg | YOLO26s-Seg | 0.000002005 |
| YOLOv9c-Seg | YOLO11m-Seg | 0.000660396 |
| YOLO11s-Seg | YOLOv8s-Seg | 0.000505355 |

เป็น descriptive screening ที่ |ΔmAP| ≤0.001; ไม่ใช่ equivalence test คู่ข้าม tier ไม่ได้ตรวจภาพใหม่ในขั้นนี้

## สิ่งผิดปกติและการตีความ

**Observation:** pipeline เพิ่มใน 8/13 ขั้น และ VRAM เพิ่มใน 7/13 ขั้น แม้ parameters ลดทั้งหมด YOLO26l เป็น global pipeline/VRAM winner และ YOLO26n เป็น Nano accuracy leader แต่ไม่ใช่ Recall leader

**Interpretation:** workload ของ native-mask postprocessing เป็นบริบทที่ช่วยอ่านแนวโน้ม แต่การสังเคราะห์นี้ไม่พิสูจน์สาเหตุ รุ่นเล็กจึงต้องเทียบค่า pipeline/VRAM จริง ไม่ตัดสินจากชื่อ n/s/m

## ตารางเลือก candidate อย่างเป็นกลาง

| Priority | Candidate | หลักฐานและเงื่อนไข |
| --- | --- | --- |
| Accuracy | YOLO26x-Seg | mAP 0.603730; inference 72.458 ms |
| Pipeline / Low VRAM | YOLO26l-Seg | pipeline 76.403 ms; VRAM 777.52 MiB; mAP 0.586237 |
| ลดเวลาของ forward โดยยังรักษา mAP | YOLO26s-Seg | inference 17.308 ms; mAP 0.536640; อยู่บน Pareto ด้าน inference |
| Nano ที่เน้น mask accuracy | YOLO26n-Seg | mAP 0.472698; Recall 0.687068 ต่ำสุดใน Nano |
| Inference ต่ำสุด | YOLOv8n-Seg | inference 9.045 ms; mAP 0.423062; pipeline ไม่เร็วสุด |

Pareto ด้าน inference มี 9 candidates; ด้าน VRAM มี YOLO26x/YOLO26l เท่านั้น วัตถุประสงค์ที่ต่างกันให้ชุดตัวเลือกต่างกัน ไม่มี weighted winner

## ข้อสรุปภาพรวม

ผลครบทั้ง 17 โมเดลช่วยเลือก candidate for later CCTV robustness evaluation ตามข้อจำกัดที่กำหนดได้ แต่ไม่ยืนยันความทนต่อสภาพ CCTV หรือ deployment readiness โมเดล pretrained ต่าง capacity/pretraining, frames สัมพันธ์กัน และไม่มี significance test; TP-only quality มีข้อจำกัดจาก matching ขั้นต่อไปต้องกำหนด robustness protocol แยกและวัด pipeline ที่ใช้งานจริง

[Master results](MASTER_RESULTS.md) · [Research insights](RESEARCH_INSIGHTS_TH.md) · [Scaling deltas](metrics/SCALING_DELTAS.csv) · [Source manifest](provenance/SOURCE_MANIFEST.json)
