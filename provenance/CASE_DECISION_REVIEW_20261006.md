# ตรวจการใช้ visual cases ประกอบการเลือกโมเดล

ตรวจภาพเต็ม 20 cases และภาพขยาย 20 cases ของทั้งห้า tier พร้อม recheck counts ของ 12 frozen candidates ต่อ tier ใช้ original MOTS20 / GT / saved RLE เท่านั้น ไม่ inference และไม่เปลี่ยน measured values ไม่ใช่ final cross-tier synthesis

**คำตอบ:** ใช้ประกอบการเลือกได้ แต่แต่ละ case มีบทบาทต่างกัน ไม่ใช่ทุกภาพที่เห็น pain point แล้วจัดผู้ชนะได้ และชุดนี้ไม่แทน 2,862 เฟรม

| Tier | Cases ที่ช่วยจำแนกพฤติกรรม | Cases ที่ควรอ่านเป็นข้อจำกัด / control |
|---|---|---|
| [Largest](https://github.com/folklazy/YOLO_Large_Seg_MOTS20_Benchmark/blob/main/PRESENTATION_SUMMARY_TH.md) | Case 1: GT 2002 แยก YOLOv8x จากอีกสาม; Case 3: GT 2029 แยกคู่ near-tied YOLO11x/YOLOv9e และ FP ของ YOLOv8x | Case 2 มี FN/FP ร่วม; Case 4 เก็บครบเหมือนกัน ไม่ใช้ชี้ผู้ชนะ |
| [Second-largest](https://github.com/folklazy/YOLO_Second_Largest_Seg_MOTS20_Benchmark/blob/main/PRESENTATION_SUMMARY_TH.md) | Case 1: YOLO26l เก็บ GT 2002 เพิ่ม; Case 3: counts เท่ากันแต่พลาด GT ต่างชุด และมี trade-off TP/FP | Case 2 แสดงข้อจำกัดร่วม; Case 4 เป็น control ของผลคล้ายกัน |
| [Medium](https://github.com/folklazy/YOLO_Medium_Seg_MOTS20_Benchmark/blob/main/PRESENTATION_SUMMARY_TH.md) | Case 1: GT 2002; Case 3: ตัวนำ mAP ไม่ได้มี TP มากสุด และ model ที่ FP น้อยพลาด GT เพิ่ม | Case 2 มี FN ร่วม/near-threshold mask; Case 4 เป็น control ไม่ใช่ accuracy near tie หรือการพิสูจน์ latency near tie |
| [Small](https://github.com/folklazy/YOLO_Small_Seg_MOTS20_Benchmark/blob/main/PRESENTATION_SUMMARY_TH.md) | Case 1: TP 10/7/8 และคนเล็ก/GT ด้านหลัง; Case 3: YOLO26s พลาด 2043 ขณะที่คู่ near-tied เก็บครบ; Case 4: FP เพิ่มของ YOLO11s ในฉากหลังร้าน | Case 2 แสดง FN ร่วมและ error คนละ GT; TP-only IoU สูงกว่าไม่เท่ากับเก็บคนครบกว่า |
| [Nano](https://github.com/folklazy/YOLO_Nano_Seg_MOTS20_Benchmark/blob/main/PRESENTATION_SUMMARY_TH.md) | Case 1: เก็บ GT ต่างชุด; Case 3: TP เพิ่มพร้อม FP เพิ่ม; Case 4: extra masks ทับ valid GT ที่มีคู่แล้ว | Case 2 ทุกโมเดล TP 7 / FN 6 เท่ากัน เป็น common failure |

## จุดที่แก้จากการตรวจภาพขยาย

- ภาพเต็มเดิมทำให้คนเล็กและ label อ่านยาก เพิ่ม ROI พร้อม GT outlines โดยใช้พิกัดและ scale เดียวกันทุกโมเดล และยังแสดงภาพเต็มทั้งหมด
- Nano Case 4 เดิมเรียก YOLO26n FP ว่าฉากหลังไม่ทับ valid GT: แก้เป็น extra mask ทับ GT 2001 ที่มีคู่แล้ว ส่วน YOLOv8n มีสอง unmatched masks ทับ GT 2019 ไม่ใช่หลักฐานคนปลอมหรือ instance merging
- บาง FN มี mask ใกล้ threshold เช่น Second-largest Case 3 GT 2017, Medium Case 2 GT 2018, Small Case 1 GT 2040/2027 และ Nano Case 1 GT 2029 จึงเพิ่ม per-GT matching diagnostics ไม่เรียกทุก FN ว่าไม่มี detection
- Largest Case 1 ไม่ได้แยก YOLO26x/YOLO11x/YOLOv9e ซึ่ง match GT 2002 ได้เหมือนกัน ส่วน Case 4 ของ Largest/Second-largest/Medium ใช้เป็น control ไม่มี missed/extra instance ให้ชี้ winner

## ขอบเขตของการตัดสินใจ

ใช้ภาพเพื่อระบุประเภท error ที่งานยอมรับได้หรือยอมรับไม่ได้ แล้วอ่าน mAP/AP75/Recall/Precision จาก canonical CSV ร่วมกัน การส่ง extra masks อาจสำคัญกับ downstream extraction แต่ยังไม่ได้วัดผลของระบบปลายทางจาก cases นี้ Latency และ VRAM ใช้ clean system benchmark ภาพไม่สามารถวัดสองค่านี้ได้

คง 4 cases ต่อ tier เพราะมีทั้งความต่าง ข้อผิดพลาดร่วม กรณีสวนอันดับ และ control ไม่เลือกเฉพาะเฟรมที่คะแนนนำชนะ และไม่รับรองว่าเลือก extreme ที่สุดของทั้ง dataset Pool มีเพียง 12 frozen frames และ selected set ยังไม่ครอบคลุม MOTS20-11 จึงไม่ใช่ representative sample หรือสถิติความถี่ failure

หลักฐาน: [review provenance](CASE_DECISION_REVIEW_20261006.json); แต่ละ tier มี `manifests/CASE_DECISION_AUDIT.json` และ `outputs/visualizations/qualitative/FOCUS_EVIDENCE.json` พร้อม hashes/ROI/diagnostics ภาพขยายไม่เพิ่มรายละเอียดจากต้นฉบับ ไม่แทน robustness test บน CCTV จริง และไม่ยืนยัน statistical significance
