# วิธีทดลองร่วม

ค่า config/protocol ที่ตรึงใน Largest เป็นหลักฐานอ้างอิง รายละเอียดและข้อกำหนดใช้ซ้ำอยู่ใน [STUDY_STANDARD.md](STUDY_STANDARD.md) ส่วน implementation hashes และสภาพแวดล้อม baseline อยู่ใน [BASELINE_REFERENCE.json](provenance/BASELINE_REFERENCE.json)

## การประเมินความแม่นยำ

ภาพ → OpenCV BGR → ปรับขนาดรักษาสัดส่วนและเติม letterbox ตรงกลาง → RGB BCHW FP32 /255 → YOLO Person instance segmentation (YOLO26 ใช้ one-to-many NMS) → mask ที่มิติต้นฉบับ → จับคู่ valid Person GT ก่อนตรวจ ignore → ตัวชี้วัดรวมที่เกณฑ์คงที่และ mask AP ที่เรียง confidence
ภาพเข้า network 1×3×640×640 ผลคืนมิติต้นฉบับ model max_det=1000 แยกจาก AP maxDet=200
GT class 2 คือ Person หลังจับ valid GT แล้ว เฉพาะ prediction ไม่มีคู่ที่ทับ union class 10 ตามพื้นที่ prediction ด้วย IOA ≥0.50 จึง ignore fixed confidence 0.25, mask IoU 0.50; AP confidence floor 0.001, NMS IoU 0.70; AP ใช้ mask IoU 0.50:0.05:0.95 และ 101 จุด Recall ประเมิน segmentation รายเฟรมไม่ใช่ MOTS tracking

## การวัดเวลา

ใช้ subset 100 เฟรมเรียงตาม manifest เดิม แยกเวลาโหลดโมเดล ทำ warmup 10 ครั้งและวัด 3 รอบที่ไม่ถูกรบกวน โดย synchronize preprocessing/H2D, forward และ postprocessing/native masks/การส่งข้อมูลแบบบีบอัด pipeline เป็นผลรวม stage เหล่านี้ FPS คำนวณจากค่าเฉลี่ยและ VRAM เป็น allocator peak
เตรียม RLE วัดแยก ไม่นับการอ่าน/decode ดิสก์ โหลดโมเดล GT คำนวณตัวชี้วัดภาพประกอบและ serialization Largest ตัดเวลารอบแรกที่ถูกรบกวนออกและใช้ clean repetitions เดิมเป็นค่าหลัก ดูขอบเขตและ hash ในมาตรฐานและ provenance ของแต่ละขนาด

## เหตุผลที่ต้องใช้โพรโทคอลร่วม

ข้อมูลลำดับเฟรมและมิติต้นฉบับมีผลต่อภาระงาน การเตรียมภาพขนาดเข้าและ FP32 มีผลทั้ง mask และเวลา ตัวประเมิน confidence/NMS/ignore/maxDet มีผลต่อ prediction ที่ถูกนับ GPU, warmup, synchronization, ขอบเขต stage และการกันรบกวนมีผลต่อเวลา/หน่วยความจำหากเปลี่ยนเงื่อนไขเหล่านี้ ต้องชี้แจงว่าผลไม่อยู่ในเงื่อนไขควบคุมเดิม
ความจุและการฝึกเดิมต่างกันตามตระกูล E/X และ C/L ไม่ใช่ความจุเท่ากัน ผลนี้ไม่ยืนยันนัยสำคัญทางสถิติหรือความทนทานต่อ CCTV

## ข้อมูลที่ใช้

| ลำดับภาพ | เฟรม | มิติภาพ | Person GT รายเฟรม | พื้นที่ ignore |
|---|---:|---:|---:|---:|
| MOTS20-02 | 600 | 1920×1080 | 7,039 | 600 |
| MOTS20-05 | 837 | 640×480 | 6,570 | 802 |
| MOTS20-09 | 525 | 1920×1080 | 4,774 | 525 |
| MOTS20-11 | 900 | 1920×1080 | 8,511 | 900 |

รวม 2,862 เฟรมและ Person GT รายเฟรม 26,894 instances คนเดียวกันหลายเฟรมนับเป็นหลาย annotation ใช้เฉพาะ gt/gt.txt ที่มากับ sequence [หลักฐานความสอดคล้องข้อมูล](provenance/DATASET_VALIDATION.json): PASS
