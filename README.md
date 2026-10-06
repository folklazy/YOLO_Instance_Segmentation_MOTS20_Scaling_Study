# การศึกษา YOLO Instance Segmentation บน MOTS20 ตามขนาดโมเดล

Largest, Second-largest, Medium, Small และ Nano เสร็จครบแล้ว สถานะ COMPLETE / PASS_WITH_WARNINGS ผลรวม 17 โมเดลยังรอคำสั่งผู้ใช้ repository นี้ควบคุมมาตรฐานและตรวจหลักฐาน ไม่ทำ inference

## ไฟล์ควบคุมการศึกษา

- [STUDY_STANDARD.md](STUDY_STANDARD.md): วิธีทดลอง บทบาทเอกสารและเงื่อนไขหยุด
- [STUDY_STATE.json](STUDY_STATE.json): สถานะการทดลองจริง แยกจากผลตรวจ
- [DATA_SCHEMA.md](DATA_SCHEMA.md): คอลัมน์ หน่วยและเงื่อนไขนำเข้าข้อมูล
- [METHODOLOGY_REFERENCE.md](METHODOLOGY_REFERENCE.md): วิธีทดลองร่วม
- [templates/](templates/): รูปแบบ README, REPORT, RESULTS และ PRESENTATION
- [provenance/](provenance/): หลักฐานต้นทาง baseline ข้อมูลและการตรวจเอกสาร

## การนำทางในชุดการศึกษา

[Largest (X/E)](https://github.com/folklazy/YOLO_Large_Seg_MOTS20_Benchmark) | [Second-largest (L/C)](https://github.com/folklazy/YOLO_Second_Largest_Seg_MOTS20_Benchmark) | [Medium (M)](https://github.com/folklazy/YOLO_Medium_Seg_MOTS20_Benchmark) | [Small (S)](https://github.com/folklazy/YOLO_Small_Seg_MOTS20_Benchmark) | [Nano (N)](https://github.com/folklazy/YOLO_Nano_Seg_MOTS20_Benchmark) | [การศึกษาหลัก](https://github.com/folklazy/YOLO_Instance_Segmentation_MOTS20_Scaling_Study)

## บทบาทและภาษาเอกสาร

เอกสารปัจจุบันใช้ภาษาไทยเป็นหลัก คงชื่อโมเดล metric คำสั่ง path ฟิลด์ schema และสถานะเครื่องตามต้นฉบับ README นำทาง; REPORT เก็บผลเทคนิค; RESULTS สรุปตัวเลข; PRESENTATION วิเคราะห์ภาพจริง ไม่ทำสองบทสรุปให้เป็นเนื้อหาซ้ำ
แต่ละขนาดมี 4 กรณีใช้เฟรมต้นฉบับและ saved RLE โดยไม่ inference เพิ่ม [เหตุผลเลือกกรณีปัจจุบัน](provenance/TIER_CASE_SELECTION_20261006_V2.md) บันทึกหนึ่งกรณีร่วมและสามกรณีตามพฤติกรรมของขนาดรวม 20 ตำแหน่งกรณีจาก 10 เฟรมต้นฉบับข้อมูลภาพซ้ำไม่ใช่ตัวอย่างอิสระเพิ่ม
[การตรวจกรณีชุดก่อน](provenance/CASE_DECISION_REVIEW_20261006.md) และเอกสารใน archive/provenance เป็นบันทึกย้อนหลัง เก็บภาษาตามต้นฉบับไม่แก้ประวัติเพื่อให้ภาษาเหมือนเอกสารปัจจุบัน Nano กรณี 4 มี mask ส่วนเกินทับ valid GT ที่ได้คู่แล้ว ไม่ใช่คนปลอมในฉากหลัง

## การตรวจและข้อควรระวัง

จาก workspace root ใช้ `.venv/bin/python YOLO_Instance_Segmentation_MOTS20_Scaling_Study/scripts/validate_documentation_redesign.py` ตรวจรูปแบบ ตาราง หลักฐานภาพและ hash หลัง push ใช้ `--remote-images` เพื่อตรวจ commit/blob/HTTP ของรูปบน GitHub
ตัวสร้างเอกสารและ validate_stage0.py เดิมเก็บเงื่อนไขย้อนหลัง เช่น NOT_RUN ของ Medium ห้ามรันตัวสร้างเหล่านี้ทับรายงานมาตรฐาน template หรือ study state ปัจจุบัน ใช้ validator ปัจจุบันและมาตรฐานภาษาเมื่อแก้เอกสาร
Small รอบ `benchmark-20261005T051531Z` และ Nano รอบ `benchmark-20261005T083307Z` มี 3 checkpoint × 2,862 เฟรมผ่าน maxDet200 และ 9 clean timing rounds ต่อขนาดการเพิ่มข้อมูลใน CSV ที่เดิมว่างไม่อนุญาตให้แก้ค่าที่วัดก่อนหน้า

## การรวมผลในอนาคต

เมื่อได้รับคำสั่งชัดเจนจึงนำเข้า 17 checkpoint ที่ไม่ซ้ำจาก CSV ของห้าขนาดตาม DATA_SCHEMA.md ผลที่วางแผน ได้แก่ EXECUTIVE_SUMMARY_TH.md, MEETING_SUMMARY_TH.md, RESEARCH_INSIGHTS_TH.md, MASTER_RESULTS.md, METRIC_GUIDE_TH.md; ตัวชี้วัด/MASTER_17_MODELS.csv, TIER_WINNERS.csv, SCALING_DELTAS.csv, PARETO_FRONTIER.csv, PARETO_VRAM.csv, PER_SEQUENCE_MASTER.csv; plots และ provenance/SOURCE_MANIFEST.json ตอนนี้ยังไม่สร้างผลเหล่านี้และไม่มีผลรวมบางส่วนที่สมมติขึ้น
การเสร็จของขนาดหนึ่งหรือการแก้เอกสารไม่อนุญาตให้เริ่ม benchmark หรือการรวมผลใหม่โดยอัตโนมัติ
