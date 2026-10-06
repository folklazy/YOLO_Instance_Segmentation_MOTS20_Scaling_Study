# มาตรฐานการศึกษา YOLO Instance Segmentation บน MOTS20 ตามขนาดโมเดล

## ขอบเขต

การศึกษาที่ควบคุมเงื่อนไขนี้ประเมิน 17 checkpoint ของ YOLO instance segmentation แบบ pretrained ทางการบน Person mask รายเฟรมใน MOTS20 ไม่มีการฝึก ปรับจูน ปรับตัวตามข้อมูลหรือโมเดลตระกูลอื่น Master ไม่ inference รุ่น E/X และ C/L คือรุ่นใหญ่/รองใหญ่ที่มีให้ใช้ มิได้มีความจุเท่ากัน

## repository และสมาชิกโมเดล

| ขนาด | Repository | Checkpoint ตามลำดับแสดงผล |
|---|---|---|
| Largest (X/E) | [YOLO_Large_Seg_MOTS20_Benchmark](https://github.com/folklazy/YOLO_Large_Seg_MOTS20_Benchmark) | `yolo26x-seg.pt`, `yolo11x-seg.pt`, `yolov9e-seg.pt`, `yolov8x-seg.pt` |
| Second-largest (L/C) | [YOLO_Second_Largest_Seg_MOTS20_Benchmark](https://github.com/folklazy/YOLO_Second_Largest_Seg_MOTS20_Benchmark) | `yolo26l-seg.pt`, `yolo11l-seg.pt`, `yolov9c-seg.pt`, `yolov8l-seg.pt` |
| Medium (M) | [YOLO_Medium_Seg_MOTS20_Benchmark](https://github.com/folklazy/YOLO_Medium_Seg_MOTS20_Benchmark) | `yolo26m-seg.pt`, `yolo11m-seg.pt`, `yolov8m-seg.pt` |
| Small (S) | [YOLO_Small_Seg_MOTS20_Benchmark](https://github.com/folklazy/YOLO_Small_Seg_MOTS20_Benchmark) | `yolo26s-seg.pt`, `yolo11s-seg.pt`, `yolov8s-seg.pt` |
| Nano (N) | [YOLO_Nano_Seg_MOTS20_Benchmark](https://github.com/folklazy/YOLO_Nano_Seg_MOTS20_Benchmark) | `yolo26n-seg.pt`, `yolo11n-seg.pt`, `yolov8n-seg.pt` |

Master คือ [YOLO_Instance_Segmentation_MOTS20_Scaling_Study](https://github.com/folklazy/YOLO_Instance_Segmentation_MOTS20_Scaling_Study) ทั้งหก repository อยู่ระดับเดียวกัน หา workspace จาก checkout ไม่ hard-code home ข้อมูล models และ `.venv` ใช้ร่วมจาก workspace เรียก `.venv/bin/python` โดยตรง ห้ามย้าย environment ห้ามสมมติ YOLOv9 x/l/m/s/n segmentation; การศึกษานี้มีเฉพาะ e/c

## ลำดับอำนาจหลักฐาน

ค่าที่วัดและตรึงไว้ > config/protocol ที่ตรึง > CSV/JSON ตัวชี้วัด > แหล่งสรุปเวลา > provenance/manifest > REPORT > README เก็บตัวเลขตามแหล่ง structured เดิม ไม่คัดจาก README ตารางมาตรฐานเป็นข้อมูลที่จัดรูปจากหลักฐานเดิม หากไม่มีค่าให้ว่าง/NA พร้อมเหตุผล ไม่ใช้ศูนย์แทน
เก็บหลักฐานและ path ย้อนหลัง ไม่ย้ายข้อมูลรันเพียงเพื่อแก้ path สำรอง Markdown ก่อนแทนเนื้อหาไว้ใน reports/archive พร้อม SHA256 เมื่อแปลเอกสารโพรโทคอลให้เก็บต้นฉบับที่ตรึงไว้และตรวจ hash จากฉบับนั้น ไม่ถือการเปลี่ยนภาษาเป็นการเปลี่ยนค่าที่ใช้ทดลอง

## ข้อมูล

ใช้เฉพาะ `datasets/MOTS/MOTS/train/` ลำดับ MOTS20-02, MOTS20-05, MOTS20-09, MOTS20-11 แล้วเฟรมเรียงขึ้น รวม 2,862 เฟรมและ Person GT รายเฟรม 26,894 instances มิใช่คนไม่ซ้ำ GT อยู่ใน `<sequence>/gt/gt.txt`: Person class 2, ignore class 10 `datasets/MOTSLabels/MOTSLabels/` ไม่ใช่แหล่งเพิ่ม ห้ามแก้ข้อมูลหรือสร้าง split
ใช้ image/preflight/timing/visualization manifests ที่ตรึงเดิม ตรวจ hash ลำดับ มิติและ metadata; shared validation ครั้งเดียวพอเมื่อข้อมูลไม่เปลี่ยน หลักฐาน DATASET_VALIDATION.json และ hash รายภาพใน Largest frozen inputs รายงานระบุความสอดคล้องข้อมูล PASS เมื่อผ่าน

## การเตรียมภาพและ inference ที่ตรึง

ใช้ `YOLO_Large_Seg_MOTS20_Benchmark/reports/benchmark-20260929T0520Z/frozen_inputs/src/benchmark_adapter.py` และ frozen pilot adapter ไม่เขียน preprocessing ใหม่ OpenCV BGR ปรับขนาดเชิงเส้นรักษาสัดส่วน เติม letterbox สี่เหลี่ยมตรงกลางด้วย 114, auto=False, scaleup=True ไม่ยืดภาพ แปลง RGB BCHW FP32 /255 รูปร่าง 1×3×640×640
retina_masks=True คืน mask ที่มิติต้นฉบับหลังแก้ letterbox ทำ binary และกรอง mask ว่าง รักษาการบีบอัด bit บน GPU และส่ง CPU แบบไม่สูญเสียข้อมูลตรวจ implementation SHA256/framework fingerprint ตาม provenance/BASELINE_REFERENCE.json ซึ่งเป็นแหล่งอำนาจของ fingerprint
CUDA:0 ตาม visibility เดิม, batch 1, imgsz 640, FP32, augment=False, rect=False, retina_masks=True, agnostic_nms=False YOLO26 ใช้ nms=True เลือก one-to-many ก่อน fusion และตรวจ backend end2end=False Person ของโมเดล class 0 (`person`)
AP confidence floor 0.001 (framework NMS ใช้ >), fixed confidence ≥0.25, NMS box IoU 0.70, model max_det=1000 หยุดเมื่อเต็มเพดานหรือมี NMS truncation เก็บผลลัพธ์ของ framework ในที่เก็บรอบใหม่ภายใน repository เจ้าของ ห้ามเปลี่ยน package/download/environment โดยไม่ได้รับอนุญาต
สภาพแวดล้อมที่ตรึง: Python 3.12.3, Ultralytics 8.4.160, torch 2.14.0+cu130, torchvision 0.29.0+cu130, NumPy 2.5.3, pycocotools 2.0.11, CUDA runtime 13.0, Tesla T4, driver 580.178.04 เก็บ baseline เดิม หากต่างต้องมีการตัดสินความเข้ากันได้ชัดเจน

## ตัวประเมินและ AP ที่ตรึง

ใช้ตัวชี้วัด.py และ mots.py เดิมตาม hash ใน BASELINE_REFERENCE.json/STANDARDIZATION.json การทดสอบ regression ต้องผ่านก่อนการรันครั้งใหม่ จับคู่ confidence ลงโดย stable ties แบบ greedy one-to-one ที่ mask IoU ≥0.50 ใช้ IoU สูงสุดแล้ว GT object ID เมื่อต้องตัดสินเสมอ ให้ valid Person GT ก่อน
prediction ที่ไม่มีคู่เท่านั้นจึงตรวจ intersection กับ union class10 เทียบพื้นที่ prediction; ≥0.50 ให้ ignore มิฉะนั้น FP ไม่ตัด Person mask P/R/F1 รวมเป็น micro จำนวน; TP IoU/Dice เป็นค่าเฉลี่ยเฉพาะคู่ที่ผ่าน
AP เป็น Person-only COCO-style segmentation รายเฟรมใช้ IoU 0.50:0.05:0.95, 101 จุด Recall, valid-GT สิ่งที่ให้ความสำคัญและ ignore semantics เดิม AP รวมจากข้อมูลทั้งหมด ไม่ใช่ค่าเฉลี่ย AP ราย sequence ไม่มี tracking ตัวชี้วัด
AP maxDet=200 แยกจาก model max_det=1000 preflight ใช้ 100 เฟรมเดิมและ cap 100/200/300/1000 เก็บผลทั้งหมด Largest เลือก 200 เป็น common cap ต่ำสุดที่ความต่างสัมบูรณ์ AP50/AP75/mAP จาก 1000 <0.0001 ทุกโมเดลขนาดต่อไปคง 200 แม้ 100 จะผ่าน หาก 200 ไม่ผ่านให้หยุดเพื่อกำหนดนโยบายร่วม ไม่เปลี่ยน cap เฉพาะขนาด

## การวัดเวลาและการรบกวน GPU

ใช้ manifests/timing_frames.json เดิม 25 เฟรมต่อ sequence รวม 100 เฟรมไม่ซ้ำ ทำ 10 warmups จาก 10 เฟรมแรก แล้ววัด 3 รอบที่ไม่ถูกรบกวน (300 observations/โมเดล), seed 20260929 รักษาลำดับตระกูล seeded/cyclic ใน runner แยกจากลำดับแสดงผล
ใช้โมเดลทีละตัว ปล่อยโมเดลเก่าและ cache ก่อนโหลด ตรวจ idle นอกช่วงวัด synchronize CUDA ที่ขอบ stage preprocessing รวม transforms/H2D; inference คือ forward; postprocessing รวม NMS/native masks/bit packing/CPU transfer/prediction objects; pipeline คือผลรวม RLE preparation วัดแยก
ไม่รวมโหลดโมเดล อ่าน/decode ดิสก์ GT ตัวชี้วัดภาพประกอบและ serialization รีเซ็ต allocator peak หลัง warmup รวม resident model รายงาน allocated peak สูงสุดจากรอบที่ยอมรับ FPS=1000/ค่าเฉลี่ย pipeline ms; population std, median/P50, P95 จาก observations สะอาดทั้งหมด เวลาโหลดเป็นค่าเฉลี่ยวินาทีแยก
ตรวจ GPU ก่อน/ระหว่าง/หลัง หากมี compute PID อื่น หรือ idle utilization ก่อนโหลด >5% ถือว่าทั้งรอบถูกรบกวน เก็บและตัดออก ทำซ้ำด้วยค่าเดิมเฉพาะเมื่อ idle ในที่เก็บใหม่ ไม่จัดอันดับเวลาที่ยังไม่ครบ
Largest ใช้เฉพาะ timing/benchmark-20260929T0520Z/clean_repetition/ เป็นผลหลัก รอบแรกคงไว้แต่ไม่ใช้ Second-largest ใช้ clean_repetition/ เช่นกัน Pipeline FPS ไม่ใช่อัตราบันทึก mask หรือ CCTV ครบกระบวนการ

## ไฟล์มาตรฐานสาธารณะ

แต่ละขนาดมี README.md, REPORT.md, RESULTS_SUMMARY_TH.md, PRESENTATION_SUMMARY_TH.md, EXPERIMENT_PROTOCOL.md; configs; manifests/STANDARDIZATION.json และ environment.json; ตัวชี้วัด/TIER_RESULTS.csv, PER_SEQUENCE_RESULTS.csv, TIMING_SUMMARY.csv, MODEL_COMPLEXITY.csv, PREFLIGHT_MAXDET.csv
สร้างผลลัพธ์/plots, ผลลัพธ์/visualizations, reports/archive, predictions, timing, logs, src เฉพาะเมื่อใช้ เก็บ layout ของรันเดิมที่ถูกต้อง ไม่ย้ายไฟล์ใหญ่ขนาดที่ยังไม่รันมี CSV เฉพาะ header และเอกสาร NOT_RUN ไม่สมมติค่าผล
schema/หน่วยอยู่ใน DATA_SCHEMA.md และ schemas.json provenance เก็บ study/schema/repo/ขนาด, run IDs, ต้นทาง paths/SHA256, evaluator/config/protocol/ชุดข้อมูล hashes, เวลา, inference_rerun, metrics_recalculated, เอกสารที่สำรองและหมายเหตุ เก็บ environment ของรอบจริง ไม่ใช้ baseline แทนหลักฐานรอบใหม่
completion ใช้ PENDING/READY/RUNNING/COMPLETE/BLOCKED แยกจากผล NOT_RUN/PASS/PASS_WITH_WARNINGS/FAIL/UNKNOWN ห้ามอนุมาน completion จากโฟลเดอร์และไม่แก้ไฟล์ที่กำลังถูกเขียน

## ภาษา รูปแบบ และบทบาทเอกสาร

ใช้ [มาตรฐานภาษา](DOCUMENT_LANGUAGE_STANDARD.md) และ template ปัจจุบัน หัวข้อ/คำอธิบายเป็นภาษาไทย คงชื่อโมเดล metric คำสั่ง path schema fields และสถานะเครื่อง เอกสารชนิดเดียวกันใช้หัวข้อ/ตารางร่วม แต่ไม่บังคับหน้าที่ต่างกันให้มีเนื้อหาเหมือนกัน
README นำทาง; REPORT บันทึกเทคนิค; RESULTS สรุปเชิงตัวเลข; PRESENTATION วิเคราะห์ภาพและพฤติกรรม ไม่สร้างหัวข้อเฉพาะผู้ฟังรายบุคคล การสังเคราะห์สำหรับการประชุมอยู่ใน MEETING_SUMMARY_TH.md เมื่ออนุญาตแล้ว
ไม่ทำซ้ำ methodology/นิยามทั่วไปยาว ๆ ใช้ METHODOLOGY_REFERENCE.md คู่มือ metric เต็มและผลรวมทำภายหลัง เรียง YOLO26, YOLO11, YOLOv9 เฉพาะสองขนาดใหญ่, YOLOv8 ตารางที่จัดตาม metric ต้องระบุชัดขนาดเครื่องใช้ largest/second_largest/medium/small/nano; ชื่ออ่านใช้ Largest (X/E), Second-largest (L/C), Medium (M), Small (S), Nano (N)
Markdown AP/P/R/F1/IoU/Dice 6 ตำแหน่ง, เวลา/FPS 3, VRAM MiB 2, parameters จำนวนเต็มคั่นหลักพัน, GFLOPs 3, checkpoint MB 2 CSV คงความละเอียดเดิม ตรวจทุกตารางจาก CSV และใช้การนำทางร่วม
กราฟมาตรฐานผลลัพธ์/plots: 01_mask_map50_95.png, 02_ap50_ap75.png, 03_inference_latency.png, 04_pipeline_fps.png, 05_peak_vram.png, 06_accuracy_vs_latency.png เก็บกราฟเดิม สร้างใหม่ได้จาก metric ที่บันทึกไว้เท่านั้น สี/ลำดับโมเดลคงที่ หน่วยชัด ไม่ใช้คะแนนถ่วงน้ำหนัก

## บทสรุปเชิงตัวเลข

RESULTS_SUMMARY_TH.md ใช้ 5–8 ข้อ ตารางผลหลักเดียว บทสรุปจากตารางรายโมเดล ผู้ชนะ 6 ด้าน ข้อค้นพบจากค่าที่วัด 3–5 ข้อข้อแลกเปลี่ยนความแม่นยำกับเวลา/หน่วยความจำข้อควรระวังและข้อมูลสำหรับรวมต่อ
ส่วนสรุปตารางอยู่หลังตารางหลัก ใช้หัวข้อย่อยทุกโมเดลตามลำดับ 1–2 ย่อหน้าสั้นอธิบายจุดเด่น ข้อจำกัดข้อแลกเปลี่ยนจริงและตัวเลือกแบบมีเงื่อนไข ไม่เรียงทุก cell มาเล่าหรือประกาศผู้ชนะสมดุลโดยไม่มีหลักฐาน
AP50/TP-only ที่ไม่มีในตารางย่ออ้าง canonical CSV/REPORT; TP-only ใช้คู่ GT คนละชุดได้ ค่าใกล้เป็นเชิงพรรณนา แยก inference/pipeline เมื่อ NOT_RUN เก็บหัวข้อโมเดลที่วางแผนและข้อความรอผล ห้ามกรณีภาพ/ตารางซ้ำ ส่วนนี้ใช้ตามการแก้ข้อกำหนดผู้ใช้เมื่อ 2026-10-05 ที่อนุญาตสรุปเชิงตีความรายโมเดล

## ภาพและการวิเคราะห์เชิงคุณภาพ

PRESENTATION ตอบว่า prediction ในเฟรมจริงต่างกันอย่างไร ใช้ภาพรวมสั้นและตารางบริบท mAP/AP75/Recall เชื่อม RESULTS จากนั้น 3–5 กรณีเฟรมเดียวกันวิเคราะห์ข้อผิดพลาดตรวจคู่คะแนนใกล้ 3–6 ข้อสังเกต/การตีความ เชื่อมผลภาพกับตัวเลข ตารางสิ่งที่ให้ความสำคัญ/ตัวเลือก/หลักฐาน ข้อจำกัดและลิงก์
ทุกกรณีระบุเหตุผล ภาพ repository แบบ relative path ข้อสังเกตตรงจากภาพ วิเคราะห์ แล้วเชื่อม metric ไม่เขียนเรียงอันดับรายโมเดลซ้ำหรือหัวข้อเฉพาะผู้ฟัง
ใช้เฉพาะ MOTS20/GT/prediction/หลักฐานจริง ใช้ comparison เดิมก่อน แล้วจึงสร้างจาก saved predictions ถ้าสร้างไม่ได้ให้ระบุช่องว่างและรอ ไม่รัน inference เพื่อเอกสาร คัดจาก TP/FP/FN/matched IoU หรือ visualization manifest แล้วตรวจภาพจำนวนเล็ก ไม่ตรวจหลายพันเฟรม
เลือกข้อได้เปรียบข้อผิดพลาดร่วมกรณีสวนอันดับ/ข้อแลกเปลี่ยนและผลคล้าย/คะแนนใกล้ตามที่มี ระบุ pool, sequence/frame, เหตุผลและ hashes กรณีเป็นข้อมูลเพื่อวิเคราะห์ มิใช่ตัวแทนชุดข้อมูล
ใช้กรณีร่วมหนึ่งกรณีข้ามขนาดและกรณีแยกพฤติกรรมเฉพาะขนาดเมื่อมีประโยชน์ ภายในกรณีทุกโมเดลต้องใช้เฟรมเดียวกันข้ามขนาดไม่บังคับชุดเดียวกัน ภาพซ้ำต้องระบุเหตุผล ไม่เพิ่มจำนวนตัวอย่างอิสระแทนภาพควบคุมที่แยกพฤติกรรมได้น้อยเมื่อมีกรณีที่ดีกว่า แต่คงผลคล้ายและกรณีสวนอันดับไว้
เมื่อปรับชุดหลักฐาน เก็บไฟล์เก่าและเปิดชุดใหม่ผ่าน manifests/QUALITATIVE_SELECTION.json เก็บต้นทาง hashes, ROI coordinates, pool และเหตุผล ทุกกรณีระบุด้านที่ใช้เลือกและขอบเขตภาพควบคุมไม่จำเป็นต้องมีปัญหาหรือผู้ชนะ
หาก GT เล็กอ่านไม่ได้ เพิ่ม ROI จากเฟรมและ mask เดิม ใช้พิกัด/สัดส่วนภาพเดียวกันทุกโมเดลและคงภาพเต็มระบุพิกัดและ hashes การขยายไม่เพิ่มรายละเอียด ตรวจ per-GT การจับคู่/ตัวเลือก IoU เมื่อจำเป็นเพื่อแยก mask ที่ขาดจาก mask ไม่ผ่านเกณฑ์ ค่าเฉพาะกรณีไม่ใช่ metric ชุดข้อมูลห้ามเปลี่ยนตัวประเมิน/จำนวนให้ภาพดูได้เปรียบ ตรวจ extra masks ว่าทับ valid GT ที่มีคู่แล้วก่อนเรียก FP ฉากหลัง
Panel เรียงต้นฉบับ/GT, YOLO26, YOLO11, YOLOv9 e/c เมื่อมี, YOLOv8 ทุกโมเดลใช้เฟรมเต็ม/สัดส่วนภาพ/confidence/ignore policy เดียวกัน label/GT ID อ่านได้ เก็บภาพในผลลัพธ์/visualizations/qualitative/ ไม่เขียนทับเก่า เก็บ CASE_SELECTION.md และ manifest ขนาดเล็กใน Git
เพิ่มข้อยกเว้นชื่อภาพที่เลือกใน .gitignore และ commit พร้อมเอกสาร ภาพอื่น/เฟรมต้นฉบับ/prediction ยัง ignored ไม่คัดลอกภาพต้นฉบับซ้ำ ตรวจ tracking/PNG/hash และหลัง push ตรวจ remote commit/blob/HTTP มีไฟล์ในเครื่องอย่างเดียวไม่รับรองแสดงบน GitHub
ระบุข้อผิดพลาดที่เห็นจริงเท่านั้น FN คือ GT ไม่มีคู่ผ่านเกณฑ์ อาจมี mask ไม่ผ่าน IoU; FP คือ prediction ไม่มีคู่ไม่ใช่คนไม่มีจริงโดยอัตโนมัติ IGN ไม่ใช่ FP per-frame จำนวนต้องตรง record เดิม ห้ามสร้างค่าระดับชุดข้อมูลใหม่หรือเปลี่ยนผล ภาพเดียวไม่ยืนยันพฤติกรรม/นัยสำคัญทั้งชุดข้อมูลไม่อนุมานสภาพฉาก/ความทนทานโดยไม่มีหลักฐาน
คะแนนใกล้ควรมีภาพเฟรมเดียวกันเมื่อทำได้ แต่เวลา/VRAM อนุมานจากภาพไม่ได้ ไม่มีคะแนนถ่วงน้ำหนักหรือความเหนือกว่าสำหรับ CCTV ขั้นสุดท้าย วิเคราะห์เมื่อทุกโมเดลและเวลาผ่านแล้วเท่านั้นขนาดไม่ครบใช้หัวข้อ/ตารางว่างและรอผล ห้ามวิเคราะห์ตัวเดียว REPORT เชื่อม PRESENTATION ผ่านส่วนการวิเคราะห์เชิงคุณภาพสั้น ๆ

## การตีความทางวิทยาศาสตร์

จัด accuracy ตาม Mask mAP50-95 แสดงผู้ชนะ AP75/Recall/inference/pipeline/FPS/VRAM แยกกัน ค่าใกล้ไม่ใช่นัยสำคัญหากไม่มีการทดสอบ แยกข้อสังเกตจากการตีความ การคาดเดาโครงสร้างไม่ใช่เหตุและผล ความจุและการฝึกเดิมเป็นปัจจัยปนกัน
MOTS20 รองรับผล segmentation รายเฟรมเวลาและ VRAM ในเงื่อนไขนี้ ไม่ยืนยันความทนทานต่อภาพพร่าแสงน้อยมุมกล้องระดับการบังหรือความพร้อม deployment เป็นตัวเลือกไปทดสอบ CCTV ต่อ TP-only ไม่รวมการพลาด GT ห้ามสรุป 17 โมเดลก่อนทุกขนาดผ่านและได้รับคำสั่ง

## เงื่อนไข execution และ Git

แต่ละคำขออ่านมาตรฐานนี้ STUDY_STATE.json และขนาดที่เกี่ยวข้อง ตรวจ status/branch/remote/process ก่อนแก้ remote ต้องเป็น repository เดิมของ folklazy เก็บงานผู้ใช้อื่น สำรองเอกสารก่อนแทน ห้ามลบ force push เขียนประวัติใหม่ commit ชุดข้อมูล/checkpoint/venv/secret หรือเพิ่ม prediction ทั้งชุดโดยไม่ตรวจ
commit แยก repository หลังผ่าน ตรวจ diff และชื่อไฟล์ที่ stage แน่นอน push เฉพาะ remote/branch ถูกต้อง บันทึก auth failure แยกจากผลทางวิทยาศาสตร์
เมื่อได้รับคำสั่งให้รันขนาดใหม่ ตรวจ environment/import/GPU โดยไม่แก้ package จัดการ checkpoint เฉพาะที่อนุญาต ใช้ baseline/manifest เดิม ตรึง config/protocol/provenance ใหม่ รันเฉพาะขนาดที่ขอ preflight/accuracy/clean timing ตาม gate ตรวจความครบ/ค่าตรง สร้าง artifacts/เอกสาร/กราฟ อัปเดต state และ commit การรันไม่ยกเลิกข้อจำกัด download ห้ามกลับมารันหรือเขียนทับอัตโนมัติ

## เงื่อนไขหยุด

Stage 0 เดิมอนุญาต setup/แปลงข้อมูลย้อนหลังเท่านั้น ไม่อนุญาต download/preflight/inference/accuracy/timing ของ M/S/N หรือผล Master การรันภายหลังทำครั้งละขนาดตามคำสั่งชัดเจน การเสร็จ/พร้อมไม่อนุญาตขนาดถัดไป อนุมัติแยกตามการรันและหยุดเมื่อครบ Master ภายหลังนำเข้าผลที่ผ่านเท่านั้น ไม่ inference การปรับเอกสารครั้งนี้ไม่เริ่ม benchmark หรือการสังเคราะห์ผลใหม่
