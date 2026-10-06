# รูปแบบข้อมูลมาตรฐาน

`study_id = yolo_instance_segmentation_mots20_scaling`; `schema_version = 1.0` ใช้ CSV UTF-8 มี header เดียว คั่นด้วยจุลภาค ค่าว่างหรือ NA หมายถึงไม่มีข้อมูลห้ามใช้ศูนย์แทนค่าที่ขาด เก็บตัวเลขเต็มตามแหล่งวัดเดิม

## TIER_RESULTS.csv

```text
study_id,schema_version,experiment,tier,family,model,checkpoint,status,frames,gt_instances,mask_map50_95,ap50,ap75,precision,recall,f1,tp,fp,fn,tp_iou_mean,tp_dice_mean,ignored_predictions,inference_ms_mean,pipeline_ms_mean,fps,peak_allocated_vram_mib,parameters,gflops,checkpoint_mb,ap_maxdet,fixed_confidence,ap_confidence_floor,nms_iou,evaluation_iou,imgsz,precision_mode,device,run_id,source_artifact
```

## PER_SEQUENCE_RESULTS.csv

```text
study_id,schema_version,experiment,tier,family,model,sequence,frames,gt_instances,predictions,tp,fp,fn,precision,recall,f1,ap50,ap75,mask_map50_95,tp_iou_mean,tp_dice_mean,ignored_predictions
```

## TIMING_SUMMARY.csv

```text
study_id,schema_version,experiment,tier,family,model,stage,mean_ms,median_ms,std_ms,p50_ms,p95_ms,repetitions,measured_frames,contamination_status
```

## MODEL_COMPLEXITY.csv

```text
study_id,schema_version,experiment,tier,family,model,loaded_parameters,fused_parameters,gflops,checkpoint_mb,model_load_seconds
```

## PREFLIGHT_MAXDET.csv

```text
study_id,schema_version,experiment,tier,family,model,checkpoint,max_dets,ap50,ap75,map50_95,ignored_detections_at_ap50,ap50_abs_difference_from_1000,ap75_abs_difference_from_1000,map50_95_abs_difference_from_1000,converged
```

## ตัวระบุ หน่วย และข้อมูลที่ขาด

experiment คือชื่อ repository; ขนาดใช้ largest/second_largest/medium/small/nano; family ใช้ YOLO26/YOLO11/YOLOv9/YOLOv8; model เป็นชื่อแสดงผลและ checkpoint เป็นชื่อไฟล์ทางการ ลำดับโมเดลอยู่ใน STUDY_STANDARD.md; ลำดับภาพ 02/05/09/11; stage เวลาใช้ preprocessing/inference/postprocessing/pipeline/rle_preparation/ultralytics_postprocess_inclusive; preflight เรียง cap 100/200/300/1000
คีย์ต้องไม่ซ้ำ: ขนาด+model สำหรับผลรวม/complexity; ขนาด+model+sequence สำหรับรายลำดับภาพ; ขนาด+model+stage สำหรับเวลา; ขนาด+model+max_dets สำหรับ preflight
status ใช้ PASS/PASS_WITH_WARNINGS/FAIL/UNKNOWN/NOT_RUN แยกจากสถานะ completion ใน STUDY_STATE.json แถวมีเฉพาะผลที่เสร็จแล้ว ไฟล์มีแต่ header หมายถึง NOT_RUN; run_id ระบุรอบ accuracy เดิม; source_artifact เป็น path สัมพัทธ์คั่นด้วยอัฒภาคพร้อม hash ใน STANDARDIZATION.json
AP, Precision, Recall, F1, TP IoU/Dice, confidence และ IoU เป็นสัดส่วน 0–1 ไม่ใช่เปอร์เซ็นต์เฟรมคือจำนวนภาพ; gt_instances คือ Person annotations รายเฟรม; tp/fp/fn และ ignored_predictions ใช้ fixed-confidence การจับคู่
predictions รายลำดับภาพคือ predictions_at_confidence ก่อนกรอง ignore: predictions = tp+fp+ignored_predictions; gt_instances=tp+fn AP รวมไม่ใช่เฉลี่ย AP รายลำดับภาพ mask_map50_95 อ้าง map50_95 เดิม ส่วน TP-only อ้าง matched_iou_mean/matched_dice_mean ตามตัวแปลงใน scripts/stage0_standardize.py
เวลาเป็น ms/frame; fps=1000/ค่าเฉลี่ย pipeline ms; peak_allocated_vram_mib เป็น MiB (2^20 bytes) ใช้ allocated peak สูงสุดจากรอบที่ยอมรับและรวมโมเดลที่ resident; checkpoint_mb เป็น MB (10^6 bytes) parameters/loaded_parameters/fused_parameters เป็นจำนวนเต็ม; gflops เป็นค่าประเมินเส้นทาง NMS-forward ที่ imgsz640 ไม่ใช่จำนวน operations ที่วัดจริง; model_load_seconds เป็นค่าเฉลี่ยเวลาโหลดจากรอบสะอาด; imgsz เป็นพิกเซลเข้า network; precision_mode FP32; device CUDA:0 ตามการมองเห็นจริง
ค่าเฉลี่ย/median/std/P50/P95 คำนวณจาก 300 observations ไม่ใช่เฉลี่ย percentile แต่ละรอบ std ใช้ประชากร; repetitions=3, measured_frames=300 คือ 100 เฟรมทำซ้ำ; CLEAN หมายถึงยอมรับเฉพาะรอบไม่ถูกรบกวน ultralytics_postprocess_inclusive เป็นข้อมูลย่อยเพื่อวิเคราะห์ ห้ามบวกซ้ำใน pipeline RLE วัดแยกและเก็บสถิติเดิมโดยไม่คำนวณแทน
max_dets ใน preflight เป็น cap ของ AP ไม่ใช่ cap ของโมเดล mask mAP ใช้ชื่อ map50_95 เดิม ความต่างเทียบ cap1000 เป็นค่าสัมบูรณ์ converged ต้อง <0.0001 ทั้ง AP50/AP75/mAP เก็บครบทุก cap

## ความละเอียด ลำดับ และหลักฐาน

CSV เก็บความละเอียดจากต้นทางค่าที่ไม่มีต้องว่าง/NA พร้อมเหตุผลใน provenance Markdown ใช้ AP/P/R/F1/IoU/Dice 6 ตำแหน่ง, เวลา/FPS/GFLOPs 3, VRAM/MB 2 และพารามิเตอร์เป็นจำนวนเต็มคั่นหลักพัน เรียงสมาชิกตามมาตรฐานเว้นแต่ระบุว่าจัดเรียงตาม metric
ลำดับอำนาจหลักฐาน: ค่าที่วัดและตรึงไว้ > config/protocol ที่ตรึง > ผลตัวชี้วัด > แหล่งเวลา > provenance > REPORT > README ห้ามใช้ README เป็นต้นทางตัวเลข

## เงื่อนไขนำเข้าผลรวมในอนาคต

ไม่สร้างผล Master จากการปรับเอกสารนี้ การรวมผลภายหลังต้องได้รับคำสั่งผู้ใช้ มี 17 checkpoint ทางการที่ไม่ซ้ำ สมาชิกครบทุกขนาดและค่าครบ study/schema IDs ตรงกันต้นทาง hashes และโพรโทคอลสอดคล้อง ปฏิเสธแถวซ้ำ ขาด NOT_RUN/FAIL หรือการเปลี่ยน schema/หน่วยโดยไม่แจ้ง
PASS_WITH_WARNINGS นำเข้าได้เมื่อส่งคำเตือนไปด้วย คำนวณผู้ชนะและ Pareto โดยไม่ใช้คะแนนถ่วงน้ำหนัก ความต่างข้ามขนาดเปรียบเทียบภายในตระกูลเดียวกัน แยก AP รายลำดับภาพจาก pooled AP เก็บ revision, CSV/ต้นทาง hashes และ run IDs ใน provenance/SOURCE_MANIFEST.json ห้ามสรุปว่าผลเสร็จจากโฟลเดอร์ template หรือ checkpoint ที่มีอยู่
