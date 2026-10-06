# Canonical data interface

`study_id = yolo_instance_segmentation_mots20_scaling`; `schema_version = 1.0`. UTF-8 CSV, one header, comma delimiter; empty or NA means unavailable, never placeholder zero. Preserve numerical strings from authoritative measurements.

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

## Identity, units and missing data

experiment is the repository name; tier is largest/second_largest/medium/small/nano. family is YOLO26/YOLO11/YOLOv9/YOLOv8. model is display name (e.g. YOLOv9e-Seg); checkpoint is exact official filename. Display/order membership is in STUDY_STANDARD.md. Model order is fixed, sequences 02/05/09/11, timing stages preprocessing/inference/postprocessing/pipeline/rle_preparation/ultralytics_postprocess_inclusive. Sort preflight caps 100/200/300/1000 within model. Unique keys: tier+model (overall/complexity), tier+model+sequence, tier+model+stage, tier+model+max_dets.

status uses scientific result state PASS/PASS_WITH_WARNINGS/FAIL/UNKNOWN/NOT_RUN. completion is separate in STUDY_STATE.json. Only completed measurements are rows; header-only files mean NOT_RUN. run_id is the preserved accuracy run ID; source_artifact is a semicolon-delimited list of repository-relative authoritative paths, with hashes in STANDARDIZATION.json.

AP, precision, recall, F1, TP IoU/Dice, confidences and IoU thresholds are fractions on 0–1 scale, never percentages. frames counts evaluated images; gt_instances counts Person frame annotations; tp/fp/fn and ignored_predictions use fixed-confidence matching. Per-sequence predictions means predictions_at_confidence before ignore filtering, so predictions = tp+fp+ignored_predictions. gt_instances=tp+fn. Overall AP is pooled, not mean sequence AP. Mask mAP50-95 aliases historical map50_95; TP means alias matched_iou_mean/matched_dice_mean. Source mappings are executable in scripts/stage0_standardize.py.

Latency is ms/frame; fps is 1000/mean pipeline ms. peak_allocated_vram_mib uses binary MiB (2^20 bytes), max accepted round peak including resident model. checkpoint_mb is decimal MB (10^6 bytes); parameters/loaded_parameters/fused_parameters are integer counts; gflops is NMS-forward estimate at imgsz640, not measured operations. model_load_seconds is arithmetic mean across clean rounds. imgsz is network pixels; precision_mode FP32; device CUDA:0 (logical visible device).

Timing mean/median/std/P50/P95 are across 300 measured observations, not averages of per-round summary percentiles; std is population std. repetitions=3 and measured_frames=300 (100 unique frames repeated); CLEAN means accepted uncontaminated rounds only. `ultralytics_postprocess_inclusive` is a historical diagnostic subset and must not be added to pipeline again. RLE is separate. Preserve source summary strings without recomputing stats.

Preflight max_dets is AP cap; mask mAP field retains historical name map50_95. Difference fields are absolute differences from cap1000; converged uses strict <0.0001 on all three AP fields. Complete sensitivity rows are retained.

## Precision, ordering and evidence

Machine CSV: preserve source precision; missing values empty/NA with reason in provenance. Markdown: AP/P/R/F1/IoU/Dice 6 decimals, latency/FPS 3, VRAM 2, parameters comma integer, GFLOPs 3, checkpoint MB 2. Use fixed membership order everywhere unless explicitly metric-sorted. Frozen structured measurement > frozen config/protocol > final metrics > timing > provenance > REPORT > README. Never use README as primary numeric evidence.

## Future master import contract

Do not generate master outputs during Stage 0. Later require exactly 17 unique approved checkpoints, expected tier membership, complete result coverage, matching study/schema IDs, source hashes, protocol compatibility and explicit synthesis authorization. Reject duplicates, missing/not-run/fail rows and silent schema/unit changes. PASS_WITH_WARNINGS may import with warnings carried forward. Compute tier winners and Pareto fronts without weighted scores; paired scaling deltas within family only. Keep per-sequence AP separate from pooled AP. Record each repository revision, CSV/source hashes and canonical run IDs in provenance/SOURCE_MANIFEST.json. Never infer result rows from templates or checkpoint presence.
