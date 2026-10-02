# Study standard — Stage 0 / schema 1.0

## Purpose and scope

Study ID: `yolo_instance_segmentation_mots20_scaling`. Evaluate exactly 17 official pretrained YOLO instance-segmentation checkpoints on frame-level Person masks in MOTS20. No training, fine-tuning, adaptation or non-YOLO models. Master performs no inference. E/X and C/L denote largest/second-largest available variants, not equal capacity.

## Repository mapping and model membership

| Tier | Repository | Checkpoints in display order |
|---|---|---|
| Largest (X/E) | [YOLO_Large_Seg_MOTS20_Benchmark](https://github.com/folklazy/YOLO_Large_Seg_MOTS20_Benchmark) | `yolo26x-seg.pt`, `yolo11x-seg.pt`, `yolov9e-seg.pt`, `yolov8x-seg.pt` |
| Second-largest (L/C) | [YOLO_Second_Largest_Seg_MOTS20_Benchmark](https://github.com/folklazy/YOLO_Second_Largest_Seg_MOTS20_Benchmark) | `yolo26l-seg.pt`, `yolo11l-seg.pt`, `yolov9c-seg.pt`, `yolov8l-seg.pt` |
| Medium (M) | [YOLO_Medium_Seg_MOTS20_Benchmark](https://github.com/folklazy/YOLO_Medium_Seg_MOTS20_Benchmark) | `yolo26m-seg.pt`, `yolo11m-seg.pt`, `yolov8m-seg.pt` |
| Small (S) | [YOLO_Small_Seg_MOTS20_Benchmark](https://github.com/folklazy/YOLO_Small_Seg_MOTS20_Benchmark) | `yolo26s-seg.pt`, `yolo11s-seg.pt`, `yolov8s-seg.pt` |
| Nano (N) | [YOLO_Nano_Seg_MOTS20_Benchmark](https://github.com/folklazy/YOLO_Nano_Seg_MOTS20_Benchmark) | `yolo26n-seg.pt`, `yolo11n-seg.pt`, `yolov8n-seg.pt` |

Master: [YOLO_Instance_Segmentation_MOTS20_Scaling_Study](https://github.com/folklazy/YOLO_Instance_Segmentation_MOTS20_Scaling_Study). Resolve workspace from checkout location; all six repositories are siblings. Shared `datasets/`, `models/`, `.venv/` remain at workspace root. Invoke `.venv/bin/python` directly; never move the environment. Never invent YOLOv9 x/l/m/s/n segmentation checkpoints; only e and c are in this study.

## Evidence hierarchy

Frozen machine-readable measurement > frozen config/protocol > final metric CSV/JSON > final timing summary > provenance/manifest > REPORT > README. Preserve numeric strings from structured sources; never transcribe README values. Canonical CSV is a normalized interface to the cited measurement evidence. If unavailable, use empty/NA and explain; zero is never a missing-value placeholder. Preserve historical paths and all raw evidence. Archive substantially replaced Markdown under `reports/archive/<name>.pre-standardization-<date>.md` and record SHA256.

## Dataset

Only `datasets/MOTS/MOTS/train/`, ordered MOTS20-02, MOTS20-05, MOTS20-09, MOTS20-11 then ascending frame. Exactly 2,862 frames and 26,894 frame-level Person GT instances (not unique people). GT is `<sequence>/gt/gt.txt`, Person class 2, ignore class 10. `datasets/MOTSLabels/MOTSLabels/` is not an additional source. Never modify inputs or create splits. Reuse the exact frozen image, timing, preflight and visualization manifests. Verify hashes, ordering, dimensions and metadata; shared validation once per unchanged dataset is sufficient. Evidence: `provenance/DATASET_VALIDATION.json`; baseline ordered per-image hashes remain in Largest frozen inputs. Human tier reports normally show Dataset compatibility: PASS.

## Frozen preprocessing and inference

Reuse `YOLO_Large_Seg_MOTS20_Benchmark/reports/benchmark-20260929T0520Z/frozen_inputs/src/benchmark_adapter.py` and its frozen pilot adapter. Do not independently rewrite preprocessing. OpenCV BGR input; aspect-preserving linear resize; square centered letterbox with padding 114, auto=False, scaleup=True, no stretch; RGB BCHW FP32 /255; shape 1×3×640×640. `retina_masks=True` restores native-resolution masks, binary masks and empty-mask filtering; preserve lossless GPU bit-packing/CPU transfer. Implementation SHA256 and framework-source fingerprint file are in `provenance/BASELINE_REFERENCE.json` (machine-readable fingerprint authority).

CUDA:0 respecting visibility, batch 1, imgsz 640, FP32, augment=False, rect=False, retina_masks=True, agnostic_nms=False. YOLO26 `nms=True` selects the one-to-many path before fusion; assert backend end2end=False. Person model class 0 (`person`). AP confidence floor 0.001 (strict > in framework NMS), fixed confidence >=0.25, NMS box IoU 0.70, model max_det=1000. Stop on cap saturation or NMS truncation warning. Framework outputs must use a fresh directory inside the owning experiment. No package upgrades, model downloads or environment changes without authorization. Frozen environment: Python 3.12.3, Ultralytics 8.4.160, torch 2.14.0+cu130, torchvision 0.29.0+cu130, NumPy 2.5.3, pycocotools 2.0.11, CUDA runtime 13.0, Tesla T4, driver 580.178.04. Preserve baseline; differences require an explicit compatibility decision.

## Frozen evaluator and AP

Reuse frozen `metrics.py` and `mots.py`; hashes in BASELINE_REFERENCE.json and per-tier STANDARDIZATION.json. Regression suite must pass before future benchmark execution. Fixed matching: descending confidence, stable ties, greedy one-to-one mask IoU >=0.50; best IoU then GT object ID. Match valid Person GT first. Only unmatched prediction with intersection against union(class-10 masks)/prediction area >=0.50 is ignored; otherwise FP. Do not clip Person masks. P/R/F1 are pooled micro counts, TP IoU/Dice conditional means. AP is Person-only frame-level COCO-style segmentation, IoU 0.50:0.05:0.95, 101 recall points, valid-GT priority and frozen ignore semantics. Overall AP is pooled, never mean sequence AP. No tracking metrics.

AP maxDet=200, distinct from model max_det=1000. Preflight caps 100/200/300/1000 on the exact frozen 100 frames; retain complete sensitivity results. Largest selected 200 as smallest common cap with absolute AP50/AP75/mAP difference from 1000 strictly <0.0001. Future tiers retain 200 even if 100 converges. If 200 fails, STOP for a common-policy decision; do not change it for one tier.

## Timing and GPU contamination

Use exact ordered frozen `manifests/timing_frames.json`: 25 predetermined frames per sequence, 100 unique frames. Ten untimed warmups cycling first ten subset frames, then three clean rounds (300 observations/model), seed 20260929. Preserve seeded/cyclic family execution ordering from runner; display order is independent. One model at a time; clear old model/cache before loading. Idle gate outside measurement; synchronize CUDA at stage boundaries. Measure preprocessing/H2D, forward inference, postprocessing (NMS/native masks/bit packing/CPU transfer/prediction objects), pipeline=sum of these; RLE preparation separately. Exclude model loading, disk/decode I/O, GT, metrics, visualization and output serialization. Reset allocator peak after warmup, include resident model; report max allocated peak across accepted runs. FPS=1000/mean pipeline ms; population std, median/P50, P95 across all clean observations. Model loading is a separate mean in seconds.

Monitor GPU before/during/after. Any other compute PID or pre-load idle utilization >5% contaminates the entire run; preserve and exclude it. Repeat unchanged only when idle, using fresh records. Never rank incomplete timing. Largest primary timing is exclusively `timing/benchmark-20260929T0520Z/clean_repetition/`; initial pass remains excluded. Second-largest also uses `clean_repetition/`. Pipeline FPS is not end-to-end mask-saving/CCTV throughput.

## Canonical public interface

Each tier: README.md, REPORT.md, RESULTS_SUMMARY_TH.md, PRESENTATION_SUMMARY_TH.md, EXPERIMENT_PROTOCOL.md; `configs/`; `manifests/STANDARDIZATION.json`, `manifests/environment.json`; `metrics/TIER_RESULTS.csv`, PER_SEQUENCE_RESULTS.csv, TIMING_SUMMARY.csv, MODEL_COMPLEXITY.csv, PREFLIGHT_MAXDET.csv. Create `outputs/plots`, `outputs/visualizations`, `reports/archive`, `predictions`, `timing`, `logs`, `src` only when used. Existing valid run-scoped layouts stay intact; expose interfaces instead of moving large artifacts. Future tiers initially have header-only CSVs and explicitly NOT_RUN documents, no fabricated result rows.

Schema and units: DATA_SCHEMA.md and schemas.json. Provenance includes study/schema/repo/tier, run IDs, source paths+SHA256, evaluator/config/protocol/dataset hashes, timestamp, inference_rerun, metrics_recalculated, archived documents and notes. Keep environment evidence; baseline reference is not a future run environment capture. Completion (PENDING/READY/RUNNING/COMPLETE/BLOCKED) is separate from result (NOT_RUN/PASS/PASS_WITH_WARNINGS/FAIL/UNKNOWN). Never infer completion from folder presence. Do not edit actively written files.

## Reports, rounding and plots

Use exact major section/table ordering in `templates/`. README is navigation; RESULTS_SUMMARY_TH concise Thai result; PRESENTATION_SUMMARY_TH mentor-ready Thai discussion; REPORT compact technical record. Avoid repeating methodology and generic definitions; common method is METHODOLOGY_REFERENCE.md. Full metric guide and final synthesis come later. Display order: YOLO26, YOLO11, YOLOv9 (only largest/second-largest), YOLOv8. Explicitly metric-sorted tables may differ. Machine tier labels: largest, second_largest, medium, small, nano. Human labels: Largest (X/E), Second-largest (L/C), Medium (M), Small (S), Nano (N).

Markdown AP/P/R/F1/IoU/Dice: 6 decimals; latency/FPS: 3; VRAM MiB: 2; parameters: comma-separated integer; GFLOPs: 3; checkpoint decimal MB: 2. CSV preserves source precision. Tables must be generated/verified from canonical CSV. Include identical Study Navigation. Standard plots under outputs/plots: 01_mask_map50_95.png, 02_ap50_ap75.png, 03_inference_latency.png, 04_pipeline_fps.png, 05_peak_vram.png, 06_accuracy_vs_latency.png. Preserve old plots; regenerate only presentation from saved metrics. Fixed model order/colors, explicit units, no weighted score.

## Scientific interpretation

Rank accuracy by Mask mAP50-95; show AP75, recall, inference/pipeline/FPS/VRAM winners separately. Describe tiny gaps as near-tied descriptively; no significance without valid analysis. Distinguish observation from interpretation; architecture speculation is not causality. Different capacities/pretraining confound causal conclusions. MOTS20 supports measured frame-level segmentation, speed and VRAM here; it does not establish blur, low-light, angle, explicit occlusion-severity robustness or deployment suitability. Say candidate for later CCTV robustness evaluation. TP-only quality excludes missed detections. Do not build 17-model conclusions until all tiers are validated.

## Execution and Git gates

On each request read this standard, STUDY_STATE.json and the requested tier; audit status/remotes/current branch and active processes before changes. Correct remote must be existing folklazy repository. Preserve unrelated user changes and archive replaced docs first. No deletion, force push, history rewrite, dataset/checkpoint/venv/secret commits or blind adds of predictions. Commit each repository independently after validation; review status, diff and exact staged paths. Push only to correct existing remote/current branch; record auth failure independently of scientific status.

For an explicitly authorized future tier: verify environment/imports/GPU without modifying working packages; resolve any authorized checkpoint acquisition; reuse baseline implementation and frozen lists; freeze new tier config/protocol/provenance; execute only that tier's preflight, accuracy and clean timing after gates; verify complete coverage and consistency; produce canonical artifacts/reports/plots; update state and commit. Separate execution authorization and download constraints still apply. No automatic resume or overwriting runs.

## STOP gate

Stage 0 authorizes setup and historical conversion only: no M/S/N downloads, preflight, inference, accuracy or speed benchmark; no final Master synthesis. Execute one tier per explicit request. Completion/readiness NEVER authorizes the next tier. After Stage 0 STOP and wait for explicit user instruction. Master later imports validated canonical results and performs no inference.
