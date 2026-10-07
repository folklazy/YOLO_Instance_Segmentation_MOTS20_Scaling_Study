# Study standard — frozen benchmark / schema 1.0

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

Per-tier PRESENTATION_SUMMARY_TH.md must remain a neutral, presentation-ready tier summary. Meeting-specific synthesis belongs only in `MEETING_SUMMARY_TH.md`.

Use exact major section/table ordering in `templates/`. README is navigation; RESULTS_SUMMARY_TH concise Thai result; PRESENTATION_SUMMARY_TH visual + qualitative + interpretive Thai analysis; REPORT compact technical record. Avoid repeating methodology and generic definitions; common method is METHODOLOGY_REFERENCE.md. Full metric guide and final synthesis come later. Display order: YOLO26, YOLO11, YOLOv9 (only largest/second-largest), YOLOv8. Explicitly metric-sorted tables may differ. Machine tier labels: largest, second_largest, medium, small, nano. Human labels: Largest (X/E), Second-largest (L/C), Medium (M), Small (S), Nano (N).

Markdown AP/P/R/F1/IoU/Dice: 6 decimals; latency/FPS: 3; VRAM MiB: 2; parameters: comma-separated integer; GFLOPs: 3; checkpoint decimal MB: 2. CSV preserves source precision. Tables must be generated/verified from canonical CSV. Include identical Study Navigation. Standard plots under outputs/plots: 01_mask_map50_95.png, 02_ap50_ap75.png, 03_inference_latency.png, 04_pipeline_fps.png, 05_peak_vram.png, 06_accuracy_vs_latency.png. Preserve old plots; regenerate only presentation from saved metrics. Fixed model order/colors, explicit units, no weighted score.

## Documentation roles and qualitative evidence

RESULTS_SUMMARY_TH.md is a compact quantitative result summary. Follow its template:
5–8 bullets; one canonical results table; per-model table interpretation; six-category winner table; 3–5 directly measured findings;
short Accuracy vs Speed and Accuracy vs Memory sections; cautions; cross-tier inputs/links.
Include a concise per-model table interpretation section after the main table: one subsection per valid model
in fixed family order, 1–2 short paragraphs explaining measured strengths, limitations, actual
trade-offs and conditional candidates. Do not merely restate every table cell or assume a balanced
winner. Cite canonical CSV/REPORT for AP50/TP-only metrics omitted from the compact table;
TP-only quality is conditional on matching and can use different GT subsets across models.
Keep tiny gaps descriptive and separate inference from pipeline. NOT_RUN tiers retain planned
model headings and pending text without fabricated comparisons. No repeated result tables or
visual cases; PRESENTATION retains the visual/qualitative role. This section implements the user's
2026-10-05 revision of the earlier prohibition on per-model interpretation.

PRESENTATION_SUMMARY_TH.md answers how predictions differ in actual frames. Follow its template:
one short overview and small mAP/AP75/Recall context table linked to RESULTS; approximately 3–5
same-frame cases; Failure Analysis; Near-tie visual check; 3–6 explicit observation/interpretation
findings; quantitative/qualitative synthesis; priority/candidate/evidence table; limitations and links.
Every case gives selection reason, actual relative image embed, direct observations, analysis,
then connection to canonical metrics. No per-model ranking essay or mentor-specific section.

Use only actual MOTS20 benchmark frames, original GT, saved predictions and canonical artifacts.
Prefer existing comparisons, then reconstruct from saved predictions; if insufficient, report that
reconstruction requires inference and leave analysis pending. Never rerun inference for docs.
Shortlist from structured TP/FP/FN/matched-IoU evidence or the frozen visualization manifest;
inspect a small set, not thousands of frames. Include advantage, shared failure, counterexample/
detection trade-off and similar-output/near-tie cases where available. Document selection pool,
sequence, frame, reasons and source hashes; selected cases are diagnostic, not representative sampling.
Use one shared cross-tier anchor plus tier-specific diagnostic cases where useful. Within a case,
all models must use the same frame; across tiers, do not force an identical case set. Repeated
diagnostic frames require a stated behavior/decision reason. Replace nondiscriminating controls
when a better saved-output case exists while retaining similar-coverage and counterexamples.
Count distinct source frames separately from case slots; shared frames are not independent samples.
When revising existing evidence, preserve old files and activate the new version through
manifests/QUALITATIVE_SELECTION.json; keep source hashes, ROI coordinates, selection pool and reasons.

State each case's decision use and evidence limits explicitly: discriminating example, shared failure,
counterexample/trade-off, or similar-output control. A control need not show a pain point or a winner.
If small GT regions are unreadable, add a supplementary ROI from original frames and saved masks,
with identical coordinates/scale for every model and the full comparison retained visibly. Document
ROI coordinates and source hashes; enlargement does not add source detail. Where needed, inspect
individual-GT matching or candidate IoU to distinguish a missing mask from a below-threshold mask.
Label these diagnostics separately from canonical dataset metrics; never change the evaluator or
counts to make a visual result more persuasive. Explicitly verify whether extra masks overlap valid
GT already assigned a match before calling them background false positives.

All tier models use the same original frame, full image region, scale, confidence and ignore policy.
Panel order: Original/GT, YOLO26, YOLO11, valid YOLOv9 e/c when applicable, YOLOv8.
Clear labels and GT IDs should support observations. Store new composites under the owning tier's
outputs/visualizations/qualitative/case_01_comparison.png through case_05_comparison.png when useful;
never overwrite existing generated artifacts. Keep CASE_SELECTION.md and small source/evidence manifests
versionable. Publish the selected comparison images embedded in canonical Markdown as intentionally
retained analysis evidence: add exact filename exceptions in the owning repository's .gitignore and
commit them with the document. Other generated images, original frames and predictions remain ignored.
Do not duplicate source images. Validate Git tracking, PNG integrity, unchanged source hashes and,
after push, remote commit identity, image blob hashes and actual raw-image HTTP responses. Local file
existence alone does not establish that an image renders on GitHub.

Use observed failure categories only. FN is unmatched GT under mask IoU policy and may include an
inaccurate mask; FP is an unmatched prediction, not automatically a nonexistent person. Ignore outputs
must not be called false positives. Per-frame counts must match existing canonical records; no new
dataset-level measurements or changed benchmark values. A selected frame does not prove dataset-wide
behavior or significance. Do not infer scene conditions/robustness without visible evidence.
Numerical near ties require a same-frame visual check where practical; system latency/VRAM cannot
be inferred from mask images. No weighted score or final CCTV superiority.

Analyze completed tiers only after all models and clean timing are validated. Future/incomplete M/S/N
use empty tables and explicit pending text, never a single-model qualitative comparison.
REPORT.md remains technical and links to PRESENTATION through a short Qualitative Analysis section.

## Scientific interpretation

Rank accuracy by Mask mAP50-95; show AP75, recall, inference/pipeline/FPS/VRAM winners separately. Describe tiny gaps as near-tied descriptively; no significance without valid analysis. Distinguish observation from interpretation; architecture speculation is not causality. Different capacities/pretraining confound causal conclusions. MOTS20 supports measured frame-level segmentation, speed and VRAM here; it does not establish blur, low-light, angle, explicit occlusion-severity robustness or deployment suitability. Say candidate for later CCTV robustness evaluation. TP-only quality excludes missed detections. Do not build 17-model conclusions until all tiers are validated.

## Execution and Git gates

On each request read this standard, STUDY_STATE.json and the requested tier; audit status/remotes/current branch and active processes before changes. Correct remote must be existing folklazy repository. Preserve unrelated user changes and archive replaced docs first. No deletion, force push, history rewrite, dataset/checkpoint/venv/secret commits or blind adds of predictions. Commit each repository independently after validation; review status, diff and exact staged paths. Push only to correct existing remote/current branch; record auth failure independently of scientific status.

For an explicitly authorized future tier: verify environment/imports/GPU without modifying working packages; resolve any authorized checkpoint acquisition; reuse baseline implementation and frozen lists; freeze new tier config/protocol/provenance; execute only that tier's preflight, accuracy and clean timing after gates; verify complete coverage and consistency; produce canonical artifacts/reports/plots; update state and commit. Separate execution authorization and download constraints still apply. No automatic resume or overwriting runs.

## STOP gate

Stage 0 authorized setup and historical conversion only. Tier execution and final synthesis each require explicit user instruction; completion/readiness never authorizes a new stage. The user has now explicitly authorized the 17-model Master synthesis from all five completed, compatible tiers. Master imports validated canonical results and performs no inference, benchmark reruns, training or adaptation. After validated synthesis, STOP.

## Documentation language

Follow [DOCUMENTATION_LANGUAGE_POLICY.md](DOCUMENTATION_LANGUAGE_POLICY.md): English technical records and navigation, Thai prose in the two _TH summaries. Use the same language roles and section structure for every tier. Historical and frozen records retain their original language. This editorial rule changes no benchmark settings.

## Master synthesis contract

Use scripts/build_master.py to validate source schemas, complete model membership, frozen protocol and provenance, then generate the 17-model table, seven winners per tier, 13 adjacent family scaling pairs and two strict Pareto fronts. Preserve source numerical strings. Compute deltas as smaller minus larger and relative changes against the larger checkpoint; fraction deltas may also be shown in percentage points. A descriptive near-tie screen of absolute mAP gap ≤0.001 is a reporting aid, not an equivalence or significance test. Keep inference-latency and allocated-VRAM Pareto objectives separate; do not compute a weighted score.

MASTER_RESULTS.md, README.md, DATA_SCHEMA.md and METHODOLOGY_REFERENCE.md use English. RESEARCH_INSIGHTS_TH.md, EXECUTIVE_SUMMARY_TH.md, MEETING_SUMMARY_TH.md and METRIC_GUIDE_TH.md use Thai prose with canonical technical terms. The Master meeting summary is neutral and reusable. Existing tier summaries and qualitative evidence remain unchanged. Generate exactly the ten requested Master PNG plots under plots/, with source CSVs and units recorded. Update only the Master completion state after final validation.
