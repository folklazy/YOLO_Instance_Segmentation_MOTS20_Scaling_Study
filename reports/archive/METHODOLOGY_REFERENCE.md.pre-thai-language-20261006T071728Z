# Common methodology reference

The validated Largest frozen config/protocol is authoritative. Exact values and reuse instructions are in [STUDY_STANDARD.md](STUDY_STANDARD.md); implementation hashes and baseline environment in [BASELINE_REFERENCE.json](provenance/BASELINE_REFERENCE.json).

## Accuracy path

Image → OpenCV BGR → aspect-preserving centered square letterbox → RGB BCHW FP32 /255 → YOLO Person instance segmentation (one-to-many NMS for YOLO26) → original-resolution masks → frozen Person-first GT matching and ignore handling → pooled fixed-threshold metrics and confidence-ranked mask AP. Network input is 1×3×640×640. Native output restores each image's original dimensions. Model max_det=1000 and evaluation AP maxDet=200 are separate policies.

GT class 2 is Person; unmatched predictions overlapping union(class10) by prediction-area IoA ≥0.50 are ignored after valid Person matching. Fixed confidence 0.25, mask matching IoU 0.50; AP confidence floor 0.001, NMS IoU 0.70. AP spans mask IoUs 0.50:0.05:0.95 with 101 recall points. This is per-frame segmentation, not official MOTS tracking.

## Timing path

Frozen ordered 100-frame subset → load model separately → ten untimed warmups → three clean repetitions → synchronized preprocessing/H2D → forward → postprocessing/native masks/compact transfer → summed pipeline latency, mean-derived FPS and allocator peak VRAM. RLE preparation is timed separately. Disk I/O/decode, loading, GT decoding, accuracy computation, visualization and serialization are outside pipeline. Largest initial contaminated timing is excluded; preserved clean repetitions are primary. Details and exact hashes are in the standard and tier provenance.

## Why compatibility matters

Dataset, frame order and native dimensions affect workload; preprocessing, input size and precision affect both masks and speed. Evaluator, confidence/NMS/ignore thresholds and maxDet affect which predictions count. GPU, warmup, synchronization, measurement boundaries and contamination controls affect latency and memory. Change any of these and numerical differences can no longer be attributed to the same controlled comparison without qualification. Capacity and pretraining still differ between families; E/X and C/L are not exact equivalences. No statistical significance or CCTV robustness is established here.

## Dataset summary

| Sequence | Frames | Resolution | Person GT instances | Ignore regions |
|---|---:|---:|---:|---:|
| MOTS20-02 | 600 | 1920×1080 | 7,039 | 600 |
| MOTS20-05 | 837 | 640×480 | 6,570 | 802 |
| MOTS20-09 | 525 | 1920×1080 | 4,774 | 525 |
| MOTS20-11 | 900 | 1920×1080 | 8,511 | 900 |

Total 2,862 frames and 26,894 Person frame-level instances. The same person in multiple frames contributes multiple annotations. Canonical GT is the bundled sequence gt/gt.txt only. [Dataset compatibility evidence](provenance/DATASET_VALIDATION.json): PASS.
