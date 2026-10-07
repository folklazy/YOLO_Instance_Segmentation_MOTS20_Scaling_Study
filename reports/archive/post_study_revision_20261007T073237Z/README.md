# YOLO Instance Segmentation — Complete MOTS20 Scaling Study

**MASTER COMPLETE — PASS WITH WARNINGS.** This repository synthesizes the five completed tiers into exactly 17 pretrained checkpoints: YOLO26 = 5, YOLO11 = 5, YOLOv8 = 5, YOLOv9 = 2. Every checkpoint uses the same 2,862 MOTS20 frames and 26,894 frame-level Person GT instances. Synthesis performs no inference, tier rerun, training or adaptation. Source warnings remain documented.

## Read the results

| Document | Purpose | Language |
|---|---|---|
| [MASTER_RESULTS.md](MASTER_RESULTS.md) | Technical results, family scaling, Pareto analysis, limitations and conditional candidates | English |
| [RESEARCH_INSIGHTS_TH.md](RESEARCH_INSIGHTS_TH.md) | Detailed cross-tier interpretation | Thai prose |
| [EXECUTIVE_SUMMARY_TH.md](EXECUTIVE_SUMMARY_TH.md) | Short decision summary | Thai prose |
| [MEETING_SUMMARY_TH.md](MEETING_SUMMARY_TH.md) | Neutral reusable meeting/presentation summary | Thai prose |
| [METRIC_GUIDE_TH.md](METRIC_GUIDE_TH.md) | Metric definitions, units and interpretation limits | Thai prose |
| [METHODOLOGY_REFERENCE.md](METHODOLOGY_REFERENCE.md) | Shared frozen method and synthesis checks | English |
| [DATA_SCHEMA.md](DATA_SCHEMA.md) | Canonical source and derived table contract | English |

## Study Navigation

[Largest](https://github.com/folklazy/YOLO_Large_Seg_MOTS20_Benchmark) | [Second-largest](https://github.com/folklazy/YOLO_Second_Largest_Seg_MOTS20_Benchmark) | [Medium](https://github.com/folklazy/YOLO_Medium_Seg_MOTS20_Benchmark) | [Small](https://github.com/folklazy/YOLO_Small_Seg_MOTS20_Benchmark) | [Nano](https://github.com/folklazy/YOLO_Nano_Seg_MOTS20_Benchmark) | [Master Study](https://github.com/folklazy/YOLO_Instance_Segmentation_MOTS20_Scaling_Study)

## Overall measured winners

| Objective | Model | Value |
| --- | --- | --- |
| Mask mAP50-95 | YOLO26x-Seg | 0.603730 |
| Inference latency | YOLOv8n-Seg | 9.045 ms/frame |
| Pipeline latency | YOLO26l-Seg | 76.403 ms/frame |
| Pipeline FPS | YOLO26l-Seg | 13.089 frames/s |
| Allocated VRAM | YOLO26l-Seg | 777.52 MiB |

## Machine-readable artifacts

- [MASTER_17_MODELS.csv](metrics/MASTER_17_MODELS.csv): source-preserving 17-model table.
- [TIER_WINNERS.csv](metrics/TIER_WINNERS.csv): seven objectives for each tier.
- [SCALING_DELTAS.csv](metrics/SCALING_DELTAS.csv): 13 adjacent size pairs × 11 metrics; smaller minus larger.
- [PARETO_FRONTIER.csv](metrics/PARETO_FRONTIER.csv): mAP / inference frontier.
- [PARETO_VRAM.csv](metrics/PARETO_VRAM.csv): mAP / peak allocated VRAM frontier.
- [PER_SEQUENCE_MASTER.csv](metrics/PER_SEQUENCE_MASTER.csv): 68 per-sequence rows, separate from pooled AP.
- [TIMING_MASTER.csv](metrics/TIMING_MASTER.csv) and [MODEL_COMPLEXITY_MASTER.csv](metrics/MODEL_COMPLEXITY_MASTER.csv): source stage measurements and complexity.
- [PLOT_READY_RESULTS.csv](metrics/PLOT_READY_RESULTS.csv) and [NEAR_TIES.csv](metrics/NEAR_TIES.csv): reproducible plotting and descriptive screening inputs.
- [SOURCE_MANIFEST.json](provenance/SOURCE_MANIFEST.json): exact source revisions, run IDs, SHA256, compatibility evidence and editorial archives.
- [MASTER_VALIDATION.json](provenance/MASTER_VALIDATION.json): final scientific/document checks.

## Ten Master plots

- [01_master_map_by_model.png](plots/01_master_map_by_model.png)
- [02_family_accuracy_scaling.png](plots/02_family_accuracy_scaling.png)
- [03_family_inference_scaling.png](plots/03_family_inference_scaling.png)
- [04_family_fps_scaling.png](plots/04_family_fps_scaling.png)
- [05_family_vram_scaling.png](plots/05_family_vram_scaling.png)
- [06_accuracy_vs_parameters.png](plots/06_accuracy_vs_parameters.png)
- [07_accuracy_vs_inference.png](plots/07_accuracy_vs_inference.png)
- [08_accuracy_vs_vram.png](plots/08_accuracy_vs_vram.png)
- [09_pareto_accuracy_latency.png](plots/09_pareto_accuracy_latency.png)
- [10_adjacent_scaling_delta.png](plots/10_adjacent_scaling_delta.png)

## Reproduce and validate

Run from the workspace root with the existing environment; no packages need to change:

```bash
.venv/bin/python -B YOLO_Instance_Segmentation_MOTS20_Scaling_Study/scripts/build_master.py --check-inputs
.venv/bin/python -B YOLO_Instance_Segmentation_MOTS20_Scaling_Study/scripts/build_master.py --validate
.venv/bin/python -B YOLO_Instance_Segmentation_MOTS20_Scaling_Study/scripts/validate_master.py
.venv/bin/python -B -m unittest discover -s YOLO_Instance_Segmentation_MOTS20_Scaling_Study/tests -v
```

The initial build uses scripts/build_master.py, plots use scripts/plot_master.py, and documents use scripts/report_master.py. Different existing outputs are never silently overwritten; archive them before a newly authorized rebuild. Calculations preserve source strings and use Decimal for scaling. The manifest records source revisions and builder hashes. None of these synthesis scripts import a model runtime or open raw predictions.

## Interpretation and retained evidence

Separate measured observations from interpretation. Near ties are descriptive; no significance test is claimed. Pipeline FPS excludes RLE and output writing. Size reduction can lower forward time while raising pipeline latency or allocated VRAM. Pareto fronts are separate two-objective analyses, without a weighted winner. Results identify candidates for later CCTV robustness evaluation, not deployment readiness or final CCTV superiority.

Existing tier quantitative summaries and visual qualitative analyses retain their distinct roles. The [current diagnostic case-selection audit](provenance/TIER_CASE_SELECTION_20261006_V2.md) remains unchanged; this synthesis does not reselect frames or inspect raw predictions. Shared diagnostic frames are not independent samples.

[STUDY_STANDARD.md](STUDY_STANDARD.md), [STUDY_STATE.json](STUDY_STATE.json), [language policy](DOCUMENTATION_LANGUAGE_POLICY.md), [templates/](templates/) and frozen provenance remain available. Historical Stage 0 conversion/validation utilities retain their original assumptions; do not use them to regenerate current results or state. The previous read-only tier documentation validator remains supported after the authorized Master editorial transition.

## Stop condition

All five tiers and the 17-model Master synthesis are complete. No further benchmark or robustness experiment is started. Any next experiment requires a separate user instruction.
