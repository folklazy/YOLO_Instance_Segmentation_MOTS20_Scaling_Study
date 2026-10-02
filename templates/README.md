# How to use tier templates

Read STUDY_STANDARD.md and DATA_SCHEMA.md first. README provides navigation;
REPORT remains the technical research record. The Thai documents have different roles:

- RESULTS_SUMMARY_TH: compact quantitative summary; 5–8 overview bullets, one canonical table,
  six-category winners, 3–5 measured findings, concise speed/memory trade-offs and cross-tier inputs.
  No per-model essays or detailed visual cases.
- PRESENTATION_SUMMARY_TH: visual evidence and qualitative interpretation; one short overview,
  small context table, ~3–5 same-frame cases with observation → analysis → numeric connection,
  observed failures, near-tie check, 3–6 observation/interpretation findings and combined selection.
  No long numerical recap, repeated per-model rankings or mentor-specific section.

For NOT_RUN/incomplete tiers: retain headings and empty tables, mark pending, keep canonical CSVs
header-only, omit image placeholders. Analyze only after every model in the tier completes.
M/S/N use YOLO26, YOLO11, YOLOv8; X/E and L/C also use valid YOLOv9 e/c.
Reuse actual comparisons first, otherwise render saved RLE + original frames; never run inference
to fill documentation. If evidence cannot be reconstructed, explicitly report the gap.

Select cases using existing per-frame artifacts and a small visual review, including shared errors,
counterexamples and similar outputs. Record sequence/frame/selection reason and source hashes.
Use the same full-frame region, confidence and scale; explain GT IDs, ignore and FN/FP handling.
Only describe observed error categories. Counts require per-frame evidence; no inferred frequency.
Visuals may interpret mAP/AP75/Recall behavior; latency and VRAM remain benchmark measurements.
Do not claim significance, architectural causality, final CCTV suitability or a weighted score.
REPORT may link to PRESENTATION in a short Qualitative Analysis section without duplicating cases.
