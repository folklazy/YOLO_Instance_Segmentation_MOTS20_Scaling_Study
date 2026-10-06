# สรุปผล {{TIER}} YOLO Instance Segmentation

<!-- COMPACT QUANTITATIVE SUMMARY. After COMPLETE + validated canonical CSV only.
For NOT_RUN, retain headings and empty tables; state pending instead of inventing winners.
Include concise per-model table interpretation; do not add visual case analysis. -->

## สรุปใน 1 นาที

<!-- 5–8 concise bullets: model membership; MOTS20 2,862 frames / 26,894 frame-level
Person GT instances; pretrained / no fine-tuning; accuracy winner; inference and
pipeline speed winners; lowest allocated VRAM; largest measured trade-off. -->
{{OVERVIEW_BULLETS}}

## ผลลัพธ์หลัก

| Model | Mask mAP50-95 | AP75 | Recall | Inference ms | Pipeline ms | FPS | Peak VRAM MiB |
|---|---|---|---|---|---|---|---|
<!-- One canonical table, fixed family order. AP/Recall 6 decimals; ms/FPS 3; MiB 2. -->

## สรุปผลจากตาราง

<!-- One subsection per valid model in fixed family order (YOLO26, YOLO11,
YOLOv9 e/c for Largest/Second-largest only, YOLOv8). Use 1–2 short paragraphs:
measured strengths → actual trade-off → conditional candidate for later evaluation.
Explain relations across metrics rather than listing each cell or declaring balanced-best.
AP50/TP-only IoU/Dice outside the compact table must cite canonical CSV / REPORT.
TP-only quality is conditional on matching and may use different GT subsets per model.
Recall is fixed-confidence mask-matching coverage, not tracking or detection alone.
Separate inference from pipeline; tiny gaps are descriptive, not significance.
For NOT_RUN, list planned model subsections with pending text and no fabricated rankings.
No visual cases, causal architecture claims, weighted score or final CCTV superiority. -->

### {{MODEL}}

{{MEASURED_STRENGTH_AND_LIMITATION}}

{{RESOURCE_TRADE_OFF_AND_CONDITIONAL_CANDIDATE}}

## Winner ของแต่ละด้าน

| ด้าน | Model | Result |
|---|---|---|
<!-- Six rows: Mask mAP50-95, AP75, Recall, Inference speed, Pipeline speed, VRAM.
Use explicit units. Winners require complete accuracy and accepted clean timing. -->

## สิ่งที่ตัวเลขบอกเรา

{{THREE_TO_FIVE_MEASURED_FINDINGS_INCLUDING_DESCRIPTIVE_NEAR_TIES}}

## Trade-off หลัก

### Accuracy vs Speed

{{CANONICAL_ACCURACY_GAP_AND_LATENCY_DIFFERENCE}}

### Accuracy vs Memory

{{CANONICAL_ACCURACY_GAP_AND_ALLOCATED_VRAM_DIFFERENCE}}

## ข้อควรระวังในการตีความ

ไม่มีการทดสอบ statistical significance; MOTS20 ไม่ใช่ผลทดสอบ CCTV robustness ขั้นสุดท้าย
Pipeline ไม่รวม RLE preparation และ disk I/O; VRAM เป็น peak allocated ของ benchmark

## ข้อมูลสำหรับนำไปรวมต่อ

{{CROSS_TIER_CANDIDATES_WITHOUT_WEIGHTED_SCORE_OR_FINAL_17_MODEL_CONCLUSION}}

[TIER_RESULTS.csv](metrics/TIER_RESULTS.csv) · [REPORT.md](REPORT.md) ·
[Visual analysis](PRESENTATION_SUMMARY_TH.md) ·
[Master Study](https://github.com/folklazy/YOLO_Instance_Segmentation_MOTS20_Scaling_Study)
