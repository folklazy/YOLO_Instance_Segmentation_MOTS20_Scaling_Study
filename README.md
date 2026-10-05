# YOLO Instance Segmentation MOTS20 Scaling Study

Stage 0: study setup and historical standardization only. Largest, Second-largest, Medium and Small are COMPLETE / PASS_WITH_WARNINGS; Nano remains READY / NOT_RUN. The final 17-model synthesis is pending. This repository performs no model inference.

## Control files

- [STUDY_STANDARD.md](STUDY_STANDARD.md): reusable execution and STOP rules
- [STUDY_STATE.json](STUDY_STATE.json): actual completion/result states
- [DATA_SCHEMA.md](DATA_SCHEMA.md): canonical import interface
- [METHODOLOGY_REFERENCE.md](METHODOLOGY_REFERENCE.md): shared methodology
- [templates/](templates/): four report reading levels
- [provenance/](provenance/): audit, baseline, dataset and validation evidence

## Study Navigation

[Largest](https://github.com/folklazy/YOLO_Large_Seg_MOTS20_Benchmark) | [Second-largest](https://github.com/folklazy/YOLO_Second_Largest_Seg_MOTS20_Benchmark) | [Medium](https://github.com/folklazy/YOLO_Medium_Seg_MOTS20_Benchmark) | [Small](https://github.com/folklazy/YOLO_Small_Seg_MOTS20_Benchmark) | [Nano](https://github.com/folklazy/YOLO_Nano_Seg_MOTS20_Benchmark) | [Master Study](https://github.com/folklazy/YOLO_Instance_Segmentation_MOTS20_Scaling_Study)

## Later synthesis design

After all five tiers are validated and synthesis is explicitly requested, import exactly 17 unique checkpoints from canonical tier CSVs. DATA_SCHEMA.md defines rejection gates. Future artifacts: EXECUTIVE_SUMMARY_TH.md, MEETING_SUMMARY_TH.md, RESEARCH_INSIGHTS_TH.md, MASTER_RESULTS.md, METRIC_GUIDE_TH.md; metrics/MASTER_17_MODELS.csv, TIER_WINNERS.csv, SCALING_DELTAS.csv, PARETO_FRONTIER.csv, PARETO_VRAM.csv, PER_SEQUENCE_MASTER.csv; plots/; provenance/SOURCE_MANIFEST.json; scripts/build_master.py. These result files are intentionally absent now. No partial/fake master result exists.

## Validation and stop

Run `.venv/bin/python YOLO_Instance_Segmentation_MOTS20_Scaling_Study/scripts/validate_stage0.py` from workspace root for read-only interface/source validation. Conversion scripts preserve run artifacts and execute no model inference; they are historical setup utilities, not benchmark launchers. Stage 0 completion does not authorize any next tier. STOP and wait for explicit instruction.

## Documentation roles

RESULTS_SUMMARY_TH.md gives the compact quantitative summary. PRESENTATION_SUMMARY_TH.md
explains actual same-frame prediction behavior, observed failures and cautious interpretation.
Largest, Second-largest, Medium and Small have four diagnostic cases each. Small now follows
the shared visual presentation format after the latest user-requested alignment; its benchmark-time
numerical presentation is archived. Nano remains pending and follows the current templates after
all its models finish. REPORT remains the technical record and links to visual analysis.
Comparisons use original MOTS20 frames and saved RLE masks without inference or changed measurements.
The selected comparison images embedded in the summaries and their small case-selection/evidence
manifests are versioned. Other generated images and saved predictions remain local/ignored.

The Stage 0 generators are historical conversion utilities with embedded old documentation designs.
Do not rerun them to regenerate current summaries, standards, templates or study state.
Use the current templates and STUDY_STANDARD.md for future tiers.
Validate the redesign with `.venv/bin/python YOLO_Instance_Segmentation_MOTS20_Scaling_Study/scripts/validate_documentation_redesign.py`
from workspace root. The existing Stage 0 validator retains historical NOT_RUN assumptions for Medium.
After publishing, add `--remote-images` to verify GitHub branch heads, image blob hashes and raw-image
responses for every image embedded in the tier's top-level Markdown documents.

Small completed run `benchmark-20261005T051531Z`: three checkpoints × 2,862 frames, AP maxDet200
convergence PASS, nine clean timing rounds and scientific/document validation PASS. The documentation
validator permits only formerly empty Small canonical CSV interfaces to receive these newly measured
results; historical measurements in other tiers remain protected by their recorded hashes.
Small is complete; STOP. Nano and final cross-tier synthesis require a new user instruction.
