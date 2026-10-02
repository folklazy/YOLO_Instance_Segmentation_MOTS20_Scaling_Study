"""Setup documentation/interfaces only; no inference and no master synthesis."""
from stage0_standardize import *
import yaml
if (MASTER/'STUDY_STATE.json').exists():
 raise SystemExit('Controls already exist; refusing to overwrite study state or documents.')
PLOTS=['01_mask_map50_95.png','02_ap50_ap75.png','03_inference_latency.png','04_pipeline_fps.png','05_peak_vram.png','06_accuracy_vs_latency.png']
base_repo=ROOT/NAMES[0];frozen=base_repo/'reports'/RUNS[0]/'frozen_inputs'
config=yaml.safe_load((frozen/'configs/benchmark.yaml').read_text())
ref_paths=['src/benchmark_adapter.py','src/frozen_pilot/yolo_adapter.py','src/frozen_pilot/metrics.py','src/frozen_pilot/mots.py','src/benchmark.py']
fingerprints={f'{NAMES[0]}/reports/{RUNS[0]}/frozen_inputs/{x}':sha(frozen/x) for x in ref_paths}
for x in ['src/repeat_clean_timing.py','manifests/framework_source_hashes.json','manifests/timing_frames.json','manifests/environment.json']:
 fingerprints[f'{NAMES[0]}/{x}']=sha(base_repo/x)
dump(MASTER/'provenance/BASELINE_REFERENCE.json',{'study_id':SID,'schema_version':'1.0','config':config,'implementation_sha256':fingerprints,'environment':{k:v for k,v in read(base_repo/'manifests/environment.json').items() if k in ['python','packages','torch_version','torchvision_version','cuda_runtime','gpu']},'prompt_discrepancies':[],'notes':['Frozen values match the expected prompt values.','Presentation order is YOLO26, YOLO11, YOLOv9, YOLOv8; historical execution order remains unchanged.']})
membership='| Tier | Repository | Checkpoints in display order |\n|---|---|---|\n'+''.join(f"| {LABELS[i]} | [{n}](https://github.com/folklazy/{n}) | "+', '.join('`'+c+'-seg.pt`' for c in CHECKPOINTS[i])+' |\n' for i,n in enumerate(NAMES[:5]))
standard=f'''# Study standard — Stage 0 / schema 1.0

## Purpose and scope

Study ID: `{SID}`. Evaluate exactly 17 official pretrained YOLO instance-segmentation checkpoints on frame-level Person masks in MOTS20. No training, fine-tuning, adaptation or non-YOLO models. Master performs no inference. E/X and C/L denote largest/second-largest available variants, not equal capacity.

## Repository mapping and model membership

{membership}
Master: [{MASTER.name}](https://github.com/folklazy/{MASTER.name}). Resolve workspace from checkout location; all six repositories are siblings. Shared `datasets/`, `models/`, `.venv/` remain at workspace root. Invoke `.venv/bin/python` directly; never move the environment. Never invent YOLOv9 x/l/m/s/n segmentation checkpoints; only e and c are in this study.

## Evidence hierarchy

Frozen machine-readable measurement > frozen config/protocol > final metric CSV/JSON > final timing summary > provenance/manifest > REPORT > README. Preserve numeric strings from structured sources; never transcribe README values. Canonical CSV is a normalized interface to the cited measurement evidence. If unavailable, use empty/NA and explain; zero is never a missing-value placeholder. Preserve historical paths and all raw evidence. Archive substantially replaced Markdown under `reports/archive/<name>.pre-standardization-<date>.md` and record SHA256.

## Dataset

Only `datasets/MOTS/MOTS/train/`, ordered MOTS20-02, MOTS20-05, MOTS20-09, MOTS20-11 then ascending frame. Exactly 2,862 frames and 26,894 frame-level Person GT instances (not unique people). GT is `<sequence>/gt/gt.txt`, Person class 2, ignore class 10. `datasets/MOTSLabels/MOTSLabels/` is not an additional source. Never modify inputs or create splits. Reuse the exact frozen image, timing, preflight and visualization manifests. Verify hashes, ordering, dimensions and metadata; shared validation once per unchanged dataset is sufficient. Evidence: `provenance/DATASET_VALIDATION.json`; baseline ordered per-image hashes remain in Largest frozen inputs. Human tier reports normally show Dataset compatibility: PASS.

## Frozen preprocessing and inference

Reuse `{NAMES[0]}/reports/{RUNS[0]}/frozen_inputs/src/benchmark_adapter.py` and its frozen pilot adapter. Do not independently rewrite preprocessing. OpenCV BGR input; aspect-preserving linear resize; square centered letterbox with padding 114, auto=False, scaleup=True, no stretch; RGB BCHW FP32 /255; shape 1×3×640×640. `retina_masks=True` restores native-resolution masks, binary masks and empty-mask filtering; preserve lossless GPU bit-packing/CPU transfer. Implementation SHA256 and framework-source fingerprint file are in `provenance/BASELINE_REFERENCE.json` (machine-readable fingerprint authority).

CUDA:0 respecting visibility, batch 1, imgsz 640, FP32, augment=False, rect=False, retina_masks=True, agnostic_nms=False. YOLO26 `nms=True` selects the one-to-many path before fusion; assert backend end2end=False. Person model class 0 (`person`). AP confidence floor 0.001 (strict > in framework NMS), fixed confidence >=0.25, NMS box IoU 0.70, model max_det=1000. Stop on cap saturation or NMS truncation warning. Framework outputs must use a fresh directory inside the owning experiment. No package upgrades, model downloads or environment changes without authorization. Frozen environment: Python 3.12.3, Ultralytics 8.4.160, torch 2.14.0+cu130, torchvision 0.29.0+cu130, NumPy 2.5.3, pycocotools 2.0.11, CUDA runtime 13.0, Tesla T4, driver 580.178.04. Preserve baseline; differences require an explicit compatibility decision.

## Frozen evaluator and AP

Reuse frozen `metrics.py` and `mots.py`; hashes in BASELINE_REFERENCE.json and per-tier STANDARDIZATION.json. Regression suite must pass before future benchmark execution. Fixed matching: descending confidence, stable ties, greedy one-to-one mask IoU >=0.50; best IoU then GT object ID. Match valid Person GT first. Only unmatched prediction with intersection against union(class-10 masks)/prediction area >=0.50 is ignored; otherwise FP. Do not clip Person masks. P/R/F1 are pooled micro counts, TP IoU/Dice conditional means. AP is Person-only frame-level COCO-style segmentation, IoU 0.50:0.05:0.95, 101 recall points, valid-GT priority and frozen ignore semantics. Overall AP is pooled, never mean sequence AP. No tracking metrics.

AP maxDet=200, distinct from model max_det=1000. Preflight caps 100/200/300/1000 on the exact frozen 100 frames; retain complete sensitivity results. Largest selected 200 as smallest common cap with absolute AP50/AP75/mAP difference from 1000 strictly <0.0001. Future tiers retain 200 even if 100 converges. If 200 fails, STOP for a common-policy decision; do not change it for one tier.

## Timing and GPU contamination

Use exact ordered frozen `manifests/timing_frames.json`: 25 predetermined frames per sequence, 100 unique frames. Ten untimed warmups cycling first ten subset frames, then three clean rounds (300 observations/model), seed 20260929. Preserve seeded/cyclic family execution ordering from runner; display order is independent. One model at a time; clear old model/cache before loading. Idle gate outside measurement; synchronize CUDA at stage boundaries. Measure preprocessing/H2D, forward inference, postprocessing (NMS/native masks/bit packing/CPU transfer/prediction objects), pipeline=sum of these; RLE preparation separately. Exclude model loading, disk/decode I/O, GT, metrics, visualization and output serialization. Reset allocator peak after warmup, include resident model; report max allocated peak across accepted runs. FPS=1000/mean pipeline ms; population std, median/P50, P95 across all clean observations. Model loading is a separate mean in seconds.

Monitor GPU before/during/after. Any other compute PID or pre-load idle utilization >5% contaminates the entire run; preserve and exclude it. Repeat unchanged only when idle, using fresh records. Never rank incomplete timing. Largest primary timing is exclusively `timing/{RUNS[0]}/clean_repetition/`; initial pass remains excluded. Second-largest also uses `clean_repetition/`. Pipeline FPS is not end-to-end mask-saving/CCTV throughput.

## Canonical public interface

Each tier: README.md, REPORT.md, RESULTS_SUMMARY_TH.md, PRESENTATION_SUMMARY_TH.md, EXPERIMENT_PROTOCOL.md; `configs/`; `manifests/STANDARDIZATION.json`, `manifests/environment.json`; `metrics/TIER_RESULTS.csv`, PER_SEQUENCE_RESULTS.csv, TIMING_SUMMARY.csv, MODEL_COMPLEXITY.csv, PREFLIGHT_MAXDET.csv. Create `outputs/plots`, `outputs/visualizations`, `reports/archive`, `predictions`, `timing`, `logs`, `src` only when used. Existing valid run-scoped layouts stay intact; expose interfaces instead of moving large artifacts. Future tiers initially have header-only CSVs and explicitly NOT_RUN documents, no fabricated result rows.

Schema and units: DATA_SCHEMA.md and schemas.json. Provenance includes study/schema/repo/tier, run IDs, source paths+SHA256, evaluator/config/protocol/dataset hashes, timestamp, inference_rerun, metrics_recalculated, archived documents and notes. Keep environment evidence; baseline reference is not a future run environment capture. Completion (PENDING/READY/RUNNING/COMPLETE/BLOCKED) is separate from result (NOT_RUN/PASS/PASS_WITH_WARNINGS/FAIL/UNKNOWN). Never infer completion from folder presence. Do not edit actively written files.

## Reports, rounding and plots

Use exact major section/table ordering in `templates/`. README is navigation; RESULTS_SUMMARY_TH concise Thai result; PRESENTATION_SUMMARY_TH mentor-ready Thai discussion; REPORT compact technical record. Avoid repeating methodology and generic definitions; common method is METHODOLOGY_REFERENCE.md. Full metric guide and final synthesis come later. Display order: YOLO26, YOLO11, YOLOv9 (only largest/second-largest), YOLOv8. Explicitly metric-sorted tables may differ. Machine tier labels: largest, second_largest, medium, small, nano. Human labels: Largest (X/E), Second-largest (L/C), Medium (M), Small (S), Nano (N).

Markdown AP/P/R/F1/IoU/Dice: 6 decimals; latency/FPS: 3; VRAM MiB: 2; parameters: comma-separated integer; GFLOPs: 3; checkpoint decimal MB: 2. CSV preserves source precision. Tables must be generated/verified from canonical CSV. Include identical Study Navigation. Standard plots under outputs/plots: {', '.join(PLOTS)}. Preserve old plots; regenerate only presentation from saved metrics. Fixed model order/colors, explicit units, no weighted score.

## Scientific interpretation

Rank accuracy by Mask mAP50-95; show AP75, recall, inference/pipeline/FPS/VRAM winners separately. Describe tiny gaps as near-tied descriptively; no significance without valid analysis. Distinguish observation from interpretation; architecture speculation is not causality. Different capacities/pretraining confound causal conclusions. MOTS20 supports measured frame-level segmentation, speed and VRAM here; it does not establish blur, low-light, angle, explicit occlusion-severity robustness or deployment suitability. Say candidate for later CCTV robustness evaluation. TP-only quality excludes missed detections. Do not build 17-model conclusions until all tiers are validated.

## Execution and Git gates

On each request read this standard, STUDY_STATE.json and the requested tier; audit status/remotes/current branch and active processes before changes. Correct remote must be existing folklazy repository. Preserve unrelated user changes and archive replaced docs first. No deletion, force push, history rewrite, dataset/checkpoint/venv/secret commits or blind adds of predictions. Commit each repository independently after validation; review status, diff and exact staged paths. Push only to correct existing remote/current branch; record auth failure independently of scientific status.

For an explicitly authorized future tier: verify environment/imports/GPU without modifying working packages; resolve any authorized checkpoint acquisition; reuse baseline implementation and frozen lists; freeze new tier config/protocol/provenance; execute only that tier's preflight, accuracy and clean timing after gates; verify complete coverage and consistency; produce canonical artifacts/reports/plots; update state and commit. Separate execution authorization and download constraints still apply. No automatic resume or overwriting runs.

## STOP gate

Stage 0 authorizes setup and historical conversion only: no M/S/N downloads, preflight, inference, accuracy or speed benchmark; no final Master synthesis. Execute one tier per explicit request. Completion/readiness NEVER authorizes the next tier. After Stage 0 STOP and wait for explicit user instruction. Master later imports validated canonical results and performs no inference.
'''
write(MASTER/'STUDY_STANDARD.md',standard)
dump(MASTER/'schemas.json',{'study_id':SID,'schema_version':'1.0','schemas':{k:v.split() for k,v in SCHEMAS.items()},'PREFLIGHT_MAXDET':list(rows(ROOT/NAMES[0]/'metrics/PREFLIGHT_MAXDET.csv')[0])})
schema='# Canonical data interface\n\n`study_id = '+SID+'`; `schema_version = 1.0`. UTF-8 CSV, one header, comma delimiter; empty or NA means unavailable, never placeholder zero. Preserve numerical strings from authoritative measurements.\n\n'
for k,v in SCHEMAS.items():schema+='## '+k+'.csv\n\n```text\n'+','.join(v.split())+'\n```\n\n'
schema+='## PREFLIGHT_MAXDET.csv\n\n```text\n'+','.join(read(MASTER/'schemas.json')['PREFLIGHT_MAXDET'])+'\n```\n\n'
schema+='''## Identity, units and missing data

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
'''
write(MASTER/'DATA_SCHEMA.md',schema)
method=f'''# Common methodology reference

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
'''
write(MASTER/'METHODOLOGY_REFERENCE.md',method)
# Templates retain the exact canonical heading and column contracts without real result values.
import re
for name in ['README','REPORT','RESULTS_SUMMARY_TH','PRESENTATION_SUMMARY_TH']:
 text=(ROOT/NAMES[0]/(name+'.md')).read_text();parts=re.split(r'(?m)^(## .+)\n',text)
 result=parts[0].splitlines()[0].replace(LABELS[0],'{{TIER}}')+'\n\n<!-- Template only. Populate from canonical CSV after validation; never invent measurements. -->\n'
 for j in range(1,len(parts),2):
  heading=parts[j];body=parts[j+1]
  if heading=='## Study Navigation':content=NAV
  elif heading=='## Reports':content='\n'.join(f'- [{x}]({x})' for x in ['PRESENTATION_SUMMARY_TH.md','RESULTS_SUMMARY_TH.md','REPORT.md','EXPERIMENT_PROTOCOL.md'])
  elif heading=='## Reproducibility':content='[configs/](configs/) · [metrics/](metrics/) · [manifests/](manifests/)'
  else:
   lines=body.strip().splitlines();headers=[]
   for ix,l in enumerate(lines):
    if l.startswith('|') and ix+1<len(lines) and re.match(r'^\|[\s:|-]+\|$',lines[ix+1]):headers.extend([l,lines[ix+1]])
   subheads=[l for l in lines if l.startswith('### ') and l in ['### Accuracy','### Speed','### Memory / Resource','### ภาพรวม']]
   content='{{'+heading.removeprefix('## ').upper().replace(' ','_')+'}}'
   if headers:content+='\n\n'+'\n'.join(headers)+'\n<!-- Add measured rows only, in fixed model order. -->'
   if subheads:content+='\n\n'+'\n\n'.join(h+'\n\n{{EVIDENCE_BASED_TEXT}}' for h in subheads)
  result+='\n'+heading+'\n\n'+content+'\n'
 write(MASTER/'templates'/f'TIER_{name}_TEMPLATE.md',result)
write(MASTER/'templates/README.md','''# How to use tier templates

Read STUDY_STANDARD.md and DATA_SCHEMA.md first. Templates define exact headings and table order; tokens are instructions, not results. For NOT_RUN tiers, explicitly mark pending and keep CSV header-only. For completed tiers, generate all numbers from canonical CSV, retain historical PASS WITH WARNINGS and source evidence. README has no definitions; REPORT follows 12 sections; Thai summaries have distinct concise/meeting purposes.

RESULTS summary: 5–8 overview bullets covering models, frames/GT, pretrained scope, accuracy/speed/VRAM winners and key observation; one strength/weakness/rank/practical paragraph per model; 3–5 evidence-based insights. PRESENTATION: exact numbered checkpoints, 1–3 main-result paragraphs, one model subsection each, 3–5 insights, priority recommendations (accuracy/speed/low VRAM/balanced without score), 3–6 natural Thai discussion bullets. Distinguish inference from pipeline speed. Balanced recommendation must state a constraint/trade-off. Report sequence strongest/weakest and ranking changes only when supported. Cite canonical CSV/report/Master. Avoid repeated generic metric definitions. All major heading structures and columns must remain identical across completed tiers.
''')
ignore='''# Shared resources and generated runtime artifacts
.venv/
.venvs/
datasets/
models/
*.pt
*.pth
*.ckpt
__pycache__/
*.py[cod]
.env
.env.*
predictions/
logs/
metrics/**/per_frame/
metrics/**/per_instance/
metrics/gt_frames.csv
metrics/gt_instances.csv
timing/**/all_frames.csv
timing/**/round*.csv
outputs/visualizations/
# Intentionally retained canonical plots and evaluation summaries are versioned.
'''
for i,n in enumerate(NAMES[:5]):
 p=ROOT/n
 if not (p/'.gitignore').exists():write(p/'.gitignore',ignore)
 if i<2:continue
 cps=[x+'-seg.pt' for x in CHECKPOINTS[i]]
 cfg=dict(config);cfg['models']=cps
 write(p/'configs/benchmark.yaml',yaml.safe_dump(cfg,sort_keys=False))
 write(p/'configs/README.md','Setup-only protocol configuration inherited from Largest. NOT a frozen run. Before authorized execution, freeze this file and checkpoint/environment/source manifests into a fresh run. Shared paths resolve from workspace root; all generated framework outputs belong in this repository.\n')
 for k,v in SCHEMAS.items():csvout(p/'metrics'/f'{k}.csv',v.split(),[])
 csvout(p/'metrics/PREFLIGHT_MAXDET.csv',read(MASTER/'schemas.json')['PREFLIGHT_MAXDET'],[])
 write(p/'metrics/README.md','Header-only canonical interfaces: NOT_RUN. No benchmark results are present. Use [Master schema](https://github.com/folklazy/'+MASTER.name+'/blob/main/DATA_SCHEMA.md). Never add placeholders for untested models.\n')
 env=read(MASTER/'provenance/STAGE0_ENVIRONMENT.json')
 dump(p/'manifests/environment.json',{'record_type':'stage0_readiness_inspection_not_run_capture','timestamp':NOW,**env})
 dump(p/'manifests/STANDARDIZATION.json',{'study_id':SID,'schema_version':'1.0','repository':n,'experiment':n,'tier':KEYS[i],'source_run_ids':[],'source_artifact_paths':[],'source_artifact_sha256':{},'evaluator_hash':None,'config_hash':sha(p/'configs/benchmark.yaml'),'protocol_hash':None,'dataset_manifest_hash':read(MASTER/'provenance/DATASET_VALIDATION.json')['baseline_manifest_sha256'],'standardization_timestamp':NOW,'inference_rerun':False,'metrics_recalculated':False,'historical_documents_archived':[],'notes':['Setup-only; NOT_RUN. Environment is readiness inspection, not a benchmark run capture. Exact evaluator reuse and per-tier preflight are required after explicit authorization. No checkpoint download or inference performed.']})
 models='| Family | Model | Tier |\n|---|---|---|\n'+''.join(f'| {family(cp)} | {label(cp)} | {LABELS[i]} |\n' for cp in cps)
 write(p/'README.md',f'# {LABELS[i]} YOLO Instance Segmentation Benchmark on MOTS20\n\n## Overview\n\nSetup for pretrained Person instance segmentation on MOTS20 under the common controlled protocol. No training or fine-tuning. Stage 0 prepared interfaces only; this tier has not run.\n\n## Models\n\n{models}\n## Experimental Status\n\nREADY / NOT_RUN — readiness is setup readiness, not execution authorization.\n\n## Main Result\n\nNot run. CSVs contain headers only.\n\n## Reports\n\n'+ '\n'.join(f'- [{x}]({x}) — pending benchmark' for x in ['PRESENTATION_SUMMARY_TH.md','RESULTS_SUMMARY_TH.md','REPORT.md'])+'\n- [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) — setup protocol\n\n## Study Navigation\n\n'+NAV+'\n\n## Reproducibility\n\n[configs/](configs/) · [metrics/](metrics/) · [manifests/](manifests/)\n')
 for doc in ['REPORT','RESULTS_SUMMARY_TH','PRESENTATION_SUMMARY_TH']:
  text=(MASTER/'templates'/f'TIER_{doc}_TEMPLATE.md').read_text().replace('{{TIER}}',LABELS[i])
  text=re.sub(r'\{\{[^}]+\}\}','ยังไม่รัน (NOT_RUN) — รอการอนุมัติ benchmark ของ tier นี้',text)
  write(p/(doc+'.md'),text)
 write(p/'EXPERIMENT_PROTOCOL.md',f'''# {LABELS[i]} protocol — setup only

Status: NOT_RUN; no frozen run exists. Use [STUDY_STANDARD.md](https://github.com/folklazy/{MASTER.name}/blob/main/STUDY_STANDARD.md) and [METHODOLOGY_REFERENCE.md](https://github.com/folklazy/{MASTER.name}/blob/main/METHODOLOGY_REFERENCE.md). Checkpoints: {', '.join('`'+cp+'`' for cp in cps)}.

`configs/benchmark.yaml` inherits the validated Largest settings and AP maxDet200. Reuse exact baseline implementation and ordered manifests; do not rewrite preprocessing or evaluator. Model and framework output paths must remain inside this experiment except shared checkpoint/dataset inputs. Save a fresh run ID and freeze config/protocol, sources, environment and input hashes before an authorized preflight. No inference, preflight, timing or downloads were performed in Stage 0. Future run authorization is required; stop if 200 fails convergence and seek a common policy decision. Never automatically continue to another tier.
''')
 prov=read(p/'manifests/STANDARDIZATION.json');prov['protocol_hash']=sha(p/'EXPERIMENT_PROTOCOL.md');dump(p/'manifests/STANDARDIZATION.json',prov)
 write(p/'src/README.md',f'''# Future authorized execution

Read sibling `{MASTER.name}/STUDY_STANDARD.md` and STUDY_STATE.json. Reuse the frozen Largest benchmark_adapter and frozen_pilot implementation identified in provenance/BASELINE_REFERENCE.json; use the validated clean timing runner. Do not create a new preprocessing/evaluator implementation. Adapt only tier membership and owning-repository paths, record diffs/hashes, preserve exact sample lists, and freeze a fresh run before executing. Stage 0 does not authorize execution or downloads. No source runner is installed here yet to avoid accidental benchmark launch.
''')
# State separates research completion from setup readiness.
state={'study_id':SID,'schema_version':'1.0','stage':'STAGE_0','stage0_status':'PASS_WITH_WARNINGS','stop_gate':'WAITING_FOR_USER_APPROVAL','last_verified':NOW,'tiers':{}}
for i,n in enumerate(NAMES):
 state['tiers'][KEYS[i]]={'repository':n,'completion_status':'COMPLETE' if i<2 else 'READY' if i<5 else 'PENDING','result_status':'PASS_WITH_WARNINGS' if i<2 else 'NOT_RUN','canonical_run_id':RUNS[i] if i<2 else None,'standardized':True,'last_verified':NOW,'notes':'Historical completed result standardized without rerun.' if i<2 else 'Setup ready; checkpoint acquisition and per-tier preflight require explicit authorization.' if i<5 else 'Control files complete; final 17-model synthesis not built.'}
dump(MASTER/'STUDY_STATE.json',state)
write(MASTER/'README.md',f'''# YOLO Instance Segmentation MOTS20 Scaling Study

Stage 0: study setup and historical standardization only. Largest and Second-largest are COMPLETE / PASS_WITH_WARNINGS; Medium, Small and Nano are setup READY / NOT_RUN. The final 17-model synthesis is pending. This repository performs no model inference.

## Control files

- [STUDY_STANDARD.md](STUDY_STANDARD.md): reusable execution and STOP rules
- [STUDY_STATE.json](STUDY_STATE.json): actual completion/result states
- [DATA_SCHEMA.md](DATA_SCHEMA.md): canonical import interface
- [METHODOLOGY_REFERENCE.md](METHODOLOGY_REFERENCE.md): shared methodology
- [templates/](templates/): four report reading levels
- [provenance/](provenance/): audit, baseline, dataset and validation evidence

## Study Navigation

{NAV}

## Later synthesis design

After all five tiers are validated and synthesis is explicitly requested, import exactly 17 unique checkpoints from canonical tier CSVs. DATA_SCHEMA.md defines rejection gates. Future artifacts: EXECUTIVE_SUMMARY_TH.md, MEETING_SUMMARY_TH.md, RESEARCH_INSIGHTS_TH.md, MASTER_RESULTS.md, METRIC_GUIDE_TH.md; metrics/MASTER_17_MODELS.csv, TIER_WINNERS.csv, SCALING_DELTAS.csv, PARETO_FRONTIER.csv, PARETO_VRAM.csv, PER_SEQUENCE_MASTER.csv; plots/; provenance/SOURCE_MANIFEST.json; scripts/build_master.py. These result files are intentionally absent now. No partial/fake master result exists.

## Validation and stop

Run `.venv/bin/python YOLO_Instance_Segmentation_MOTS20_Scaling_Study/scripts/validate_stage0.py` from workspace root for read-only interface/source validation. Conversion scripts preserve run artifacts and execute no model inference; they are historical setup utilities, not benchmark launchers. Stage 0 completion does not authorize any next tier. STOP and wait for explicit instruction.
''')
write(MASTER/'.gitignore','__pycache__/\n*.py[cod]\n.env\n.env.*\n.venv/\n*.pt\n')
# Root README update is additive; preserve the old root document outside repositories.
rootread=ROOT/'README.md';backup=ROOT/'.workspace-backups/README.pre-standardization-20261002.md'
if not backup.exists():backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(rootread,backup)
intro='''## Study setup — 2026-10-02

All six scaling-study repositories are now present as separate workspace siblings. Largest and Second-largest expose canonical CSVs and four standardized report levels; their historical PASS WITH WARNINGS is retained. Medium, Small and Nano are setup-only READY / NOT_RUN. No inference, checkpoint downloads or final synthesis occurred during Stage 0.

'''+ '\n'.join(f'- [{LABELS[i] if i<5 else "Master Study"}]({n}/README.md)' for i,n in enumerate(NAMES))+f'\n\nRead [{MASTER.name}/STUDY_STANDARD.md]({MASTER.name}/STUDY_STANDARD.md) before any future tier. Shared datasets, models and `.venv/` remain at workspace root; invoke `.venv/bin/python` directly. Await explicit authorization before executing the next tier. Earlier dated status notes below are preserved.\n\n'
if '## Study setup — 2026-10-02' not in rootread.read_text():write(rootread,rootread.read_text().replace('# Human Body Extraction\n','# Human Body Extraction\n\n'+intro,1))
dump(MASTER/'provenance/WORKSPACE_DOCUMENTATION.json',{'original':'README.md','backup':str(backup.relative_to(ROOT)),'original_sha256':sha(backup),'updated_sha256':sha(rootread),'note':'Workspace root is not a Git repository; additive README update retained locally.'})
print('Control files, templates and three setup-only tiers prepared.')
