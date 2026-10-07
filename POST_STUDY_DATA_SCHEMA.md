# Post-study derived data interface

**DERIVED FROM EXISTING CANONICAL RESULTS.** Original measured CSVs and synthesis schemas remain unchanged. This interface adds analysis only.

Source strings are preserved for copied fields. Deltas/shares use Decimal precision 50; AP/Recall are fractions, percentage-point deltas multiply fractional changes by 100. Latencies are ms/frame, allocated VRAM MiB, checkpoint sizes decimal MB and parameters integer counts.

## PARETO_PIPELINE.csv

All 17 models. is_pareto is True/False; maximize unrounded mask_map50_95 and minimize pipeline_ms_mean. Strict dominance requires no worse in both objectives and a strict improvement in at least one. Equal points remain.

```text
model,tier,family,mask_map50_95,pipeline_ms_mean,fps,inference_ms_mean,peak_allocated_vram_mib,is_pareto
```

## TIMING_COMPOSITION.csv

All 17 models. Original stage means from TIMING_MASTER.csv; shares divide each included stage by pipeline. Component sum residual tolerance is 1e-10 ms. RLE preparation is separate, and ultralytics_postprocess_inclusive is a diagnostic subset, never added again.

```text
model,tier,family,preprocess_ms,inference_ms,postprocess_ms,pipeline_ms,rle_preparation_ms,ultralytics_postprocess_inclusive_ms,inference_share_pct,postprocess_share_pct,preprocess_share_pct,component_sum_minus_pipeline_ms
```

## PER_SEQUENCE_SCALING_ANALYSIS.csv

All 68 model/sequence rows. Competition ranks start at 1; sequence rank compares models within one tier and sequence. Pooled ranks use overall AP separately. Endpoint/adjacent deltas are smaller minus larger. Endpoint context repeats for each model in its family/sequence; do not sum these repeats as independent observations. The largest model has blank adjacent fields.

```text
model,tier,family,sequence,mask_map50_95,recall,within_tier_sequence_map_rank,pooled_tier_map_rank,sequence_rank_minus_pooled_rank,endpoint_larger_model,endpoint_smaller_model,endpoint_map_delta_smaller_minus_larger,endpoint_map_delta_pp,endpoint_recall_delta_pp,adjacent_larger_model,adjacent_map_delta_pp
```

## NEAR_TIE_RESOURCE_DELTAS.csv

All five pairs from the unchanged absolute mAP-gap screen <=0.001. Direction is model_b minus model_a; relative change divides by model_a. map_abs_gap_pp is unsigned while map_signed_delta_pp is signed. Resource suffixes identify A/B input values, absolute deltas and relative percentages. No statistical equivalence was tested.

```text
model_a,model_b,tier_a,tier_b,direction,map_a,map_b,map_abs_gap,map_abs_gap_pp,map_signed_delta_pp,screen_threshold,statistical_equivalence_tested,inference_a,inference_b,inference_delta_b_minus_a,inference_relative_pct,pipeline_a,pipeline_b,pipeline_delta_b_minus_a,pipeline_relative_pct,vram_a,vram_b,vram_delta_b_minus_a,vram_relative_pct,parameters_a,parameters_b,parameters_delta_b_minus_a,parameters_relative_pct,checkpoint_size_a,checkpoint_size_b,checkpoint_size_delta_b_minus_a,checkpoint_size_relative_pct
```

## Provenance and validation

POST_STUDY_REVISION.json records before/after protected hashes, archived active documents, original repository revisions and derived artifact hashes. Original SOURCE_MANIFEST.json remains immutable evidence for the completed synthesis. POST_STUDY_VALIDATION.json reports independent derived math, active document roles and saved-only same-frame evidence checks.

Read-only validation: `scripts/validate_post_study.py`. Optional `--recheck-saved-masks` decodes only the selected existing artifacts to verify per-frame diagnostics against canonical count records; it imports no model runtime and performs no dataset accuracy evaluation or timing.

CSV analysis generator: `scripts/post_study_analysis.py`; original measured outputs are never rewritten.
