# Post-study diagnostic case selection

Existing diagnostic frame pool and canonical per-frame rows; balanced control, differences and common failures. Not representative or full-dataset extrema.

| Case | Sequence/frame | Reason |
|---|---|---|
| 1 | MOTS20-09/000001 | same coverage control |
| 2 | MOTS20-09/000525 | different GT coverage despite near-equal pooled mAP |
| 3 | MOTS20-11/000450 | same coverage and differing extra masks |
| 4 | MOTS20-09/000263 | counterexample and common failure |

All models use identical original/GT, full-frame extent, ROI coordinates, thresholds and scale. Full frames are retained. Counts agree with existing per-frame CSVs.

[Evidence](EVIDENCE.json)
