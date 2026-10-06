# Documentation language by file role

Current documentation follows the existing English technical / Thai summary split.
Language is determined by a file's purpose, not by the model tier.

| File role | Language | Applies to |
|---|---|---|
| Technical research record | English | REPORT.md and TIER_REPORT_TEMPLATE.md |
| Navigation and technical instructions | English | README.md, EXPERIMENT_PROTOCOL.md, STUDY_STANDARD.md, DATA_SCHEMA.md, METHODOLOGY_REFERENCE.md, supporting configs/metrics/src READMEs and plots/INDEX.md |
| Active qualitative selection ledger | English | outputs/visualizations/qualitative/selection_v2/CASE_SELECTION.md |
| Quantitative summary | Thai prose | RESULTS_SUMMARY_TH.md and its template |
| Visual qualitative analysis | Thai prose | PRESENTATION_SUMMARY_TH.md and its template |

Model names, checkpoint filenames, metric names, units, schema fields, status tokens,
commands and established section labels retain their canonical spelling. Thai summaries
may therefore contain labels such as Case, Observation, Interpretation, Failure Analysis
and Trade-off. This intentional terminology does not permit Thai prose in English technical
files or require a new translation of the Thai summaries. Template authoring comments
and machine placeholders may remain English.

Keep section order and table schemas shared across tiers. REPORT contains the technical
record, numerical findings, warnings and limitations; it links to qualitative evidence instead
of duplicating the case discussion. Thai summaries preserve their separate quantitative and
visual roles. All measured values must come from canonical CSVs.

Historical run reports, frozen inputs, archives, superseded selection ledgers and provenance
narratives retain their original language. Archive changed current documents and record their
before/after hashes; do not rewrite historical evidence or frozen settings for language consistency.

Run scripts/validate_documentation_redesign.py from the workspace as documented in README.md.
It checks the active Markdown inventory, rejects Thai in English technical documents, checks
shared report headings and numerical tables against canonical CSVs, resolves current evidence
links, and verifies immutable benchmark artifacts and archived document snapshots.
