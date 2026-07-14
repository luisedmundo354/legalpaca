# Retrieval cross-validation analysis

This report is generated from five version-bound complete-ranking bundles. Every stored query metric was independently recomputed from the raw ranking and its multi-positive gold set before aggregation.

## Locked analysis

Queries are averaged within each held-out case, matched seeds 17/29/43 are then averaged within that case, and the 42 case values are finally averaged. The primary endpoint is case-macro Hit@20 under the fold-global candidate regime. Intervals use 10,000 paired case resamples with analysis seed 17; they are conditional on the three trained seeds. The hierarchical case/seed interval is reported only as a sensitivity analysis.

## Primary cells

| Representation | Sampler | Hit@20 | Seed SD |
|---|---:|---:|---:|
| flat_masked | global_uniform | 0.155845 | 0.008300 |
| flat_masked | local_unique | 0.156720 | 0.007875 |
| structured | global_uniform | 0.132220 | 0.009652 |
| structured | local_unique | 0.129962 | 0.002754 |

## Prespecified contrasts

| Contrast | Estimate | 95% paired-case CI | Hierarchical sensitivity CI | Seed 17 | Seed 29 | Seed 43 | Seed SD | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Flat: global-uniform minus local-unique | -0.000875 | [-0.014743, 0.012937] | [-0.018292, 0.017707] | -0.004120 | 0.015464 | -0.013969 | 0.014983 | uncertain_crosses_zero |
| Structured: global-uniform minus local-unique | 0.002258 | [-0.010347, 0.016180] | [-0.012870, 0.019154] | -0.007253 | 0.007109 | 0.006918 | 0.008237 | uncertain_crosses_zero |
| Local-unique: structured minus flat | -0.026758 | [-0.071052, 0.016175] | [-0.071304, 0.018141] | -0.026085 | -0.016843 | -0.037347 | 0.010268 | uncertain_crosses_zero |
| Global-uniform: structured minus flat | -0.023625 | [-0.060788, 0.012392] | [-0.061254, 0.014537] | -0.029218 | -0.025198 | -0.016459 | 0.006523 | uncertain_crosses_zero |
| Difference in structural effects (global minus local) | 0.003133 | [-0.012239, 0.018868] | [-0.018700, 0.025205] | -0.003133 | -0.008355 | 0.020887 | 0.015596 | uncertain_crosses_zero |

A positive paper-facing claim is permitted only when the point estimate is positive and its paired case-bootstrap interval is wholly above zero. Intervals crossing zero are described as uncertain.

## Context-excluded robustness

| Representation | Sampler | Fold-global | Context-excluded | Difference |
|---|---:|---:|---:|---:|
| flat_masked | global_uniform | 0.155845 | 0.160806 | 0.004960 |
| flat_masked | local_unique | 0.156720 | 0.164108 | 0.007388 |
| structured | global_uniform | 0.132220 | 0.134337 | 0.002116 |
| structured | local_unique | 0.129962 | 0.133864 | 0.003902 |

The context-excluded regime removes visible non-gold passages from the complete fold-global ranking without rescoring and never removes a gold.

## Sampler definitions

`local_unique` uses 40 unique same-case and 20 unique other-case negatives. `global_uniform` samples 60 unique negatives passage-uniformly from all eligible passages in the outer training folds. Both exclude the current query's positives, use no replacement, and share matched positive selection across representation and sampler cells.

## Study boundary and correction

The controlled analysis uses the corrected 490-query dataset, in which all 42 cases yield queries. The frozen March 471-query dataset is retained only for legacy-configuration comparison and is not mixed into this aggregate. The parser correction includes the left-directed supporting edge into case 42's final Conclusion. The corrected controlled runs use 24–26 training cases per outer fold, so absolute values are not presented as a bitwise replication of the March 34-case configuration.

## Evidence

`rankings_manifest.json` records the exact S3 key, VersionId, byte size, and SHA-256 for each raw ranking input; the multi-gigabyte ranking files are not duplicated in this compact report. `jobs.json` binds every fold to its completed SageMaker receipt. The exact experiment and fold manifests, the complete corrected dataset input files with their manifest, and all terminal/acquisition receipts are copied under `input/`. All derived tables and SVG inputs are listed in `analysis_manifest.json`.
