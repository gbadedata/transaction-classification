# Changes

## Fixed routing threshold on validation customers (Nov 2026)

**Why.** The confidence-routing table cut each test set at its own 70th/80th/90th percentile, so the cut-off was chosen on the data being scored. A deployed system has to fix the threshold beforehand.

**What changed.**
- `txnclass/pipeline.py`: `threshold_for_coverage` (threshold that auto-classifies a target share) and `routing_at_threshold` (coverage and accuracy at a fixed threshold), with unit tests.
- `analysis.py`: holds back 132 training customers as validation customers, fixes the threshold on them, and applies it unchanged to the 220 unseen test customers.
- README section 8 and Limitations updated; `results/metrics.json` regenerated from the full dataset.

**Numbers.** Nothing previously reported changed (random split 0.907 / 0.797; unseen customers 0.873 / 0.649; ranked routing 98.7% / 96.7% / 93.5%). New:

| Target coverage | Coverage achieved (test customers) | Accuracy |
|---|---|---|
| 70% | 68.0% | 98.6% |
| 80% | 78.8% | 96.8% |
| 90% | 87.5% | 93.8% |
