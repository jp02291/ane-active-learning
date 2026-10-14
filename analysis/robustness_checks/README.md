# Conditional random and diversity baselines

This directory retains the conditional acquisition baselines on the 25
candidates measured during the campaign. They test prioritization within a
campaign-selected pool, not full-space superiority. At budgets 10 and 20, the
observed order did not outperform the random distribution on prespecified
best-FoM or hypervolume summaries. See `analysis/baseline/` for reproduction.

The earlier simplified nested branch comparison has been removed from the
current package and replaced by the completed, inner-fold-local single-attempt
evaluation in [../nested_policy/README.md](../nested_policy/README.md).
The historical simplified outputs remain available in earlier Git/release
versions; they are not a second nested result for the revised SI.
