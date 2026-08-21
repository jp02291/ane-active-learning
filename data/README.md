# Measured data

`data.csv` holds one row per measured composition, 70 in total.

| column | meaning |
|---|---|
| `Fe`, `Co`, `Mn`, `Ga`, `Al`, `Si`, `Ge`, `Pt` | atomic fractions |
| `kxx` | thermal conductivity at 300 K [W m-1 K-1] |
| `S_ANE` | anomalous Nernst coefficient at 300 K [uV K-1] |
| `kappa_source` | `measured`, or `model` where kappa was reconstructed |
| `cycle_added` | 0 for the initial dataset, 1-3 for each closed-loop cycle |
| `reference` | `This work`, or the literature source |
| `label` | the composition as written in the manuscript |

## Reproducing a single cycle

`cycle_added` is what makes the earlier cycles reproducible. Cycle 2 was run on
the data that existed when it started, not on everything measured since:

```python
from ane.data import load_dataset
df = load_dataset("data/data.csv", up_to_cycle=2)   # 55 rows
```

| | rows available |
|---|---|
| cycle 1 | 45 |
| cycle 2 | 55 |
| cycle 3 | 65 |
| after cycle 3 | 70 |

Each cycle added ten candidates, five chosen for predicted performance and five
for ensemble disagreement. In the final cycle only the five performance
candidates were synthesized, since no further model update was planned; the
five unmeasured exploration candidates are in `unmeasured_candidates.csv` with
the predictions that selected them.

## Two things to know before using it

**Not every kappa is measured.** Thermal conductivity was not reported for the
13 literature entries. For those rows kappa was reconstructed as a Callaway
lattice term plus a Wiedemann-Franz electronic term from the reported
electrical conductivity; see `ane.physics`. They are marked `model` in
`kappa_source` and are model-derived. Comparison against compositions measured
here indicates the reconstruction runs about 9% high in the overlapping Fe-Ga
range.

**The initial 45 rows are the reconciled analysis values.**
`authoritative_cycle1_45.csv` links every row to Supplementary Table S1 and to
the recovered historical 36/9 membership. The atomic fractions in those 45
rows are closed to exactly one and retain the unrounded property values used by
the recovered analysis files. The rounded manuscript spelling remains in the
`label` column.

## Other files

`candidates.csv` records what the loop actually did: the thirty compositions it
proposed across three cycles, which rule selected each one, what the surrogate
predicted, and what was then measured. It is the audit trail for the closed
loop rather than an input to it.

`reported_splits/cycle1/` contains the recovered 36-row training and 9-row test
membership. `split/` is the output directory written by
`scripts/0_split_data.py`. For cycles 2 and 3 the partition is reconstructed
from `data.csv` and the seed in the configuration. The partition is not random;
see `ane.data` for what it does and why.

## `literature_reconstruction.csv`

The 13 literature entries of the initial dataset, whose thermal conductivity was
not reported and was reconstructed with the Callaway–Klemens model of
Supplementary Note S1. One row per entry: composition, reference, the reported
anomalous Nernst coefficient, the electronic and lattice components, and the
total that appears in `data.csv` and in Supplementary Table S1.

`sigma_back_derived` is **not** the value reported in the source papers. Refs [1]
and [2] report an electrical conductivity that was not carried into this
repository, so the column is recovered from the deposited total by
`sigma = (kappa_total - kappa_L) / (L0 T)`, with `kappa_L` from `ane.physics`.
It is included because it is the input the reconstruction consumes and a reader
otherwise cannot re-enter the calculation at all; the values it gives,
0.88-1.92e6 S/m, sit in the same range as the measured conductivities of
Supplementary Table S7. Replace this column with the reported conductivities
before release, and the regression test in `tests/test_callaway_reconstruction.py`
becomes an end-to-end check rather than a consistency one.
