# Distances in the model input space (Discussion)

This script backs the Discussion's account of the cycle-1 overpredictions: the
three Co-free Fe-Mn alloys nominated in cycle 1 lay farther from the training
data than any other cycle-1 candidate, and their two closest training
compositions contained 12.5 at.% Mn with |S_ANE| of 1.4 uV/K or less.

```bash
python run_distance.py
```

## Definition

As stated in Methods (Statistics and reproducibility), a distance is the
Euclidean distance between two compositions in the 15-dimensional input space
of the surrogate: the seven ILR coordinates and eight descriptors of
`ane.features.featurize`, with each feature min-max scaled to the 36 cycle-1
training compositions (`data/split/cycle1/train.csv`).

This is not the distance used by the diversity constraint of the selection
rule, which is measured between atomic-fraction vectors (`ane.select`), and not
the distance of Supplementary Note S3, which is also in atomic-fraction space.

## Result

| quantity | value |
|---|---|
| nearest-training distance of the three Fe-Mn candidates | 0.928 to 1.149 |
| largest nearest-training distance of the other seven cycle-1 candidates | 0.687 |
| nearest-neighbour distance among the training compositions, median | 0.174 |
| nearest-neighbour distance among the training compositions, maximum | 1.163 |
| largest \|S_ANE\| of the two nearest training entries of the Fe-Mn candidates | 1.361 uV/K |

The two nearest training entries are Fe0.625Mn0.125Ga0.250 (\|S_ANE\| = 1.361)
and Fe0.500Co0.125Mn0.125Al0.250 (0.134), both with 12.5 at.% Mn.

## Outputs

| File | Contents |
| --- | --- |
| `results/cycle1_candidate_distances.csv` | per cycle-1 candidate: nearest and second-nearest training compositions, their distances, Mn fractions and \|S_ANE\| |
| `results/distance_summary.json` | the values in the table above |

`tests/test_input_space_distance.py` recomputes the summary and pins it.
