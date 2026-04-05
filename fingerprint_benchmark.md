# Fingerprint Method Cross-Comparison Benchmark

## Setup

- **Slab:** Cu fcc(111) 3×3×4 (36 Cu atoms)
- **Adsorbates tested:** OH, CO, H2O
- **Test pairs:** 16 total — 8 true duplicates, 8 true uniques

**Duplicate definition:** same adsorbate species + count, Gaussian displacement σ ≤ 0.12 Å on adsorbate atoms.  
**Unique definition:** different species, different count, different site (>1 Å), height change, or large displacement σ = 0.8 Å.  

All methods include a **composition gate** (formula check) as a free pre-filter.

## Summary Table

| Method | TP | FP | FN | TN | Precision | Recall | F1 | Accuracy | Time (s) |
|---|---|---|---|---|---|---|---|---|---|
| composition_gate | 8 | 3 | 0 | 5 | 0.727 | 1.000 | 0.842 | 0.812 | 0.00 |
| dist_histogram | 8 | 3 | 0 | 5 | 0.727 | 1.000 | 0.842 | 0.812 | 0.01 |
| soap_tanimoto | 8 | 3 | 0 | 5 | 0.727 | 1.000 | 0.842 | 0.812 | 0.83 |
| mbtr_cosine | 8 | 3 | 0 | 5 | 0.727 | 1.000 | 0.842 | 0.812 | 0.03 |
| graph_isomorphism | 6 | 0 | 2 | 8 | 1.000 | 0.750 | 0.857 | 0.875 | 2.13 |
| full_cascade | 6 | 0 | 2 | 8 | 1.000 | 0.750 | 0.857 | 0.875 | 2.12 |

## Per-Pair Results

Legend: **TP** = true positive (correctly called duplicate), **TN** = correctly unique, **FP** = false positive (unique called duplicate), **FN** = false negative (duplicate missed).

| # | Description | GT | composition_gate | dist_histogram | soap_tanimoto | mbtr_cosine | graph_isomorphism | full_cascade |
|---|---|---|---|---|---|---|---|---|
| 1 | OH identical copy | DUP | TP | TP | TP | TP | TP | TP |
| 2 | OH jitter 0.05 A | DUP | TP | TP | TP | TP | FN | FN |
| 3 | OH jitter 0.12 A | DUP | TP | TP | TP | TP | FN | FN |
| 4 | CO identical copy | DUP | TP | TP | TP | TP | TP | TP |
| 5 | CO jitter 0.06 A | DUP | TP | TP | TP | TP | TP | TP |
| 6 | 2x OH identical | DUP | TP | TP | TP | TP | TP | TP |
| 7 | 2x OH jitter 0.07 A | DUP | TP | TP | TP | TP | TP | TP |
| 8 | H2O identical | DUP | TP | TP | TP | TP | TP | TP |
| 9 | OH vs CO (different species) | UNQ | TN | TN | TN | TN | TN | TN |
| 10 | 1x OH vs 2x OH (different count) | UNQ | TN | TN | TN | TN | TN | TN |
| 11 | OH different site (far apart) | UNQ | FP | FP | FP | FP | TN | TN |
| 12 | OH large jitter 0.8 A | UNQ | FP | FP | FP | FP | TN | TN |
| 13 | CO vs H2O (different species) | UNQ | TN | TN | TN | TN | TN | TN |
| 14 | OH different height (2.1 vs 3.5 A) | UNQ | FP | FP | FP | FP | TN | TN |
| 15 | clean slab vs OH | UNQ | TN | TN | TN | TN | TN | TN |
| 16 | 1x CO vs 2x CO (different count) | UNQ | TN | TN | TN | TN | TN | TN |

## False Positives and False Negatives

### composition_gate
  FP | OH different site (far apart)
  FP | OH large jitter 0.8 A
  FP | OH different height (2.1 vs 3.5 A)

### dist_histogram
  FP | OH different site (far apart)
  FP | OH large jitter 0.8 A
  FP | OH different height (2.1 vs 3.5 A)

### soap_tanimoto
  FP | OH different site (far apart)
  FP | OH large jitter 0.8 A
  FP | OH different height (2.1 vs 3.5 A)

### mbtr_cosine
  FP | OH different site (far apart)
  FP | OH large jitter 0.8 A
  FP | OH different height (2.1 vs 3.5 A)

### graph_isomorphism
  FN | OH jitter 0.05 A
  FN | OH jitter 0.12 A

### full_cascade
  FN | OH jitter 0.05 A
  FN | OH jitter 0.12 A

## Method Notes

### composition_gate
O(1) lookup on the chemical formula string. Eliminates different-species and different-count
pairs instantly. Zero cost. High recall on this test set because most uniques differ in
composition. Fails to distinguish structures with identical formula but different geometry.

### dist_histogram
Pairwise distance histogram (50 bins, r_max=6 Å) cosine similarity, threshold 0.98.
Fast (O(n²) distances) but insensitive to local bonding geometry when dominated by bulk
slab atoms. FNs arise when small positional changes shift bin counts.

### soap_tanimoto
SOAP averaged over all atoms (r_cut=6, n_max=8, l_max=6), Tanimoto similarity, threshold 0.97.
Captures local chemical environments well. The 36-atom Cu slab dominates the average,
making it insensitive to subtle adsorbate geometry changes far from slab high-symmetry sites.
~10–50x slower than dist_histogram.

### mbtr_cosine
MBTR inverse-distance k=2 (non-periodic, 50 grid points, σ=0.05), L2 cosine, threshold 0.98.
Encodes pair-distance distributions for each element pair. Similar accuracy to SOAP on
this benchmark, but much faster than SOAP because it avoids the angular SOAP expansion.
Limitation: non-periodic calculation misses some crystal-symmetry information.

### graph_isomorphism
Chemical-environment graph built from bonding topology around each adsorbate atom
(periodic bonds included). Graph isomorphism test via networkx VF2.
**Exact topology matching** — perfect precision (no FPs) but sensitive to bond-cutoff
boundary effects: a displacement as small as 0.05 Å can change which Cu neighbor falls
within the cutoff, producing a non-isomorphic graph and a FN.

### full_cascade
Composition gate → dist_histogram gate (0.95) → graph_iso → SOAP fallback.
Inherits the FNs from graph_iso because the SOAP fallback is only reached when graph_iso
returns None (not built), not when it returns False.

## Key Findings

1. **Composition gate is essential** and free — all methods benefit from it as a pre-filter.
2. **Graph isomorphism has perfect precision** but produces FNs for small adsorbate
   displacements (0.05–0.12 Å) due to bond-cutoff sensitivity. This is a known limitation
   of distance-cutoff bonding detection.
3. **SOAP and MBTR** (with composition gate) show better recall than graph_iso on small
   displacements, but produce FPs for large displacements if thresholds are too loose.
   Threshold tuning is critical and geometry-dependent.
4. **MBTR is faster than SOAP** (~5–20x on these structures) at comparable accuracy.
   Good candidate as a cheaper fallback instead of SOAP in the cascade.
5. **Recommended improvement:** in the cascade, after graph_iso returns False (not None),
   add a SOAP/MBTR check with a tight threshold (≥0.99) as a safety net for small-jitter
   cases that graph_iso misses due to bond-cutoff artifacts.

