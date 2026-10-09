# Compatibility and migration guide (0.2.x)

This modernization intentionally changes behavior that was broken in the 2017 prototype.

## Supported environment

- Python 3.10–3.13; NumPy 1.26/2.x; pandas 2.x; scikit-learn 1.3+.
- Core: NumPy, pandas, SciPy, scikit-learn.
- Optional: `tsfresh`, `patsy` (B-splines), `statsmodels` (LOWESS).
- Legacy `MLSD/Transformers/pyFDA/` remains in the source tree but is not on the default import path. It has not been certified against current Python versions.

## API behavior

| Prior behavior | 0.2.x |
|---|---|
| Printed errors for bad inputs | Raises `TypeError` or `ValueError` |
| Ragged data coerced by `np.asarray` | One object-array slot per observation |
| Series transformer silently dropped NaN feature columns | Fixed 21-column schema; insufficient data produces NaN |
| Bag transformer often transposed rows/columns | One sample per row and named feature columns |
| Text TF-IDF sometimes refitted on test data | Vocabulary frozen after `fit` |
| Image extraction referred to undefined `x` and `args` | Bounded per-channel image statistics |
| `SDataFrame.join` was a broken classmethod | Instance method returning a new aligned frame |
| `C_resample` was a broken classmethod | Instance method returning a resampled copy |
| `ffill`/`bfill` stored methods instead of calling them | Fill operations update stored pandas Series |
| `activeTrans` skipped the second column | Includes all columns, fits exactly once |

`BasicSeries` outputs value, first-difference, and second-difference statistics.
`BasicBag(mesh=True)` treats the observation order as meaningful. Use
`BasicBag(mesh=False)` for genuinely unordered bags.

`FPCA` is an **approximation** using interpolation to a shared time grid and
scikit-learn PCA, not the formerly imported elastic `fdasrsf` algorithm.
Do not compare its coefficients directly with historical elastic FPCA results.

## Evaluation rules

1. Split before fitting: `split_frame(...)` or `PurgedWalkForwardSplit`.
2. Fit `SDataFrame` transformers on training rows only.
3. Apply the fitted training frame's `transform(holdout_frame)`.
4. Fit imputation, scaling, selection, and estimators on training rows only.
5. For forward-looking labels, choose a purge `gap` at least as large as
   the maximum known label horizon; an ordinary temporal holdout alone may
   still leak via overlapping future windows.
6. Index alignment is strict and positional; duplicate or misordered labels
   raise rather than silently mixing responses across samples.

`extracted_features` remains a convenient fit-and-transform shortcut on the
whole frame. **Do not use it on test data or before a train/test split.**
