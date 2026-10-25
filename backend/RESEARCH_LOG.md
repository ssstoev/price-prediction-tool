## 2026-08-08 — Hypothesis: Target encoding neighbourhood will improve RMSE 

**Motivation**: Error analysis from run `run_id` shows that the model captures BETWEEN neighbourhood variance well rather well but fails to capture the WITHIN-neighbourhood one. Including neighbourhood std/mean/count + smoothing could improve the results

**What to change**: Add `std_neighbourhood`, `avg_neighbourhood`,`count_neighbourhood` and smoothen these values using the general std/avg/mean.

**Expected outcome**: RMSE drops ~5-10% on price_m2 target.

**Result**: [fill after run]  
Run IDs: `abc123`, `def456`  
Outcome: RMSE went from X → Y. Neighbourhood cardinality still too high,
smoothing helped more (see run `def456` tags).

---

## 2026-08-08 — Hypothesis: `floor_ratio` feature will reduce within-building variance

**Motivation**: Raw `floor` number is confounded by building height. Floor 7 of 8 is a top-floor flat; floor 7 of 20 is mid-range. The model cannot distinguish these today.

**What to change**: Add `floor_ratio = floor / building_total_floors` as a feature in the relevant feature sets.

**Expected outcome**: RMSE drops modestly; feature importance of `floor` increases after the transformation.

**Result**: [fill after run]  
Run IDs: []  
Outcome: []

---

## 2026-08-08 — Hypothesis: `size_per_room` feature captures flat quality/density

**Motivation**: A 40m² 1-room studio and a 40m² 2-room flat are priced differently. The model conflates them because it sees size and rooms separately.

**What to change**: Add `size_per_room = size_m2 / nr_of_rooms` as a feature.

**Expected outcome**: Small RMSE improvement; stronger signal for price_m2 target since density correlates with quality tier.

**Result**: [fill after run]  
Run IDs: []  
Outcome: []

---

## 2026-08-08 — Hypothesis: Residual modeling (predict deviation from neighbourhood mean)

**Motivation**: The model wastes capacity learning the neighbourhood-level price floor. Removing it lets the remaining features explain finer-grained variance.

**What to change**: Two variants to compare:
1. Simple: change target to `log(price_m2) - log(neighbourhood_mean_price_m2)`; add neighbourhood mean back at inference.
2. Two-stage stacking: Stage 1 predicts `neighbourhood_mean`; Stage 2 predicts residual using all other features.

**Expected outcome**: Stage 2 model achieves lower RMSE on the residual, and reconstructed price RMSE beats the baseline end-to-end.

**Result**: [fill after run]  
Run IDs: []  
Outcome: []

---

## 2026-08-08 — Hypothesis: Expanding XGBoost regularization grid will reduce overfitting

**Motivation**: Current grid only tunes `n_estimators`, `max_depth`, `learning_rate`, `colsample_bytree`. Regularization params are left at defaults, likely causing overfitting on small neighbourhoods.

**What to change**: Add to XGBoost `param_grid`: `min_child_weight`, `subsample`, `reg_alpha` (L1), `reg_lambda` (L2).

**Expected outcome**: CV RMSE ≈ test RMSE (gap closes); absolute RMSE drops 3-8%.

**Result**: [fill after run]  
Run IDs: []  
Outcome: []

---

## 2026-08-08 — Hypothesis: LightGBM will outperform XGBoost with less tuning

**Motivation**: LightGBM's `num_leaves`-based growth is a different bias-variance tradeoff than `max_depth`. On tabular real-estate data it frequently yields 5-15% RMSE reduction with equivalent effort.

**What to change**: Add LightGBM as a model in `config.py`; tune `num_leaves`, `min_child_samples`, `learning_rate`, `colsample_bytree`.

**Expected outcome**: LightGBM beats XGBoost on cv_rmse without requiring a larger grid.

**Result**: [fill after run]  
Run IDs: []  
Outcome: []

---

## 2026-08-08 — Hypothesis: CatBoost with native categorical handling avoids TargetEncoder leakage

**Motivation**: `TargetEncoder` fitted inside cross-validation folds can still leak target statistics if not implemented carefully. CatBoost's ordered target statistics are leak-free by construction.

**What to change**: Add CatBoost as a model; pass `neighbourhood` as a categorical column directly without pre-encoding.

**Expected outcome**: If current TargetEncoder leaks, CatBoost test RMSE < train RMSE gap shrinks. May also improve absolute RMSE.

**Result**: [fill after run]  
Run IDs: []  
Outcome: []

---

## 2026-08-08 — Hypothesis: kNN imputation outperforms median imputation for floor features

**Motivation**: `floor` and `building_total_floors` nulls are not random — flats in the same building share values. A global median imputation ignores this structure. (WIP note already in features.py.)

**What to change**: Replace `SimpleImputer(strategy="median")` with `KNNImputer(n_neighbors=5)` in the numeric pipeline.

**Expected outcome**: Imputed values are more accurate; small downstream RMSE improvement, especially for listings with missing floor data.

**Result**: [fill after run]  
Run IDs: []  
Outcome: []

---

## 2026-08-08 — Hypothesis: Per-neighbourhood outlier filtering is cleaner than global bounds

**Motivation**: Hardcoded global bounds (`price_m2 > 600`, `< 15000`) in `data_load.py` are arbitrary. A 2000€/m2 flat is normal in some neighbourhoods and an outlier in others. Global filtering discards valid high-end listings.

**What to change**: Replace global bounds with per-neighbourhood IQR filtering (e.g., remove rows outside `[Q1 - 1.5×IQR, Q3 + 1.5×IQR]` per neighbourhood).

**Expected outcome**: Training set is cleaner without losing valid data; RMSE on high-price neighbourhoods improves.

**Result**: [fill after run]  
Run IDs: []  
Outcome: []

---

## 2026-08-08 — Hypothesis: Alternative target transform (Box-Cox / Yeo-Johnson) reduces residual skew

**Motivation**: The fixed `log` transform assumes multiplicative noise. If the optimal lambda differs from 1 (log), residual errors are larger than necessary.

**What to change**: Try `PowerTransformer(method="yeo-johnson")` as a target transform option in the experiment config alongside the existing `log`.

**Expected outcome**: Residual distribution is more Gaussian; MAE and RMSE both improve slightly.

**Result**: [fill after run]  
Run IDs: []  
Outcome: []