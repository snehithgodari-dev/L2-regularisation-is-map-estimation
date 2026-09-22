
# L2 Regularization = MAP Estimation

L2 regularization isn't a trick to prevent overfitting , it's MAP estimation
with a Gaussian prior. This repo derives that algebraically, implements MLE
and MAP estimation for linear regression from scratch in NumPy, verifies the
MAP solution against `sklearn.linear_model.Ridge` across a full
regularization path, and applies the Bayesian interpretation to a real
dataset to derive lambda from an explicit prior belief instead of a
cross-validation search.

## Derivation

Starting point: Bayes' theorem, `P(θ|X) ∝ P(X|θ) · P(θ)`.

1.MLE ignores the prior, maximizes `P(X|θ)` only. Closed form (OLS):
  `θ̂ = (XᵀX)⁻¹Xᵀy`
2.MAP includes the prior. With a Gaussian prior `θ ~ N(0, τ²)`, the
  log-posterior is `log P(X|θ) − (1/2τ²)‖θ‖²` that second term is exactly
  the L2 penalty. Closed form: `θ̂ = (XᵀX + λI)⁻¹Xᵀy`, where `λ = σ²/τ²`.

Ridge regression is MAP estimation with a Gaussian prior same
equation, different name.

Other regularizers have the same interpretation:
- Lasso (L1) = MAP with a Laplace prior , sparsity from the sharp peak at zero
- Early stopping ≈ L2 is training trajectory implicitly encodes regularization
- Dropout ≈ approximate Bayesian inference (Hinton's original paper

## Results

 Check                                                                                                         Result 

MAP (Ridge) vs. `sklearn.linear_model.Ridge`, full path (λ = 0 → 500)                               | max abs diff `< 1e-10` (machine precision) 
MLE vs. MAP at λ = 0                                                                                | identical (max abs diff `< 1e-10`) 
| Dataset used                                                         | sklearn diabetes regression set (442 patients, 10 features) — real, not synthetic |

Applied: deriving λ from a stated prior belief, not cross-validation

| Method                                  | Test MSE                                               | Test R² |

| MLE (λ = 0)                             | 2900.19                                                | 0.4526 |
| MAP, naive prior (τ = 1)                | 4180.03                                                | 0.2110 |
| MAP, grounded prior (τ = std(y))        | 2895.44                                                | 0.4535 |
| Ridge, CV-searched α (`RidgeCV`)        | 2891.21                                                | 0.4543 |

An ungrounded prior guess (τ = 1) badly over-regularizes. Anchoring τ to the
actual scale of the problem (`τ = std(y)` "a standardized feature moving
the outcome by up to one target-standard-deviation is a large but plausible
effect") gets within 0.1% R² of what a full cross-validation grid search
finds without running a search at all. This is the practical payoff of the
MAP interpretation: λ isn't a knob to tune, it's a stand-in for a stated
belief about the weights.

Numbers are from one train/test split (`random_state=42`); re-run to see
them vary slightly.

## Repo structure

```
l2-is-map-estimation/
├── README.md
├── mle_map_regression.py       # MLE + MAP implementation, sklearn Ridge verification
├── mle_map_applied.py          # applied test: prior-derived lambda vs RidgeCV
├── mle_map_explained.ipynb     # full walkthrough with derivation + commentary
└── mle_map_code_only.ipynb     # clean code, no explanation
```

## How to run
 open either `.ipynb` in Jupyter/Colab and run all cells.

