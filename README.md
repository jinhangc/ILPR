# ILPR

This repository contains the implementation of Iterative Local Polynomial Regression estimator for semiparametric dynamic pricing.

Contents:

- `code/simulation/`: synthetic experiment code.
- `code/semi_real/`: semi-real experiment code.
- `data/competition_data_2023_09_25-2.csv`: real-data source used in the semi-real experiments.
- `outputs/simulation_comparison/`: synthetic summary CSV and plot.
- `outputs/semi_real_comparison/`: semi-real summary CSV, main comparison plot, histogram plot, and histogram summary CSV.

Main entry points:

- Synthetic experiments:
  - `python code/simulation/run_methods.py`
  - `python code/simulation/plot_methods.py`
- Semi-real experiments:
  - `python code/semi_real/run_real_methods.py`
  - `python code/semi_real/plot_real_methods.py`
  - `python code/semi_real/plot_real_improvement_histogram.py`


## Abstract:
We study a contextual dynamic pricing problem under a semiparametric demand model, where the purchase probability takes the form $1 - F(p - m(\bx))$ with mean utility $m(\bx)$ given the product feature $\bx$ and the market noise distribution $F$. Existing approaches for this model either suffer from suboptimal regret rates or require strong structural assumptions that limit flexibility. 

We propose a stagewise greedy pricing algorithm that iteratively refines the estimate of the unknown distribution $F$ using local polynomial regression, while pricing greedily using the currently available estimates. Our algorithm exploits feature diversity to reuse endogenous samples collected during exploitation for nonparametric function estimation, avoiding the need for costly global random exploration in the previous literature. 

We derive a general upper regret bound for a given utility estimator $\hat{m}$ and  provide the explicit regret rates for $m$ in  linear, nonparametric additive, and sparse linear classes. In particular, for linear function class, we establish a regret bound of order $T^{\max\{1/2,\,3/(2\beta+1)\}}$, where $\beta$ denotes the smoothness of $F$ and $T$ represents the time horizon. This improves upon the best known rates for semiparametric contextual pricing and attains the parametric $\sqrt{T}$ rate when $\beta \ge 5/2$. We complement our upper bound with a matching lower bound, demonstrating the optimality of our approach. Numerical experiments corroborate the theoretical findings and highlight the practical benefits of iterative refinement.
