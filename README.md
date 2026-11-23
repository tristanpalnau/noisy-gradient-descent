# Noisy gradient descent: escaping a hilltop

A pilot experiment from my Fall 2026 Directed Reading Program in mathematics at Kansas State University, studying gradient descent under noisy gradients.

## Question

Gradient descent moves by subtracting a step size times the gradient. At any point where the gradient is zero, it does not move at all, so a hilltop looks the same as a valley floor. In machine learning, gradients are usually estimated from small random samples of data, so they carry noise. This pilot asks: **if gradient descent starts on a hilltop, how long does noise take to push it off, and how much does the noise level matter?**

## Setup

- Function: f(x) = (x² − 1)², a double well with a local maximum at x = 0 and minima at x = ±1
- Start: x = 0, where the true gradient is zero
- Update: x ← x − α (f′(x) + σ·ξ), with ξ drawn from a standard normal distribution
- Step size: α = 0.01
- Noise levels: σ ∈ {0.001, 0.01, 0.1, 1}
- 1,000 independent runs per noise level; a run has escaped once |x| ≥ 0.5
- Random seed fixed at 1 for reproducibility

## Results

| Noise level σ | Median steps to escape |
|---|---|
| 0.001 | 257 |
| 0.01 | 199 |
| 0.1 | 141 |
| 1 | 82 |

![Median steps to escape vs noise level](my_pilot_figure.png)

Without noise (σ = 0), the iterate never leaves x = 0. With noise, each tenfold increase in σ saves only about 60 steps. Once noise nudges x away from 0, the true gradient takes over and pushes it away quickly, so the size of the first nudge matters relatively little.

## Run it

```
pip install -r requirements.txt
python research.py
```

Prints the four medians and saves `my_pilot_figure.png`.

## Next steps

Planned experiments, not yet run: how noise spreads the iterates around a minimum on a two-dimensional quadratic, and how escape time from a true saddle point changes with dimension (up to 1,000 dimensions).
