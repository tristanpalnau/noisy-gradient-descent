import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(1)


def d(x):
    return 4*x**3 - 4*x


def escape_time(a, sigma, max_steps=100000):
    x = 0.0
    for step in range(max_steps):
        if abs(x) >= 0.5:
            return step
        x = x - a * (d(x) + sigma * rng.normal())
    return max_steps


# Collect the results
sigmas = [0.001, 0.01, 0.1, 1]
medians = []

for sigma in sigmas:
    times = [escape_time(0.01, sigma) for _ in range(1000)]
    medians.append(np.median(times))

print(medians)

# Make the figure
plt.rcParams["font.family"] = "Arial"
plt.figure(figsize=(5, 2.5))
plt.plot(sigmas, medians, marker="o")
plt.xscale("log")
plt.xlabel("Noise level σ (log scale)")
plt.ylabel("Median steps to |x| ≥ 0.5")
plt.tight_layout()
plt.savefig("my_pilot_figure.png", dpi=300)