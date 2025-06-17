### Description:

A/B testing is a method of comparing two versions (A and B) of a variable to determine which one performs better. It's commonly used in product design, marketing, and UX optimization. In this project, we create a Python-based framework to simulate and analyze an A/B test using statistical hypothesis testing (e.g., t-test or z-test for proportions).

- Simulates conversion data
- Applies both T-Test and Z-Test for proportions
- Calculates and interprets p-values, conversion lift, and significance
- Visualizes the results

## A/B Testing: Conversion Rate Analysis with Python

This script performs a simple A/B test to compare the conversion rates of two versions of a web page: a control group (A) and a variant group (B). It demonstrates both t-test and z-test statistical approaches to evaluate the significance of observed differences.

### Imports

```python
import numpy as np
from scipy.stats import ttest_ind, norm
import matplotlib.pyplot as plt
```

* `numpy`: For numerical operations and simulating data.
* `scipy.stats`: For statistical tests.
* `matplotlib.pyplot`: For visualizing conversion rates.

### Simulate Conversion Data

```python
np.random.seed(42)

group_a = np.random.binomial(1, 0.12, 1000)
group_b = np.random.binomial(1, 0.15, 1000)
```

* Simulates binary conversion outcomes (0 or 1) for 1000 users in each group.
* Group A has a 12% conversion rate, Group B has a 15% conversion rate.

### Calculate Conversion Rates and Lift

```python
conv_rate_a = np.mean(group_a)
conv_rate_b = np.mean(group_b)

print(f"Conversion Rate A (Control): {conv_rate_a:.3f}")
print(f"Conversion Rate B (Variant): {conv_rate_b:.3f}")
print(f"Lift: {(conv_rate_b - conv_rate_a) / conv_rate_a * 100:.2f}%")
```

* Calculates average conversion rates for each group.
* Computes the relative lift between the groups.

### Option 1: T-Test

```python
t_stat, p_value_t = ttest_ind(group_a, group_b)
print(f"\nT-Test p-value: {p_value_t:.4f}")
```

* A two-sample t-test checks if the means of the two independent samples are significantly different.
* More suitable for normally distributed continuous metrics like revenue or time.

### Option 2: Z-Test for Proportions

```python
n_a, n_b = len(group_a), len(group_b)
success_a, success_b = sum(group_a), sum(group_b)

p_pool = (success_a + success_b) / (n_a + n_b)
se = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
z = (conv_rate_b - conv_rate_a) / se
p_value_z = 1 - norm.cdf(z)

print(f"Z-Test z-score: {z:.2f}")
print(f"Z-Test one-tailed p-value: {p_value_z:.4f}")
```

* Suitable for binary outcomes (e.g., converted or not).
* Computes pooled standard error and z-score for the difference in proportions.
* Calculates one-tailed p-value to test if variant B performs significantly better.

### Visualization

```python
plt.figure(figsize=(8, 4))
plt.bar(["Control (A)", "Variant (B)"], [conv_rate_a, conv_rate_b], color=["blue", "green"])
plt.title("A/B Test Conversion Rates")
plt.ylabel("Conversion Rate")
plt.grid(axis='y')
plt.tight_layout()
plt.show()
```

* Creates a simple bar chart comparing conversion rates of the two groups.

### Summary

This script is a practical example of applying A/B testing methodology using simulated data. It provides two statistical approaches to determine the significance of observed conversion differences and visualizes the results for better understanding.

**Note**: While this example uses simulated data, the same approach applies to real-world A/B test analysis.
