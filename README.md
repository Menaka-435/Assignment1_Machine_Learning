# CS5710 - Machine Learning
## Homework 1: Linear Regression Implementation

**Student:** Menaka Naga Sai Pothina  
**University:** University of Central Missouri  
**Course:** CS5710 - Machine Learning, Fall 2025  

---

## Overview

This assignment demonstrates **Linear Regression** implemented using:

1. **Closed-form solution (Normal Equation)**  
2. **Gradient Descent from scratch**  

The goal is to fit a linear model to synthetic data generated from:

\[
y = 3 + 4x + \epsilon
\]

where \(\epsilon\) is Gaussian noise. The performance and convergence of the two methods are compared.

---

## Dataset Generation

- 200 data points generated randomly with \(x \in [0,5]\)
- Gaussian noise added (\(\mu=0, \sigma=1\))  
- Bias column added to `X` for intercept term in linear regression.

```python
np.random.seed(42)
X = np.random.uniform(0, 5, 200)
epsilon = np.random.normal(0, 1, 200)
y = 3 + 4 * X + epsilon
