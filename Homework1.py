"""
CS5710 - Machine Learning
Homework 1
Student: Menaka Naga Sai Pothina
University of Central Missouri

This file contains the implementation of Linear Regression using:
1. Closed-form solution (Normal Equation)
2. Gradient Descent

Dataset: y = 3 + 4x + ε, where ε ~ Gaussian noise
"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Step 1: Generate synthetic dataset
# -----------------------------

np.random.seed(42)  # For reproducibility
X = np.random.uniform(0, 5, 200)        # 200 samples, x ∈ [0,5]
epsilon = np.random.normal(0, 1, 200)   # Gaussian noise
y = 3 + 4 * X + epsilon                 # True function with noise

# Add bias column (for θ0)
X_b = np.c_[np.ones((200, 1)), X]       # shape: (200, 2)

# -----------------------------
# Step 2: Closed-form solution (Normal Equation)
# -----------------------------

theta_closed = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

print("Closed-form solution:")
print("Intercept (θ0):", theta_closed[0])
print("Slope (θ1):", theta_closed[1])

# Predictions using closed-form solution
y_pred_closed = X_b.dot(theta_closed)

# -----------------------------
# Step 3: Gradient Descent implementation
# -----------------------------

# Initialize parameters
theta_gd = np.zeros(2)  # [θ0, θ1]
eta = 0.05              # learning rate
iterations = 1000
m = len(y)              # number of samples
losses = []             # track MSE loss for plotting

# Gradient Descent Loop
for i in range(iterations):
    # Predictions
    y_pred = X_b.dot(theta_gd)

    # Compute residuals
    residuals = y - y_pred

    # Compute gradient
    gradients = -(2/m) * X_b.T.dot(residuals)

    # Update rule
    theta_gd = theta_gd - eta * gradients

    # Compute Mean Squared Error (MSE) at each iteration
    mse = np.mean(residuals ** 2)
    losses.append(mse)

print("\nGradient Descent solution after 1000 iterations:")
print("Intercept (θ0):", theta_gd[0])
print("Slope (θ1):", theta_gd[1])

# -----------------------------
# Step 4: Plot dataset and fitted lines
# -----------------------------

plt.figure(figsize=(8,6))
plt.scatter(X, y, alpha=0.5, label="Data points")
plt.plot(X, y_pred_closed, color='red', label="Closed Form Fit")
plt.plot(X, X_b.dot(theta_gd), color='green', linestyle="--", label="Gradient Descent Fit")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression: Closed Form vs Gradient Descent")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# Step 5: Plot loss curve for Gradient Descent
# -----------------------------

plt.figure(figsize=(8,6))
plt.plot(losses, color='blue')
plt.xlabel("Iterations")
plt.ylabel("MSE Loss")
plt.title("Gradient Descent Convergence")
plt.grid(True)
plt.show()

# -----------------------------
# Step 6: Comparison Result
# -----------------------------

print("\nComparison:")
print(f"Closed Form → Intercept: {theta_closed[0]:.3f}, Slope: {theta_closed[1]:.3f}")
print(f"Gradient Descent → Intercept: {theta_gd[0]:.3f}, Slope: {theta_gd[1]:.3f}")

print("\nObservation: Both methods converge to nearly the same parameters. "
      "Gradient Descent shows step-by-step convergence (visible in loss curve), "
      "while the Normal Equation computes the solution directly.")
