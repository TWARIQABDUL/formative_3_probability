"""
Verifies Kolade's hand-calculated Gradient Descent Iteration 2 against the same
math done in Python (matrix multiplication, not scalar shortcuts), continuing
the relay from Arnold's Iteration 1 output:
    m1 = [-1.45, 0.87]
    b1 = [0.99, 0.89]
"""

import numpy as np

# ---- Given data (unchanged across iterations) ----
X = np.array([[1, 3],
              [4, 10]])
y = np.array([5, 6])
alpha = 0.01
n = len(y)

# ---- Handed off from Arnold's Iteration 1 ----
m = np.array([-1.45, 0.87])
b = np.array([0.99, 0.89])

print("=== Step 1: Predict y_hat ===")
y_hat = X @ m + b
print("X @ m =", X @ m)
print("y_hat = X @ m + b =", y_hat)

print("\n=== Step 2: Error ===")
error = y_hat - y
print("error = y_hat - y =", error)

print("\n=== Step 3: Cost (MSE) ===")
J = (1 / n) * np.sum(error ** 2)
print("J =", J)

print("\n=== Step 4: Gradient w.r.t. m ===")
grad_m = (2 / n) * (X.T @ error)
print("X.T =\n", X.T)
print("grad_m = (2/n) * X.T @ error =", grad_m)

print("\n=== Step 5: Gradient w.r.t. b ===")
grad_b = (2 / n) * error
print("grad_b = (2/n) * error =", grad_b)

print("\n=== Step 6 & 7: Update m and b ===")
m_new = m - alpha * grad_m
b_new = b - alpha * grad_b
print("m_new = m - alpha * grad_m =", m_new)
print("b_new = b - alpha * grad_b =", b_new)

# ---- Compare against Kolade's hand calculations ----
print("\n=== Check against hand calculations ===")
expected_y_hat = np.array([2.15, 3.79])
expected_error = np.array([-2.85, -2.21])
expected_J = 6.5033
expected_grad_m = np.array([-11.69, -30.65])
expected_grad_b = np.array([-2.85, -2.21])
expected_m_new = np.array([-1.3331, 1.1765])
expected_b_new = np.array([1.0185, 0.9121])

checks = [
    ("y_hat", y_hat, expected_y_hat),
    ("error", error, expected_error),
    ("J", J, expected_J),
    ("grad_m", grad_m, expected_grad_m),
    ("grad_b", grad_b, expected_grad_b),
    ("m_new", m_new, expected_m_new),
    ("b_new", b_new, expected_b_new),
]

all_match = True
for name, computed, expected in checks:
    match = np.allclose(computed, expected, atol=1e-3)
    all_match &= match
    status = "MATCH" if match else "MISMATCH"
    print(f"{name:8s}: computed={np.round(computed, 4)}  expected={expected}  -> {status}")

print("\nRESULT:", "All hand calculations verified correct!" if all_match else "Some values don't match — recheck the flagged rows above.")

print("\n=== Handoff to Crispin (Iteration 3 input) ===")
print("m2 =", np.round(m_new, 4))
print("b2 =", np.round(b_new, 4))
