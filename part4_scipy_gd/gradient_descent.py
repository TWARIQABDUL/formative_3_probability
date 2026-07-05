"""
Part 4: Gradient Descent in Code
Formative 3 - Optimization & Deck Lead: Crispin Hebert Hirwa

This module converts the Part 3 manual relay calculations into working Python,
picking up EXACTLY where Kolade's Iteration 2 left off (relay: Arnold -> Kolade
-> Crispin -> AbdalazizI).

It does two things side by side, on purpose:

1. MANUAL derivative  -> the exact chain-rule formula the team derived by hand
                          (dJ/dm and dJ/db), coded explicitly, step by step.
2. SCIPY derivative    -> a generic numerical gradient computed by SciPy that
                          takes ANY cost function and returns its derivative,
                          without us hand-deriving anything for it.

We then compare the two on every iteration to PROVE the manual chain-rule
math is correct. This satisfies the rubric requirement to "implement a
function that accepts an equation and computes its derivative" using SciPy,
while still anchoring everything to the group's actual hand derivation.

CONFIRMED MODEL (from Kolade's verify_iteration2.py -- matches Arnold's
Iteration 1 too):
    y_hat = X @ m + b
    b is a PER-SAMPLE bias VECTOR (shape (n,)), added elementwise -- NOT a
    single scalar broadcast. This was verified by reproducing Kolade's
    Iteration 2 numbers exactly:
        m2 = [-1.3331, 1.1765], b2 = [1.0185, 0.9121]
"""

import numpy as np
from scipy.optimize import approx_fprime

# ---------------------------------------------------------------------------
# 1. GIVEN DATA (unchanged across the whole relay, per Kolade's script)
# ---------------------------------------------------------------------------
X = np.array([[1.0, 3.0],
              [4.0, 10.0]])          # shape (n=2 samples, 2 features)
y = np.array([5.0, 6.0])             # shape (2,)
ALPHA = 0.01                          # learning rate (matches team's relay)

# ---------------------------------------------------------------------------
# HANDOFF FROM KOLADE (Iteration 2 output) -- this is Crispin's starting point
# ---------------------------------------------------------------------------
INITIAL_M = np.array([-1.3331, 1.1765])   # m2
INITIAL_B = np.array([1.0185, 0.9121])    # b2 (PER-SAMPLE vector, not scalar)

N_SAMPLES = X.shape[0]


# ---------------------------------------------------------------------------
# 2. FORWARD PASS (prediction)
# ---------------------------------------------------------------------------
def predict(m, b, X=X):
    """
    y_hat = X @ m + b, where b is a per-sample vector added elementwise.
    Explicit matrix multiplication, not a scalar loop.
    """
    return X @ m + b


# ---------------------------------------------------------------------------
# 3. COST FUNCTION: Mean Squared Error
# ---------------------------------------------------------------------------
def mse_cost(params, X=X, y=y):
    """
    params = [m1, m2, b1, b2] packed into one vector so SciPy can treat this
    as a single multivariate function f(params) -> scalar.
    This is the "equation" SciPy will differentiate in section 4.

    NOTE: b is a per-sample vector (b1, b2), matching the team's convention
    confirmed from Kolade's Iteration 2 script -- not a single scalar.
    """
    m = params[:2]
    b = params[2:4]
    y_hat = predict(m, b, X)
    error = y_hat - y
    return np.mean(error ** 2)


# ---------------------------------------------------------------------------
# 4. SCIPY DERIVATIVE: generic numerical gradient of ANY cost function
# ---------------------------------------------------------------------------
def scipy_gradient(params, X=X, y=y, epsilon=1e-6):
    """
    Uses SciPy's approx_fprime to numerically differentiate mse_cost
    with respect to every entry in `params` (m1, m2, b1, b2).

    This function does NOT know the chain rule was ever derived by hand --
    it works for ANY cost function you hand it. That's the "accepts an
    equation and computes its derivative" requirement.
    """
    grad = approx_fprime(params, mse_cost, epsilon, X, y)
    return grad[:2], grad[2:4]   # split back into (dJ/dm, dJ/db)


# ---------------------------------------------------------------------------
# 5. MANUAL DERIVATIVE: explicit chain rule (matches Arnold's derivation)
# ---------------------------------------------------------------------------
def manual_gradient(m, b, X=X, y=y):
    """
    Chain rule, shown step by step (no skipped arithmetic) -- matches
    Arnold's Iteration 1 derivation and Kolade's Iteration 2 script exactly.

    J(m,b) = (1/n) * sum( (y_hat_i - y_i)^2 )

    Step 1: y_hat = X @ m + b                     <- b is per-sample vector
    Step 2: error = y_hat - y                     <- (y_hat - y)
    Step 3: dJ/db = (2/n) * error                 <- since dy_hat_i/db_i = 1,
                                                      this IS dJ/db directly
                                                      (elementwise, no sum)
    Step 4: dJ/dm = (2/n) * X.T @ error            <- chain rule: dy_hat/dm = X
    """
    n = X.shape[0]

    # Step 1
    y_hat = predict(m, b, X)

    # Step 2
    error = y_hat - y

    # Step 3: gradient w.r.t. b (per-sample, elementwise -- matches team's math)
    dJ_db = (2.0 / n) * error

    # Step 4: gradient w.r.t. m (chain rule: dJ/dm = (2/n) * X^T . error)
    dJ_dm = (2.0 / n) * (X.T @ error)

    return dJ_dm, dJ_db


# ---------------------------------------------------------------------------
# 6. GRADIENT DESCENT LOOP (every step visible -- not abstracted away)
# ---------------------------------------------------------------------------
def run_gradient_descent(m_init=INITIAL_M, b_init=INITIAL_B,
                          learning_rate=0.01, n_iterations=50, verbose=True):
    """
    Returns a history dict with m, b, mse, and both gradients logged
    at every iteration, so this can feed straight into plots.py.
    """
    m = m_init.copy()
    b = b_init.copy()

    history = {
        "iteration": [],
        "m1": [], "m2": [], "b1": [], "b2": [],
        "mse": [],
        "manual_grad_m": [], "manual_grad_b": [],
        "scipy_grad_m": [], "scipy_grad_b": [],
    }

    for i in range(n_iterations):
        params = np.concatenate([m, b])   # [m1, m2, b1, b2]

        # --- current cost ---
        current_mse = mse_cost(params)

        # --- both gradients, computed independently ---
        manual_dJ_dm, manual_dJ_db = manual_gradient(m, b)
        scipy_dJ_dm, scipy_dJ_db = scipy_gradient(params)

        # --- log BEFORE updating (so iteration 0 = initial state) ---
        history["iteration"].append(i)
        history["m1"].append(m[0])
        history["m2"].append(m[1])
        history["b1"].append(b[0])
        history["b2"].append(b[1])
        history["mse"].append(current_mse)
        history["manual_grad_m"].append(manual_dJ_dm.copy())
        history["manual_grad_b"].append(manual_dJ_db.copy())
        history["scipy_grad_m"].append(scipy_dJ_dm.copy())
        history["scipy_grad_b"].append(scipy_dJ_db.copy())

        if verbose:
            print(f"Iter {i:2d} | m=[{m[0]:.4f}, {m[1]:.4f}] | "
                  f"b=[{b[0]:.4f}, {b[1]:.4f}] | MSE={current_mse:.4f} | "
                  f"manual_grad_m=[{manual_dJ_dm[0]:.4f}, {manual_dJ_dm[1]:.4f}] "
                  f"scipy_grad_m=[{scipy_dJ_dm[0]:.4f}, {scipy_dJ_dm[1]:.4f}] "
                  f"| manual_grad_b=[{manual_dJ_db[0]:.4f}, {manual_dJ_db[1]:.4f}] "
                  f"scipy_grad_b=[{scipy_dJ_db[0]:.4f}, {scipy_dJ_db[1]:.4f}]")

        # --- explicit update step (visible, not hidden in a helper) ---
        m = m - learning_rate * manual_dJ_dm
        b = b - learning_rate * manual_dJ_db

    return history, m, b


if __name__ == "__main__":
    print("=" * 100)
    print("PART 4: Gradient Descent in Code (SciPy-verified, chain-rule driven)")
    print("Starting from Kolade's Iteration 2 handoff -> computing Iteration 3 onward")
    print("=" * 100)

    # --- Sanity check: does our Iteration 3 step match a hand calc, if you do one? ---
    print("\n--- Crispin's Iteration 3 (first step from handoff) ---")
    dJ_dm_3, dJ_db_3 = manual_gradient(INITIAL_M, INITIAL_B)
    m3 = INITIAL_M - ALPHA * dJ_dm_3
    b3 = INITIAL_B - ALPHA * dJ_db_3
    print(f"grad_m = {np.round(dJ_dm_3, 4)}")
    print(f"grad_b = {np.round(dJ_db_3, 4)}")
    print(f"m3 (handoff to AbdalazizI) = {np.round(m3, 4)}")
    print(f"b3 (handoff to AbdalazizI) = {np.round(b3, 4)}")

    print("\n--- Full run for plots (50 iterations from Iteration 3 onward) ---")
    history, final_m, final_b = run_gradient_descent(learning_rate=ALPHA, n_iterations=50)
    print("-" * 100)
    print(f"Final m: {np.round(final_m, 4)}, Final b: {np.round(final_b, 4)}")
    print(f"Final predictions: {np.round(predict(final_m, final_b), 4)}")
    print(f"Target y:           {y}")