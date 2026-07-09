# Formative 3: Probability

This repository contains code and notebooks covering various probability and machine learning concepts. It is divided into four main parts:

- **Part 1:** EM Clustering (`part1_em_clustering/`)
- **Part 2:** Bayesian IMDB (`part2_bayesian_imdb/`)
- **Part 3:** Manual Gradient Descent (`part3_manual_gradient_descent/`)
- **Part 4:** SciPy Gradient Descent (`part4_scipy_gd/`)

## Team Contributions

As outlined in our Formative 3 Execution Plan, the responsibilities were divided as follows:

| Team Member | Assigned Lead Role | Core Responsibilities & Deliverables |
| --- | --- | --- |
| **Arnold Mutara** | Lead Mathematical Architect (Part 3 Lead & Part 1 Math) | Derive explicit Chain Rule matrix equations for MSE loss. Anchor the 4-iteration manual math relay race. Formulate the mathematical justification for why global-mean splitting fails in Gaussian mixtures. |
| **AbdalazizI Twariki** | Core Algorithm Lead (Part 1 Code & Repo Manager) | Implement pure Python EM algorithm from scratch. Build live interactive posterior prediction script for coach presentation. Maintain modular, DRY repository architecture. |
| **Kolade Oluwatunmise Adepoju** | Pure Python & Bayesian Lead (Part 2 Lead) | Write pure Python text parsing engine for IMDb reviews (strictly no external ML libs). Compute prior, likelihood, marginal, and posterior tables formatted in clear Markdown blocks. |
| **Crispin Hebert Hirwa** | Optimization & Deck Lead (Part 4 Lead & Deck Master) | Implement SciPy derivatives and programmatic gradient descent. Generate Matplotlib parameter trajectory and loss curves. Assemble cohesive 15-minute presentation slide deck. |

## Setup Instructions

Follow these steps to set up the project locally.

### Prerequisites
- Python 3.8 or newer
- `pip` (Python package installer)

### 1. Clone the Repository
```bash
git clone https://github.com/TWARIQABDUL/formative_3_probability.git
cd formative_3_probability
```

### 2. Create a Virtual Environment (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Install all required packages listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

## Running the Code and Expected Results

### Part 1: EM Clustering
**How to run:**
```bash
python -m part1_em_clustering
```
**Expected Results:**
- Downloads the Galton dataset via `kagglehub`.
- Runs the Expectation-Maximization loop to cluster heights into two Gaussian distributions.
- Outputs a formatted tracking table logging exact values for means (μ1, μ2), variances (σ1², σ2²), mixing coefficients (π1, π2), and Log-Likelihood at each iteration.
- Performs a live posterior classification of a random test height (e.g., 68.5 inches), printing the probability of belonging to each cluster (Child vs. Basketball Player).

### Part 2: Bayesian Sentiment Analysis on IMDB
**How to run:**
```bash
python part2_bayesian_imdb/bayes_engine.py
```
**Expected Results:**
- Downloads and loads the IMDB dataset.
- Evaluates chosen positive and negative keywords without using external ML libraries (pure Python).
- Outputs Markdown-formatted tables displaying the Prior, Likelihood, Marginal, and Posterior probabilities of positive sentiment for each keyword.

### Part 3: Manual Gradient Descent (Verification)
**How to run:**
```bash
python part3_manual_gradient_descent/verify_iteration1.py
python part3_manual_gradient_descent/verify_iteration2.py
```
**Expected Results:**
- Executes step-by-step matrix multiplication verifying the team's manual gradient descent relay race hand-calculations (MSE cost, gradients w.r.t m and b).
- Prints checks against the hand-calculated answers and outputs `"RESULT: All hand calculations verified correct!"`.

### Part 4: SciPy Gradient Descent
**How to run:**
```bash
python part4_scipy_gd/gradient_descent.py
```
**Expected Results:**
- Runs programmatic gradient descent starting from Kolade's Iteration 2 handoff.
- Outputs the trajectory of parameters and compares the explicit chain-rule derivation against SciPy's `approx_fprime` generic numerical gradient to prove mathematical correctness.
- *(Optional)* Running `python part4_scipy_gd/plots.py` will generate the `error_trajectory.png` and `parameter_trajectory.png` plots used in the presentation deck.
