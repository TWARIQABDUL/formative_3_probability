import os
import csv
import kagglehub
from em_algorithm import GaussianMixture1D

path = kagglehub.dataset_download("jacopoferretti/parents-heights-vs-children-heights-galton-data")

csv_file = None
for file in os.listdir(path):
    if file.endswith(".csv"):
        csv_file = os.path.join(path, file)
        break

if not csv_file:
    raise FileNotFoundError("Could not find a CSV file in the downloaded dataset directory!")

print(f"Loading data from: {csv_file}")

# 2. Extract heights (comparing Fathers and Children as unlabelled data)
unlabelled_heights = []

with open(csv_file, mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            # Note: Galton dataset heights are usually recorded in inches.
            # Let's check common column names (e.g., 'Father', 'Height' or 'father', 'childHeight')
            # Adjust column names below if needed after checking your CSV header!
            father_h = float(row.get('Father', row.get('father', 0)))
            child_h = float(row.get('Height', row.get('childHeight', row.get('height', 0))))
            
            if father_h > 0:
                unlabelled_heights.append(father_h)
            if child_h > 0:
                unlabelled_heights.append(child_h)
        except ValueError:
            continue

print(f"Total unlabelled height observations loaded: {len(unlabelled_heights)}")

# 3. Initialize and run the Expectation-Maximization algorithm
# In inches: Children mean ~65.0, Father mean ~69.0 (If in cm, use ~165.0 and ~175.0)
avg_height = sum(unlabelled_heights) / len(unlabelled_heights) if unlabelled_heights else 67.0

# Let's set initial guesses slightly below and slightly above the dataset mean
gmm = GaussianMixture1D(
    data=unlabelled_heights,
    mu1=avg_height - 3.0,  # Cluster 1: Initial guess for Children
    mu2=avg_height + 3.0,  # Cluster 2: Initial guess for Fathers/Pros
    var1=10.0,             # Initial variance guess
    var2=10.0,
    pi1=0.5,
    pi2=0.5
)

# Run EM and print the tracking table (Iteration 0, 1, 2...)
gmm.fit(max_iterations=25)

# 4. Test the Live Coach Prediction function with a random test height (e.g., 68.5 inches)
gmm.predict_posterior(test_height=68.5)