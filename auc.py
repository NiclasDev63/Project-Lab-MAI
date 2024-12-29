import numpy as np
from sklearn.metrics import roc_auc_score

# Step 1: Load scores from files
fake_scores = np.loadtxt("FakeScores.txt")  # Load fake scores
real_scores = np.loadtxt("Realscores.txt")  # Load real scores

# Step 2: Assign labels
fake_labels = np.zeros(len(fake_scores))  # Label all fake scores as 0
real_labels = np.ones(len(real_scores))  # Label all real scores as 1

# Step 3: Combine scores and labels
all_scores = np.concatenate([fake_scores, real_scores])  # Combine scores
all_labels = np.concatenate([fake_labels, real_labels])  # Combine labels

# Step 4: Calculate AUC
auc = roc_auc_score(all_labels, all_scores)

# Print the result
print(f"AUC: {auc:.4f}")
