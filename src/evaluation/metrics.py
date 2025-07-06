import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import jaccard_score

# Sample evaluation functions

def compute_actionability(x, cf, actionable_features, actionability_threshold=0.8):
    """
    Compute actionability as % of changes in actionable features.
    Return True if ratio exceeds threshold, else False.
    """
    # Check if cf is empty or has mismatched shape
    if cf.empty or cf.shape[1] != x.shape[1]:
        return False  # Not actionable if counterfactual is invalid

    changes = (x != cf).values[0]
    total_changes = np.sum(changes)

    actionable_changes = np.sum([
        changes[x.columns.get_loc(f)] for f in actionable_features if f in x.columns
    ])

    if total_changes == 0:
        return False

    return (actionable_changes / total_changes) >= actionability_threshold


def compute_validity(model, cf, desired_output):
    """
    Check if model predicts the desired output for the counterfactual.
    """
    return model.predict(cf)[0] == desired_output


def compute_plausibility(cf, X_train, n_neighbors=20):
    """
    Use LOF to return True if cf is plausible (i.e., not an outlier).
    """
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
    lof.fit(X_train)

    label = lof.predict(cf)[0]
    return label == 1
def compute_sparsity(x, cf):
    """
    Compute sparsity as % of features changed.
    """
    changes = (x != cf).values[0]
    return np.mean(changes)

def compute_proximity(x, cf, num_cols, cat_cols):
    """
    Compute proximity using Euclidean distance for numerical features
    and Jaccard Index for categorical features.
    """
    x_num = x[num_cols].values[0]
    cf_num = cf[num_cols].values[0]
    euclidean = np.linalg.norm(x_num - cf_num)

    x_cat = x[cat_cols].astype(bool).astype(int).values[0]
    cf_cat = cf[cat_cols].astype(bool).astype(int).values[0]
    jaccard = jaccard_score(x_cat, cf_cat)

    return euclidean, jaccard


def compute_feasibility(x, cf, actionable_features, model, desired_output, X_train,actionability_threshold):
    """
    Computes the Feasibility of a counterfactual example.

    Parameters:
    - x: original instance (as a DataFrame row or Series)
    - cf: counterfactual instance (same format as x)
    - actionable_features: list of feature names allowed to change
    - model: the predictive model (with .predict())
    - desired_output: expected flipped class for cf
    - X_train: training data for plausibility check

    Returns:
    - feasibility_score: sum of actionability, plausibility, and validity
    """
    actionability = compute_actionability(x, cf, actionable_features,actionability_threshold)  # Between 0 and 1
    plausibility = compute_plausibility(cf, X_train)  # Higher is better
    validity = compute_validity(model, cf, desired_output)  # 0 or 1

    # Normalize plausibility score to be between 0 and 1
    # (if not already), for example using inverse LOF:

    feasibility = actionability and plausibility and validity
    return feasibility
