import pandas as pd
from sklearn.inspection import permutation_importance
def summarize_features(X):
    summary = {}
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            summary[col] = {
                "mean": X[col].mean(),
                "min": X[col].min(),
                "max": X[col].max()
            }
        else:
            summary[col] = X[col].value_counts(normalize=True).to_dict()
    return summary


import pandas as pd


def extract_statistical_rules(X_pos, num_quantiles=(0.1, 0.9), cat_threshold=0.75):
    """
    Creates a compact rule dictionary for the positive class.
    Returns: dict like {'Attribute5': (10000, 30000), 'Attribute6': 'A11'}
    """
    rules = {}
    for col in X_pos.columns:
        col_data = X_pos[col]

        if pd.api.types.is_numeric_dtype(col_data):
            q_low = col_data.quantile(num_quantiles[0])
            q_high = col_data.quantile(num_quantiles[1])
            rules[col] = (round(q_low, 3), round(q_high, 3))

        else:
            val_counts = col_data.value_counts(normalize=True)
            top_val = val_counts.idxmax()
            freq = val_counts.max()
            if freq >= cat_threshold:
                rules[col] = top_val  # e.g., 'A11'

    return rules


def top_features(clf, X_train_proc, y_train, n_repeats=5, random_state=42):

    # Run permutation importance
    result = permutation_importance(clf, X_train_proc, y_train, n_repeats=5, random_state=42)

    # Create importance ranking
    perm_importances = pd.Series(result.importances_mean, index=X_train_proc.columns).sort_values(ascending=False)

    # Get top features based on threshold
    top_features = perm_importances[perm_importances > 0.01].index.tolist()

    return top_features


def combined_features(top_features,p_statistical):
    for feat in top_features:
        if '_' in feat and feat not in p_statistical:
            base, val = feat.split('_', 1)
            # Only add if base attr not already in p
            if base not in p_statistical:
                p_statistical[base] = val
    return p_statistical
