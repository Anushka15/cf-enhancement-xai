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


# def extract_statistical_rules(X_pos, X_other, num_quantiles=(0.1, 0.9), cat_threshold=0.85):
#     """
#     Creates a compact rule dictionary for the positive class.
#     Returns: dict like {'Attribute5': (10000, 30000), 'Attribute6': 'A11'}
#     """
#     rules = {}
#     for col in X_pos.columns:
#         col_data_pos = X_pos[col]
#
#         if pd.api.types.is_numeric_dtype(col_data_pos):
#             q_low = col_data_pos.quantile(num_quantiles[0])
#             q_high = col_data_pos.quantile(num_quantiles[1])
#             rules[col] = (round(q_low, 3), round(q_high, 3))
#
#         else:
#             # Frequency in positive class
#             pos_freq = col_data_pos.value_counts(normalize=True)
#
#             # Frequency in negative/other class
#             other_freq = X_other[col].value_counts(normalize=True)
#
#             # Sort by confidence (pos_freq descending)
#             for val, conf in pos_freq.items():
#                 if conf >= cat_threshold:
#                     other_conf = other_freq.get(val, 0)
#                     # Only add if it's more specific to the positive class
#                     if conf > other_conf:
#                         rules[col] = val
#                         break  # Only the top valid one
#     return rules

def extract_statistical_rules(X_pos, X_other, num_quantiles=(0.1, 0.9), cat_threshold=0.85):
    """
    Creates a compact rule dictionary for the positive class.
    Returns: dict like {'Attribute5': (10000, 30000), 'Attribute6': 'A11'}
    """
    rules = {}
    for col in X_pos.columns:
        col_data_pos = X_pos[col]
        col_data_other = X_other[col]

        if pd.api.types.is_numeric_dtype(col_data_pos):
            unique_vals = col_data_pos.dropna().unique()

            # If it's binary (0/1), treat as categorical
            if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1}):
                pos_freq = col_data_pos.value_counts(normalize=True)
                other_freq = col_data_other.value_counts(normalize=True)

                for val, conf in pos_freq.items():
                    if conf >= cat_threshold:
                        other_conf = other_freq.get(val, 0)
                        if conf > other_conf:
                            rules[col] = str(int(val))  # return as str for compatibility
                            break
            else:
                # Continuous numeric
                q_low = col_data_pos.quantile(num_quantiles[0])
                q_high = col_data_pos.quantile(num_quantiles[1])
                rules[col] = (round(q_low, 3), round(q_high, 3))
        else:
            # Categorical (non-numeric)
            pos_freq = col_data_pos.value_counts(normalize=True)
            other_freq = col_data_other.value_counts(normalize=True)

            for val, conf in pos_freq.items():
                if conf >= cat_threshold:
                    other_conf = other_freq.get(val, 0)
                    if conf > other_conf:
                        rules[col] = val
                        break
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
