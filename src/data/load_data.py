from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_input_data(test_size=0.3, random_state=42):
    """
    Loads the Statlog (German Credit Data) dataset from UCI ML repo,
    splits it into training and testing sets, and returns all components.

    Returns:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Testing features
        y_train (pd.Series): Training labels
        y_test (pd.Series): Testing labels
        metadata (dict): Dataset metadata
        variables (dict): Feature information
    """
    # Fetch dataset
    data = fetch_ucirepo(id=144)

    # Features and targets
    X = data.data.features
    y = data.data.targets

    # Combine into full DataFrame
    dataset_df = pd.concat([X, y], axis=1)
    X = dataset_df.loc[:, dataset_df.columns != 'class']
    y = dataset_df['class']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, data.metadata, data.variables, dataset_df

def preprocess_data(X_train, X_test, datasetdf, numeric_columns):
    """
    Applies preprocessing to train and test features:
    - One-hot encodes categorical variables
    - Aligns test set columns to match train set
    - Standardizes specified numeric columns

    Args:
        X_train (pd.DataFrame): Raw training features
        X_test (pd.DataFrame): Raw test features
        numeric_columns (List[str]): List of numeric columns to standardize

    Returns:
        pd.DataFrame: Processed X_train
        pd.DataFrame: Processed X_test
    """
    # One-hot encode categorical features
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_test = pd.get_dummies(X_test, drop_first=True)
    datasetdf = pd.get_dummies(datasetdf, drop_first=True)

    # Align test columns to train columns
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # Standardize numeric columns
    scaler = StandardScaler()
    X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns]) #fit on trainingdata
    X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])
    datasetdf[numeric_columns] = scaler.transform(datasetdf[numeric_columns])

    return X_train, X_test, datasetdf


def preprocess_only_numerical_features(X_train, X_test,datasetdf, numeric_columns):
    scaler = StandardScaler()
    X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
    X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])
    datasetdf[numeric_columns] = scaler.transform(datasetdf[numeric_columns])

    return X_train, X_test, datasetdf

def preprocess_bank(df):
    """
    Preprocessing identical to how the authors do it, for a fair comparison
    """
    """
    :param bankloan: bank dataframe
    :return:
    """
    features = ['Income', 'Family', 'CCAvg', 'Education', 'Mortgage',
                'SecuritiesAccount', 'CDAccount', 'Online', 'CreditCard']
    catf = ['SecuritiesAccount', 'CDAccount', 'Online', 'CreditCard']
    numf = ['Income', 'Family', 'CCAvg', 'Education', 'Mortgage']
    p = {'Income': 40, 'CCAvg': 1.5, 'Family': 3, 'Education': 2, 'Mortgage': 80, 'CDAccount': 1, 'Online': 1,
          'SecuritiesAccount': 1, 'CreditCard': 1} #uf is p
    step = {'Income': 1, 'CCAvg': 0.1, 'Family': 1, 'Education': 1, 'Mortgage': 1, 'CDAccount': 1, 'Online': 1,
         'SecuritiesAccount': 1, 'CreditCard': 1}
    # uf  = getMCSvalues()
    f2change = ['Income', 'CCAvg', 'Mortgage','CDAccount', 'Online']
    outcome_label = 'Personal Loan'
    desired_outcome = 1.0
    nbr_features = 9
    protectf = []

    # desired space
    data_lab1 = pd.DataFrame()
    data_lab1 = df[df["Personal Loan"] == 1]
    data_lab0 = df[df["Personal Loan"] == 0]
    data_lab1 = data_lab1.drop(['Personal Loan'], axis=1)
    return features, catf, numf, uf, f2change, outcome_label, desired_outcome, nbr_features, protectf, data_lab0, data_lab1