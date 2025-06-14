from sklearn.ensemble import RandomForestClassifier

def train_classifier(X_train,y_train,n_estimators=100, random_state=42):
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(X_train, y_train)
    return clf