from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

def test_classifier_prediction(clf,X_test,y_test):
    y_pred = clf.predict(X_test)
    return y_pred,classification_report(y_test, y_pred),accuracy_score(y_test, y_pred)