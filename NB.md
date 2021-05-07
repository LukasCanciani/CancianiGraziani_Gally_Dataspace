```python
import numpy as np
import pandas as pd

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from imblearn.over_sampling import SMOTE
```


```python
df = pd.read_csv("wine.csv", delimiter=";")
del df["ash"]
X = df.loc[:, df.columns != "class"]
y = df["class"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
```


```python
best_clf = GaussianNB()

sm = SMOTE()
X_train_oversampled, y_train_oversampled = sm.fit_resample(X_train, y_train)
best_clf.fit(X_train_oversampled, y_train_oversampled)

y_pred = best_clf.predict(X_test)
print("NB EVALUATION WITH SMOTE")
print("Accuracy %f"%accuracy_score(y_test, y_pred))
print("Precision %f"%precision_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
print("Recall %f"%recall_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
print("F1-Score %f"%f1_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
```

    NB EVALUATION WITH SMOTE
    Accuracy 0.962963
    Precision 0.965608
    Recall 0.962963
    F1-Score 0.962963
    


```python
best_clf = GaussianNB()

best_clf.fit(X_train, y_train)
y_pred = best_clf.predict(X_test)
print("NB EVALUATION WITHOUT SMOTE")
print("Accuracy %f"%accuracy_score(y_test, y_pred))
print("Precision %f"%precision_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
print("Recall %f"%recall_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
print("F1-Score %f"%f1_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
```

    NB EVALUATION WITHOUT SMOTE
    Accuracy 0.962963
    Precision 0.965608
    Recall 0.962963
    F1-Score 0.962963
    
