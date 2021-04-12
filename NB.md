```python
import numpy as np
import pandas as pd

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score

from imblearn.over_sampling import SMOTE
```


```python
df = pd.read_csv("wine.csv", delimiter=";")
del df["ash"]
X = df.loc[:, df.columns != "class"]
y = df["class"]
```


```python
kf = StratifiedKFold(n_splits=5)
acc = []
prec =[]
rec = []
f1 = []
for fold, (train_index, test_index) in enumerate(kf.split(X, y), 1):  
    y_train = y[train_index]  
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    y_test = y[test_index]  
    model = GaussianNB()
    model.fit(X_train, y_train )  
    y_pred = model.predict(X_test)
   
    acc.append(model.score(X_test, y_test))
    prec.append(precision_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
    rec.append(recall_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
    f1.append(f1_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
    
print("Average scores for 5 fold CV - Without SMOTE")
print("Accuracy: " + str(np.mean(acc).round(3)))
print("Weighted Precision: " + str(np.mean(prec).round(3)))
print("Weighted Recall:    " + str(np.mean(rec).round(3)))
print("Weighted F1-Score:  " + str(np.mean(f1).round(3)))
```


```python
kf = StratifiedKFold(n_splits=5)
acc = []
prec =[]
rec = []
f1 = []
for fold, (train_index, test_index) in enumerate(kf.split(X, y), 1):  
    y_train = y[train_index]  
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    y_test = y[test_index]  
    
    sm = SMOTE()
    X_train_oversampled, y_train_oversampled = sm.fit_resample(X_train, y_train)
  
    model = GaussianNB()
    model.fit(X_train_oversampled, y_train_oversampled )  
    y_pred = model.predict(X_test)
   
    acc.append(model.score(X_test, y_test))
    prec.append(precision_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
    rec.append(recall_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
    f1.append(f1_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
    
print("Average scores for 5 fold CV - With SMOTE")
print("Accuracy: " + str(np.mean(acc).round(3)))
print("Weighted Precision: " + str(np.mean(prec).round(3)))
print("Weighted Recall:    " + str(np.mean(rec).round(3)))
print("Weighted F1-Score:  " + str(np.mean(f1).round(3)))
```


```python

```


```python

```
