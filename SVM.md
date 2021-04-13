```python
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score

from imblearn.over_sampling import SMOTE
```


```python
df = pd.read_csv("wine.csv", delimiter=";")
display(df)
del df["ash"]
#Norm

numeric = ["alcohol", "malic acid","alcalinity","magnesium","total phenols","flavanoids","nonflavanoid phenols","proanthocyanins", "color intensity", "hue", "od280/od315", "proline"]
for col in numeric:
    min = df[col].min()
    max = df[col].max()
    df[col] = (2*(df[col] - min) / (max - min)) -1
    
display(df)
```


```python
X = df.loc[:, df.columns != "class"]
y = df["class"]

param_grid = {'C': [0.001,0.01,0.1,1,10], 'gamma' : [0.001,0.01,0.1,1,10]}

gs = GridSearchCV(SVC(kernel='rbf'), param_grid)

gs.fit(X,y)
model = gs.best_estimator_
print(model.get_params())
```


```python
kf = StratifiedKFold(n_splits=5)
acc = []
prec =[]
rec = []
f1 = []
for fold, (train_index, test_index) in enumerate(kf.split(X, y), 1):  
    y_train = y[train_index]  # Based on your code, you might need a ravel call here, but I would look into how you're generating your y
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    y_test = y[test_index]  # See comment on ravel and  y_train
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
    y_train = y[train_index]  # Based on your code, you might need a ravel call here, but I would look into how you're generating your y
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    y_test = y[test_index]  # See comment on ravel and  y_train
    
    sm = SMOTE()
    X_train_oversampled, y_train_oversampled = sm.fit_resample(X_train, y_train)
  
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
