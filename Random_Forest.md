```python
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score

# package for import the models
from sklearn.ensemble import RandomForestClassifier
import sys
import functools

from imblearn.over_sampling import SMOTE

#x Stampa albero
from sklearn.tree import export_graphviz
from six import StringIO  
from IPython.display import Image  
import pydotplus
```


```python
df = pd.read_csv("wine.csv", delimiter=";")
del(df['ash'])
display(df)
print(f"the column where at least a column is null are: {sum(df.isnull().sum())}")
print(f"the shape of the dataset is the following: {df.shape}")
count = df.isnull().sum()
print("\nMissing values per column")
print(count.to_string())
```


```python
X = df.loc[:, df.columns != "class"]
y = df["class"]

grid_params = {
    "n_estimators": [10, 20, 30, 50, 100],
    "criterion": ["gini","entropy","auto"],
    "max_features": ["sqrt", "auto"],
    "max_depth": [1,2,3,4,5,6],
    "bootstrap": [True, False],
    "oob_score": [True, False]
}

gs = GridSearchCV(
    RandomForestClassifier(),
    grid_params,
)

gs.fit(X,y)
model = gs.best_estimator_

#Analisi rateo x classe
classes = np.unique(y)
for c in classes: 
    total = len(y[y==c])
    ratio = (total / float(len(y))) * 100
    print("Class %s : %d (%.3f%%)" % (str(c), total, ratio))    
```


```python
print(model.get_params())
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
#plot of importance per attribute
x_val = [x+1 for x in range(model.feature_importances_.__len__())]
pair_val = []

for key,val in zip(x_val,model.feature_importances_):
    pair_val.append([key,val])

pair_val = sorted(pair_val, key=functools.cmp_to_key(lambda x,y: y[1]-x[1]))

y_val = [x[1] for x in pair_val]
x_val = [x[0] for x in pair_val]

plt.figure(figsize=(15,7), dpi=300)

names = df.columns
ax = sns.barplot(y=y_val,x=x_val)
ax.set_xticklabels(names[1:], rotation='vertical', fontsize=10)

plt.show()

#top 5 feature
print("Top 5 Feature")
for index in x_val[0:5]:
    print(names[index])
```
