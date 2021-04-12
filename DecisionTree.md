```python
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import  StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score

from imblearn.over_sampling import SMOTE

#print tree
from sklearn.tree import export_graphviz
from six import StringIO  
from IPython.display import Image  
import pydotplus
```


```python
df = pd.read_csv("wine.csv", delimiter=";")
del df['ash']
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
classes = np.unique(y)
for c in classes: 
    total = len(y[y==c])
    ratio = (total / float(len(y))) * 100
    print("Class %s : %d (%.3f%%)" % (str(c), total, ratio))
```


```python
grid_params = {
    'criterion':('gini','entropy'), 
'min_samples_split':[2,3,4,5,6,7,8],
'max_depth':[1,2,3,4,5,6,10,15,20], 
'class_weight':('balanced', None)
}

gs = GridSearchCV(
    DecisionTreeClassifier(),
    grid_params,
)

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

dot_data=StringIO()
export_graphviz(model, out_file=dot_data, filled=True, rounded=True, special_characters = True, feature_names =X_train.columns.values.tolist())

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png("wineDTC.png")
Image(graph.create_png())
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

```
