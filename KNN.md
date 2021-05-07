```python
import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from imblearn.over_sampling import SMOTE
```


```python
df = pd.read_csv("wine.csv", delimiter=";")
#display(df)
del df["ash"]
#Norm

numeric = ["alcohol", "malic acid","alcalinity","magnesium","total phenols","flavanoids","nonflavanoid phenols","proanthocyanins", "color intensity", "hue", "od280/od315", "proline"]
for col in numeric:
    min = df[col].min()
    max = df[col].max()
    df[col] = (2*(df[col] - min) / (max - min)) -1
    
#display(df)

X = df.loc[:, df.columns != "class"]
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
```


```python
n_neighbors = [3,5,11,19]
weights = ['uniform','distance']
metric = ['euclidean','manhattan']
    
scores_smote = pd.DataFrame({'n': [], 'w': [], 'm': [], 'accuracy': []})
for n in n_neighbors:
    for w in weights:
        for m in metric:
            acc = []
            clf = KNeighborsClassifier(n_neighbors = n, weights = w, metric = m)
            kf = StratifiedKFold(n_splits=5)
            for fold, (train_index, val_index) in enumerate(kf.split(X_train, y_train), 1):  
                y_train_fold = y_train.iloc[train_index]  
                X_train_fold = X_train.iloc[train_index]
                X_val_fold = X_train.iloc[val_index]
                y_val_fold = y_train.iloc[val_index]  
    
                sm = SMOTE()
                X_train_fold_oversampled, y_train_fold_oversampled = sm.fit_resample(X_train_fold, y_train_fold)
  
                clf.fit(X_train_fold_oversampled, y_train_fold_oversampled )  
                y_pred_fold = clf.predict(X_val_fold)
                acc.append(accuracy_score(y_val_fold, y_pred_fold))
            scores_smote = scores_smote.append(pd.Series({'n': n, 'w': w, 'm' : m, 'accuracy': np.mean(acc)}), ignore_index = True)
best_config = scores_smote.iloc[scores_smote['accuracy'].idxmax()]
print(f"Best configuration WITH SMOTE:\n{best_config}")
best_n = best_config['n']
best_w = best_config['w']
best_m = best_config['m']
```

    Best configuration WITH SMOTE:
    n                 3.0
    w            distance
    m           euclidean
    accuracy     0.959667
    Name: 2, dtype: object
    


```python
best_clf = KNeighborsClassifier(n_neighbors = int(best_n), weights = best_w, metric = best_m)

sm = SMOTE()
X_train_oversampled, y_train_oversampled = sm.fit_resample(X_train, y_train)
best_clf.fit(X_train_oversampled, y_train_oversampled)

y_pred = best_clf.predict(X_test)
print("KNN EVALUATION WITH SMOTE")
print("Accuracy %f"%accuracy_score(y_test, y_pred))
print("Precision %f"%precision_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
print("Recall %f"%recall_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
print("F1-Score %f"%f1_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
```

    KNN EVALUATION WITH SMOTE
    Accuracy 0.981481
    Precision 0.982253
    Recall 0.981481
    F1-Score 0.981359
    


```python
n_neighbors = [3,5,11,19]
weights = ['uniform','distance']
metric = ['euclidean','manhattan']
    
scores = pd.DataFrame({'n': [], 'w': [], 'm': []})
for n in n_neighbors:
    for w in weights:
        for m in metric:
            acc = []
            clf = KNeighborsClassifier(n_neighbors = n, weights = w, metric = m)
            kf = StratifiedKFold(n_splits=5)
            for fold, (train_index, val_index) in enumerate(kf.split(X_train, y_train), 1):  
                y_train_fold = y_train.iloc[train_index]  
                X_train_fold = X_train.iloc[train_index]
                X_val_fold = X_train.iloc[val_index]
                y_val_fold = y_train.iloc[val_index]  
    
                clf.fit(X_train_fold, y_train_fold )  
                y_pred_fold = clf.predict(X_val_fold)
                acc.append(accuracy_score(y_val_fold, y_pred_fold))
            scores = scores.append(pd.Series({'n': n, 'w': w, 'm' : m, 'accuracy': np.mean(acc)}), ignore_index = True)
best_config = scores.iloc[scores['accuracy'].idxmax()]
print(f"Best configuration WITHOUT SMOTE:\n{best_config}")
best_n = best_config['n']
best_w = best_config['w']
best_m = best_config['m']
```

    Best configuration WITHOUT SMOTE:
    n                11.0
    w             uniform
    m           manhattan
    accuracy         0.96
    Name: 9, dtype: object
    


```python
best_clf = KNeighborsClassifier(n_neighbors = int(best_n), weights = best_w, metric = best_m)
best_clf.fit(X_train, y_train)
y_pred = best_clf.predict(X_test)
print("KNN EVALUATION WITHOUT SMOTE")
print("Accuracy %f"%accuracy_score(y_test, y_pred))
print("Precision %f"%precision_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
print("Recall %f"%recall_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
print("F1-Score %f"%f1_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
```

    KNN EVALUATION WITHOUT SMOTE
    Accuracy 0.981481
    Precision 0.982571
    Recall 0.981481
    F1-Score 0.981443
    
