```python
import numpy as np
import pandas as pd

from sklearn.svm import SVC
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
c = [0.001,0.01,0.1,1,10]
gamma = [0.001,0.01,0.1,1,10]

scores_smote = pd.DataFrame({'C': [], 'gamma': [], 'accuracy': []})
for C in c:
    for g in gamma:
        acc = []
        clf = SVC(kernel='rbf', C = C, gamma = g)
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
        scores_smote = scores_smote.append(pd.Series({'C': C, 'gamma': g, 'accuracy': np.mean(acc)}), ignore_index = True)
best_config = scores_smote.iloc[scores_smote['accuracy'].idxmax()]
print(f"Best configuration WITH SMOTE:\n{best_config}")
best_c = best_config['C']
best_gamma = best_config['gamma']
    
```

    Best configuration WITH SMOTE:
    C           1.000000
    gamma       0.100000
    accuracy    0.975333
    Name: 17, dtype: float64
    


```python
best_clf = SVC(kernel='rbf', C=  best_config['C'], gamma = best_config['gamma'])

sm = SMOTE()
X_train_oversampled, y_train_oversampled = sm.fit_resample(X_train, y_train)
best_clf.fit(X_train_oversampled, y_train_oversampled)

y_pred = best_clf.predict(X_test)
print("SVM EVALUATION WITH SMOTE")
print("Accuracy %f"%accuracy_score(y_test, y_pred))
print("Precision %f"%precision_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
print("Recall %f"%recall_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
print("F1-Score %f"%f1_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
```

    SVM EVALUATION WITH SMOTE
    Accuracy 0.962963
    Precision 0.965168
    Recall 0.962963
    F1-Score 0.962606
    


```python
c = [0.001,0.01,0.1,1,10]
gamma = [0.001,0.01,0.1,1,10]

scores = pd.DataFrame({'C': [], 'gamma': [], 'accuracy': []})
for C in c:
    for g in gamma:
        acc = []
        clf = SVC(kernel='rbf', C = C, gamma = g)
        kf = StratifiedKFold(n_splits=5)
        for fold, (train_index, val_index) in enumerate(kf.split(X_train, y_train), 1):  
            y_train_fold = y_train.iloc[train_index]  
            X_train_fold = X_train.iloc[train_index]
            X_val_fold = X_train.iloc[val_index]
            y_val_fold = y_train.iloc[val_index]  
  
            clf.fit(X_train_fold, y_train_fold )  
            y_pred_fold = clf.predict(X_val_fold)
            acc.append(accuracy_score(y_val_fold, y_pred_fold))
        scores = scores.append(pd.Series({'C': C, 'gamma': g, 'accuracy': np.mean(acc)}), ignore_index = True)
best_config = scores.iloc[scores['accuracy'].idxmax()]
print(f"Best configuration WITHOUT SMOTE:\n{best_config}")
best_c = best_config['C']
best_gamma = best_config['gamma']
```

    Best configuration WITHOUT SMOTE:
    C           1.000000
    gamma       0.100000
    accuracy    0.975333
    Name: 17, dtype: float64
    


```python
best_clf = SVC(kernel='rbf', C=  best_config['C'], gamma = best_config['gamma'])
best_clf.fit(X_train, y_train)
y_pred = best_clf.predict(X_test)
print("SVM EVALUATION WITHOUT SMOTE")
print("Accuracy %f"%accuracy_score(y_test, y_pred))
print("Precision %f"%precision_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
print("Recall %f"%recall_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
print("F1-Score %f"%f1_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
```

    SVM EVALUATION WITHOUT SMOTE
    Accuracy 0.981481
    Precision 0.982804
    Recall 0.981481
    F1-Score 0.981599
    
