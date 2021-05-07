```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
```


```python
data = pd.read_csv("wine.csv", delimiter=";")
del data["ash"]
X = data.iloc[:, data.columns != "class"]
y = data["class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
```


```python
C = [1,2,3,4,5,6,7,8,9,10]
solver = ['newton-cg','liblinear','saga','sag']

scores_smote = pd.DataFrame({'c': [], 's': [], 'accuracy': []})

for c in C:
    for s in solver:
        acc = []
        clf = LogisticRegression(C = c, solver = s)
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
        scores_smote = scores_smote.append(pd.Series({'c': c, 's': s, 'accuracy': np.mean(acc)}), ignore_index = True)

best_config = scores_smote.iloc[scores_smote['accuracy'].idxmax()]
print(f"Best configuration WITH SMOTE:\n{best_config}")
best_c = best_config['c']
best_s = best_config['s']
```

   ...
    

    Best configuration WITH SMOTE:
    c                 1.0
    s           newton-cg
    accuracy     0.967667
    Name: 0, dtype: object
    

    ...


```python
best_clf = LogisticRegression(C = int(best_c), solver = best_s)

sm = SMOTE()
X_train_oversampled, y_train_oversampled = sm.fit_resample(X_train, y_train)
best_clf.fit(X_train_oversampled, y_train_oversampled)

y_pred = best_clf.predict(X_test)
print("LOGISTIC REGRESSION EVALUATION WITH SMOTE")
print("Accuracy %f"%accuracy_score(y_test, y_pred))
print("Precision %f"%precision_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
print("Recall %f"%recall_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
print("F1-Score %f"%f1_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
```

    LOGISTIC REGRESSION EVALUATION WITH SMOTE
    Accuracy 0.944444
    Precision 0.944505
    Recall 0.944444
    F1-Score 0.943974
    


```python
C = [1,2,3,4,5,6,7,8,9,10]
solver = ['newton-cg','liblinear','saga','sag']

scores = pd.DataFrame({'c': [], 's': [], 'accuracy': []})

for c in C:
    for s in solver:
        acc = []
        clf = LogisticRegression(C = c, solver = s)
        kf = StratifiedKFold(n_splits=5)
        for fold, (train_index, val_index) in enumerate(kf.split(X_train, y_train), 1):  
            y_train_fold = y_train.iloc[train_index]  
            X_train_fold = X_train.iloc[train_index]
            X_val_fold = X_train.iloc[val_index]
            y_val_fold = y_train.iloc[val_index]  
    
            clf.fit(X_train_fold, y_train_fold)  
            y_pred_fold = clf.predict(X_val_fold)
            acc.append(accuracy_score(y_val_fold, y_pred_fold))
        scores = scores.append(pd.Series({'c': c, 's': s, 'accuracy': np.mean(acc)}), ignore_index = True)

best_config = scores.iloc[scores['accuracy'].idxmax()]
print(f"Best configuration WITHOUT SMOTE:\n{best_config}")
best_c = best_config['c']
best_s = best_config['s']
```

   ...

    Best configuration WITHOUT SMOTE:
    c                 1.0
    s           newton-cg
    accuracy     0.975667
    Name: 0, dtype: object
    
    ...
    


```python
best_clf = LogisticRegression(C = int(best_c), solver = best_s)
best_clf.fit(X_train, y_train)

y_pred = best_clf.predict(X_test)
print("LOGISTIC REGRESSION EVALUATION WITHOUT SMOTE")
print("Accuracy %f"%accuracy_score(y_test, y_pred))
print("Precision %f"%precision_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
print("Recall %f"%recall_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
print("F1-Score %f"%f1_score(y_test, y_pred,average="weighted",labels=np.unique(y_pred)))
```

    LOGISTIC REGRESSION EVALUATION WITHOUT SMOTE
    Accuracy 0.925926
    Precision 0.925986
    Recall 0.925926
    F1-Score 0.925456
    
