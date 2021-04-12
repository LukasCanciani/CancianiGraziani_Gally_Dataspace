#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.feature_selection import chi2, SelectKBest
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv("wine.csv", delimiter=";")


# In[ ]:


display(df)
print(f"the column where at least a column is null are: {sum(df.isnull().sum())}")
print(f"the shape of the dataset is the following: {df.shape}")
count = df.isnull().sum()
print("\nMissing values per column")
print(count.to_string())


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(40,40), dpi=120)
for ind, feature in enumerate(df.keys()[1:]):
    row = ind/4
    column = ind%4
    fig = sns.violinplot(data=df, y=feature,
        ax=axs[int(row)][column],
    x='class').set_title(feature)
    axs[int(row)][column].set_xlabel("")
    axs[int(row)][column].set_ylabel("")
plt.show()


# In[ ]:


sns.jointplot(data=df, x="class", y="flavanoids", color="blue", kind="kde")
sns.jointplot(data=df, x="class", y="ash", color="red", kind="kde")


# In[ ]:


corrmat = df.corr()
top_corr_feat = corrmat.index
plt.figure(figsize=(20,20))
g= sns.heatmap(df[top_corr_feat].corr(), annot=True,cmap="RdYlGn")


# In[ ]:


X = df.loc[:, df.columns != "class"]
y = df["class"]
bestfeatures = SelectKBest(score_func=chi2, k=4)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcol= pd.DataFrame(X.columns)
featureScores = pd.concat([dfcol,dfscores],axis=1)
featureScores.columns = ['Feature','Chi2-Score']
print(featureScores.nlargest(13,'Chi2-Score'))


# In[ ]:


model = RandomForestClassifier()
model.fit(X,y)
print(model.feature_importances_)
feat_imp=pd.Series(model.feature_importances_, index=X.columns)
feat_imp.nlargest(13).plot(kind='barh')
plt.show()


# In[ ]:




