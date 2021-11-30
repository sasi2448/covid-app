import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import davies_bouldin_score,roc_auc_score
from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# import seaborn as sns
from collections import Counter
from sklearn.naive_bayes import GaussianNB
import pickle

df = pd.read_csv('Cleaned-Data.csv')

def removeDuplicates(df):
    return df_clean.drop_duplicates()

def checkingForNulls(df):
    if df.isnull().sum().sum() == 0:
        return df
    else:
        return  df.dropna()   

# step 01
df_clean = checkingForNulls(df)

#step 02 
df_clean = df_clean.drop(['Country'],axis = 1)

df = df_clean.copy()

#scaling data
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
df_scaled=scaler.fit_transform(df)


bouldin_score=[]

for i in range(4,15):
    km=KMeans(n_clusters=i, random_state=2)
    labels=km.fit_predict(df_scaled)
    bouldin_score.append(davies_bouldin_score(df_scaled, labels))

km_model = KMeans(n_clusters=8, random_state=2)
km_model.fit(df_scaled)
labels = km_model.labels_
corona_df = pd.DataFrame(km_model.cluster_centers_, columns=df_clean.columns)

df_scaled_output = pd.concat([pd.DataFrame(df_clean, columns=df_clean.columns), pd.DataFrame({"Cluster": labels})],
                             axis=1)

df_scaled_output['Result'] = df_scaled_output['Cluster'
].replace({1: 'Positive', 0: 'Negative', 2: 'Negative', 3: 'Negative',

           4: 'Negative', 5: 'Negative', 6: 'Negative', 7: 'Positive'})

df_main = df_scaled_output.copy()
df_main_1 = df_main.copy()
df_main_1.drop(['Cluster'], axis=1, inplace=True)

X = df_main_1.drop(['Result'
                    #                    ,'Age_0-9','Age_10-19','Age_20-24','Age_25-59','Age_60+'
                    #                     ,'Gender_Female','Gender_Male','Gender_Transgender'

                    ], axis=1)
y = df_main_1['Result']
X.shape

# from imblearn.over_sampling import RandomOverSampler
#
# ros = RandomOverSampler(random_state=42)
#
# # fit predictor and target
# X_ros, y_ros = ros.fit_resample(X, y)
#
# print('Original dataset shape', Counter(y))
# print('Resample dataset shape', Counter(y_ros))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=31)

clf = GaussianNB()
# fit the classifier to the training set
clf.fit(X_train, y_train)

# plt.figure(num=None, figsize=(10,8), dpi=80, facecolor='w', edgecolor='k')
# feat_importances = pd.Series(clf.feature_importances_, index= X.columns)
# feat_importances.nlargest(23).plot(kind='barh')

# y_pred = clf.predict(X_test)

pickle.dump(clf, open('model.pkl','wb'))


