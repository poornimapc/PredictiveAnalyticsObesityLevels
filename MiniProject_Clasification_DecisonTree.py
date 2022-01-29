# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 22:32:45 2021

@author: Poornima Peiris
"""
#01 #################### Importing required libraries ###########################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#02 #################### Importing data set and creating a data frame ###########
df_obesity = pd.read_csv('Obesity_dataset.csv', sep=",")
df_obesity.head()
print(df_obesity.shape)
print(df_obesity.info())
print(df_obesity.describe())
df_obesity.dtypes

##################### Round numerical values. Taking to standard format #######
#df_obesity.round(decimals=0)

#03 #################### Missing value Traatment and formating Data ##############
print ("\nMissingvalues:",df_obesity.isnull().sum().values.sum())
df_obesity.isna().sum()

#04 ####################  Outliers Removal column by column ######################

#------------------- Weight --------------------------------------------------- 
sns.boxplot(x=df_obesity['Weight'], palette="rocket_r")
df_obesity.hist(column='Weight')

Q1 = df_obesity["Weight"].quantile(0.25)
Q3 = df_obesity["Weight"].quantile(0.75)
IQR = Q3 - Q1
print(IQR)

Lower_Boundary  = Q1 - (1.5 * IQR)
Upper_Boundary  = Q3 + (1.5 * IQR)
print(Lower_Boundary )
print(Upper_Boundary )

# To print all the data above the upper fence and below the lower fence, add the following code:
df_obesity[((df_obesity["Weight"] < Lower_Boundary ) |(df_obesity["Weight"] > Upper_Boundary ))]


# Filter out the outlier data and print only the potential data. To do so, just negate the preceding result using the ~ operator:
df_obesity = df_obesity[~((df_obesity ["Weight"] < Lower_Boundary ) |(df_obesity["Weight"] > Upper_Boundary ))]

#------------------- Height ---------------------------------------------------
sns.boxplot(x=df_obesity['Height'],palette="Wistia" )
#sns.boxplot( y=df_obesity["Height"],palette="YlOrBr")
df_obesity.hist(column='Height')

Q1 = df_obesity["Height"].quantile(0.25)
Q3 = df_obesity["Height"].quantile(0.75)
IQR = Q3 - Q1
print(IQR)

Lower_Boundary  = Q1 - (1.5 * IQR)
Upper_Boundary  = Q3 + (1.5 * IQR)
print(Lower_Boundary )
print(Upper_Boundary )

# To print all the data above the upper fence and below the lower fence, add the following code:
df_obesity[((df_obesity["Height"] < Lower_Boundary ) |(df_obesity["Height"] > Upper_Boundary ))]


# Filter out the outlier data and print only the potential data. To do so, just negate the preceding result using the ~ operator:
df_obesity = df_obesity[~((df_obesity ["Height"] < Lower_Boundary ) |(df_obesity["Height"] > Upper_Boundary ))]


#------------------- Age ------------------------------------------------------
sns.boxplot(x=df_obesity['Age'],palette="cool" )
#sns.boxplot( y=df_obesity["Height"],palette="YlOrBr")
df_obesity.hist(column='Age')

Q1 = df_obesity["Age"].quantile(0.25)
Q3 = df_obesity["Age"].quantile(0.75)
IQR = Q3 - Q1
print(IQR)

Lower_Boundary  = Q1 - (1.5 * IQR)
Upper_Boundary  = Q3 + (1.5 * IQR)
print(Lower_Boundary )
print(Upper_Boundary )

# To print all the data above the upper fence and below the lower fence, add the following code:
df_obesity[((df_obesity["Age"] < Lower_Boundary ) |(df_obesity["Age"] > Upper_Boundary ))]


# Filter out the outlier data and print only the potential data. To do so, just negate the preceding result using the ~ operator:
df_obesity = df_obesity[~((df_obesity ["Age"] < Lower_Boundary ) |(df_obesity["Age"] > Upper_Boundary ))]

#------------------- FCVC - Frequency of consumption of vegetables ------------
sns.boxplot(x=df_obesity['FCVC'],palette="YlGnBu")
df_obesity.hist(column='FCVC')


Q1 = df_obesity["FCVC"].quantile(0.25)
Q3 = df_obesity["FCVC"].quantile(0.75)
IQR = Q3 - Q1
print(IQR)

Lower_Boundary  = Q1 - (1.5 * IQR)
Upper_Boundary  = Q3 + (1.5 * IQR)
print(Lower_Boundary )
print(Upper_Boundary )

# To print all the data above the upper fence and below the lower fence, add the following code:
df_obesity[((df_obesity["FCVC"] < Lower_Boundary ) |(df_obesity["FCVC"] > Upper_Boundary ))]


# Filter out the outlier data and print only the potential data. To do so, just negate the preceding result using the ~ operator:
df_obesity = df_obesity[~((df_obesity ["FCVC"] < Lower_Boundary ) |(df_obesity["FCVC"] > Upper_Boundary ))]

# --------------------- NCP - Number of main meals ----------------------------

sns.boxplot(x=df_obesity['NCP'],palette="gist_rainbow")
df_obesity.hist(column='NCP')


Q1 = df_obesity["NCP"].quantile(0.25)
Q3 = df_obesity["NCP"].quantile(0.75)
IQR = Q3 - Q1
print(IQR)

Lower_Boundary  = Q1 - (1.5 * IQR)
Upper_Boundary  = Q3 + (1.5 * IQR)
print(Lower_Boundary )
print(Upper_Boundary )

# To print all the data above the upper fence and below the lower fence, add the following code:
df_obesity[((df_obesity["NCP"] < Lower_Boundary ) |(df_obesity["NCP"] > Upper_Boundary ))]


# Filter out the outlier data and print only the potential data. To do so, just negate the preceding result using the ~ operator:
# df_obesity = df_obesity[~((df_obesity ["NCP"] < Lower_Boundary ) |(df_obesity["NCP"] > Upper_Boundary ))]
# outliers not removed in here

# --------------------- CH20 - Consumption of water daily --------------------
sns.boxplot(x=df_obesity['CH2O'],palette="PuRd")
df_obesity.hist(column='CH2O')


# ----------------------FAF - Physical activity frequency ---------------------
sns.boxplot(x=df_obesity['FAF'],palette="CMRmap_r")
df_obesity.hist(column='FAF')


# -----------------------TUE - Time using technology devices ------------------
sns.boxplot(x=df_obesity['TUE'],palette="Greens")
df_obesity.hist(column='TUE')



#05 #################### EDA / Descrrptive Statistics about Class Variable (Chaecking balance or Not) and other


labels = 'Normal_Weight', 'Overweight_Level_I', 'Overweight_Level_II','Obesity_Type_I', 'Insufficient_Weight', 'Obesity_Type_II','Obesity_Type_III'
sizes = [df_obesity.NObeyesdad[df_obesity['NObeyesdad']=='Normal_Weight'].count(), 
         df_obesity.NObeyesdad[df_obesity['NObeyesdad']=='Overweight_Level_I'].count(),
         df_obesity.NObeyesdad[df_obesity['NObeyesdad']=='Overweight_Level_II'].count(),
         df_obesity.NObeyesdad[df_obesity['NObeyesdad']=='Obesity_Type_I'].count(),  
         df_obesity.NObeyesdad[df_obesity['NObeyesdad']=='Insufficient_Weight'].count(), 
         df_obesity.NObeyesdad[df_obesity['NObeyesdad']=='Obesity_Type_II'].count(),
         df_obesity.NObeyesdad[df_obesity['NObeyesdad']=='Obesity_Type_III'].count()]
explode = (0, 0.1,0,0,0,0,0.1)
fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')
plt.title("Proportion of Obesity Level", size = 20)
plt.show()

#---------------- # Basic correlogram

sns.pairplot(df_obesity)

# let's plot pair plot to visualise the attributes all at once
sns.pairplot(data=df_obesity, hue = 'NObeyesdad')


#06 ################################### Data Preprocessing - Convert catogorical to numerical 
df_obesity['Gender'].unique()
df_obesity['Gender']=df_obesity['Gender'].map({'Male':0,'Female':1})
df_obesity['family_history_with_overweight'].unique()
df_obesity['family_history_with_overweight']=df_obesity['family_history_with_overweight'].map({'no':0,'yes':1})
df_obesity['FAVC'].unique()
df_obesity['FAVC']=df_obesity['FAVC'].map({'no':0,'yes':1})
df_obesity['CAEC'].unique()
df_obesity['CAEC']=df_obesity['CAEC'].map({'no':0,'Sometimes':1, 'Frequently':2, 'Always':3  })
df_obesity['SMOKE'].unique()
df_obesity['SMOKE']=df_obesity['SMOKE'].map({'no':0,'yes':1})
df_obesity['SCC'].unique()
df_obesity['SCC']=df_obesity['SCC'].map({'no':0,'yes':1})
df_obesity['CALC'].unique()
df_obesity['CALC']=df_obesity['CALC'].map({'no':0,'Sometimes':1, 'Frequently':2, 'Always':3  })
df_obesity['MTRANS'].unique()
df_obesity['MTRANS']=df_obesity['MTRANS'].map({'Public_Transportation':0,'Walking':1, 'Automobile':2, 'Motorbike':3 ,  'Bike':4  })
df_obesity['NObeyesdad'].unique()
df_obesity['NObeyesdad_num']=df_obesity['NObeyesdad'].map({'Normal_Weight':0,'Overweight_Level_I':1, 'Overweight_Level_II':2, 'Obesity_Type_I':3 ,  'Insufficient_Weight':4 , 'Obesity_Type_II':5 , 'Obesity_Type_III':6  })

df_obesity.dtypes



#07 ################################  Heat Map

# plot using a color palette
# Heat map
#visualization of the correlation matrix using heatmap plot

sns.set()
sns.set(font_scale = 1.25)
sns.heatmap(df_obesity[df_obesity.columns[:28]].corr(), yticklabels=1, fmt = ".1f", square=2, xticklabels=1, linewidths=.5)
plt.show()

df_obesity.dtypes


#################################  Coreation Map


# Pearson Corealtion ----------------------------------------------------------

plt.figure(figsize=(30, 15))
heatmap = sns.heatmap(np.round(np.abs(df_obesity.corr('pearson')),2), vmin=0, vmax=1, annot=True, cmap='BuPu')
heatmap.set_title('Correlation Heatmap - Pearson', fontdict={'fontsize':18}, pad=13);

# Pearson with class variable
df_obesity.corr('pearson')[['NObeyesdad_num']].sort_values(by='NObeyesdad_num', ascending=False)
plt.figure(figsize=(8, 12))
heatmap = sns.heatmap(np.round(np.abs(df_obesity.corr()[['NObeyesdad_num']]),2).sort_values(by='NObeyesdad_num', ascending=False), vmin=0, vmax=1, annot=True, cmap='BuPu')
heatmap.set_title('Features Correlating with Obesity Type - Pearson', fontdict={'fontsize':18}, pad=16);

# Selecting Variables
cor = df_obesity.corr('pearson')
cor_target = abs(cor['NObeyesdad_num'])
relevant_features = cor_target [cor_target > 0.25]
relevant_features

# Spearman Corealtion ---------------------------------------------------------
plt.figure(figsize=(30, 15))
heatmap = sns.heatmap(np.round(np.abs(df_obesity.corr('spearman')),2), vmin=0, vmax=1, annot=True, cmap='RdPu')
heatmap.set_title('Correlation Heatmap - spearman', fontdict={'fontsize':18}, pad=13);

# Spearman with class variable -----------------------------------------------
df_obesity.corr(method='spearman')[['NObeyesdad_num']].sort_values(by='NObeyesdad_num', ascending=False)
plt.figure(figsize=(8, 12))
heatmap = sns.heatmap(np.round(np.abs(df_obesity.corr(method='spearman')[['NObeyesdad_num']]),2).sort_values(by='NObeyesdad_num', ascending=False), vmin=0, vmax=1, annot=True, cmap='RdPu')
heatmap.set_title('Features Correlating with Pbesity Type - Spearman', fontdict={'fontsize':18}, pad=16);

# Selecting variables
cor = df_obesity.corr('spearman')
cor_target = abs(cor['NObeyesdad_num'])
relevant_features = cor_target [cor_target > 0.25]
relevant_features
#08 ######################  Final Features #######################################

df_obesity.dtypes

#feature_names = ['Weight', 'family_history_with_overweight', 'FCVC']
feature_names = ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight' , 'FAVC' ,'FCVC' ]
#feature_names = ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight' ]

X=df_obesity[feature_names]
y=df_obesity.NObeyesdad  # Class label coloumn (in categorical)


# encoding categorical data e.g. obesity outcome as a dummy variable
y,class_names = pd.factorize(y)
target_names=list(map(str,class_names)) # To solve object of type 'numpy.int64' has no len() error in classification_report

#09 ######################  Splitting the dataset into the Training set and Test set ######
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 42)

#10 ######################  Application of DecisionTreeClassifier ################
# Fitting Classifier to the Training Set with deapth, pruning
from sklearn.tree import DecisionTreeClassifier

# Selecting Gini or Entropy
from sklearn.metrics import accuracy_score
dtree = DecisionTreeClassifier(criterion='gini')
dtree.fit(X_train, y_train)
pred = dtree.predict(X_test)
print('Criterion=gini', accuracy_score(y_test, pred))
dtree = DecisionTreeClassifier(criterion='entropy')
dtree.fit(X_train, y_train)
pred = dtree.predict(X_test)
print('Criterion=entropy', accuracy_score(y_test, pred))


# Apllication of DT
classifier = DecisionTreeClassifier(criterion='entropy', random_state=42)
#classifier = DecisionTreeClassifier(criterion='entropy',max_depth=5, random_state=42)
classifier.fit(X_train, y_train)


#11 ######################  Model performance evaluation on Training set #########
y_pred_train =classifier.predict(X_train)

from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report


accuracy = metrics.accuracy_score(y_train, y_pred_train)
print("Accuracy: {:.2f}".format(accuracy))
cm=confusion_matrix(y_train,y_pred_train)
print('Confusion Matrix: \n', cm)
print(classification_report(y_train, y_pred_train, target_names=target_names))

#######################  Model performance evaluation on Test set #########
y_pred=classifier.predict(X_test)

# Classification results on test set
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}".format(accuracy))

from sklearn.metrics import confusion_matrix, classification_report
cm=confusion_matrix(y_test,y_pred)
print('Confusion Matrix: \n', cm)
print(classification_report(y_test, y_pred, target_names=target_names))


#12##################################### Tree drawing ###########################

from sklearn.tree import export_graphviz
#from sklearn.externals.six import StringIO
from six import StringIO
from IPython.display import Image
import pydotplus
import os
os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"

dot_data=StringIO()
export_graphviz(classifier, out_file=dot_data,filled=True, rounded=True,special_characters=True, feature_names=feature_names,class_names=class_names)
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('ob1.png')
Image(graph.create_png())

# -------------------------- getting rules

from sklearn.tree import export_text
tree_rules = export_text(classifier, feature_names=list(X_train.columns))

#13 ########################### Post Processing ###############################
#######################  Application of DecisionTreeClassifier - Pruning  ################

# Fitting Classifier to the Training Set with deapth, pruning
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy',max_depth=6, random_state=42)
classifier.fit(X_train, y_train)


#######################  Model performance evaluation on Training set #########
y_pred_train =classifier.predict(X_train)

from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report


accuracy = metrics.accuracy_score(y_train, y_pred_train)
print("Accuracy: {:.2f}".format(accuracy))
cm=confusion_matrix(y_train,y_pred_train)
print('Confusion Matrix: \n', cm)
print(classification_report(y_train, y_pred_train, target_names=target_names))

#######################  Model performance evaluation on Test set #########
y_pred=classifier.predict(X_test)

# Classification results on test set
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}".format(accuracy))

from sklearn.metrics import confusion_matrix, classification_report
cm=confusion_matrix(y_test,y_pred)
print('Confusion Matrix: \n', cm)
print(classification_report(y_test, y_pred, target_names=target_names))


#14 ##################################### Tree drawing ###########################

from sklearn.tree import export_graphviz
#from sklearn.externals.six import StringIO
from six import StringIO
from IPython.display import Image
import pydotplus
import os
os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"

dot_data=StringIO()
export_graphviz(classifier, out_file=dot_data,filled=True, rounded=True,special_characters=True, feature_names=feature_names,class_names=class_names)
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('ob1_pruned.png')
Image(graph.create_png())

# -------------------------- getting rules

from sklearn.tree import export_text
tree_rules = export_text(classifier, feature_names=list(X_train.columns))





