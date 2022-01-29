# PredictiveAnalyticsObesityLevels
**This is to estimation of obesity levels based on eating habits and  physical condition by applying predictive modeling technique. Supervised Leaning Classification is used in here

:credit_card: **Project Title** </br>
# Estimation of Obesity Levels Based on Eating Habits and Physical Condition by Using Predictive Modelling Technique


![obesity_image](https://user-images.githubusercontent.com/86964329/151661032-e094ca2b-b916-4880-9025-1df2818433bb.PNG)

:blue_book: **Project Overview** </br>
Objective of this mini project is to estimation of obesity levels based on eating habits and physical condition by applying predictive modeling technique. Classification under the 
Supervised learning adapted to carry out the project.

The data set contain data for estimation of obesity levels in individuals from the countries of Mexico, Peru and Colombia, based on their eating habits and physical condition. There are 17 attributes and 2111 records.
URL https://archive.ics.uci.edu/ml/datasets/Estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition


:computer:**Technical Overview** </br>
I have implement this mini project by using Spider (as Scientific Python Development Environment) for data programing. There were several python libraries being used for analyzing the data, model building and model validation. They are Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn, pydotplus.

:computer:**Methodology** </br>
First data set was downloaded and understand it by carrying out a descriptive statistic. Then identified class label. Then applied preprocessing techniques and features were selected by using statistical method coefficient. Classification was done after divided data into two parts and applied decision tree algorithm. Model evaluation done using accuracy, precision, recall, f1-score and confusion matrix. Decision tree pruning was done as post processing techniques. 
Below diagram illustrated the adapted methodology.

![image](https://user-images.githubusercontent.com/86964329/151661352-bfbea439-ca09-4933-960a-4ea0e0f3bb7c.png)

:computer:**Pre-processing Activities** </br>

**Handling Missing Values**</br> 
Missing value occurred may be due to many reasons. By handling missing values, it will increase performance of the model. Common methods are replacing missing values with mean or median of entire column (imputation). There are no missing values in the data set

**Encoding Categorical Data**</br>
Categorical data can’t be use at mathematical equations. Such as ‘Male’ and ‘Female’ in gender column. These columns need to convert to numerical value. Encoded categorical columns, Gender, family_history_with_overweight, FAVC, CAEC, SMOKE, SCC, CALC, MTRANS using map function. 

**Identifying Class Variable**</br>
‘NObeyesdad’ class variable in numerical additionally added.
NObeyesdad” (Obesity Level) – categorical variable in the data set selected as a class variable and it is a multi-class variable which has seven types as 
- Normal_Weight
- Overweight_Level_I
- Overweight_Level_II
- Obesity_Type_I
- Insufficient_Weight
- Obesity_Type_II
- Obesity_Type_III
 
**Factorization**</br> 
Use pandas.factorize to encode the multi class categorical variable to apply in DT.

**Feature Selection**</br>
Selection of features contributed most to class variable (prediction variable). Statistical method Pearson and spearman coefficient was used to select most significant features. correlation applied and cheeked. Correlation value greater than 0.25 was selected as best features. Under post processing activities Correlation value and variable selection change and tested.

**Outlier Removals**</br>
Outlier is a data point that differs significantly from other observations. By using statistical method IQR outliers were deleted in numerical columns. Outliers in Weight, Height, Agecolumn was deleted. To identify outliers boxplot and histogram of data column separately analyzed. General rule applied.

:page_facing_up: **Source Code**</br>
There is a one python file and it containing reading data , preprocessing , model application, validation.  
