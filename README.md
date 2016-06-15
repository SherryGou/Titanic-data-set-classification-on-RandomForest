# Titanic-data-set-classification-on-RandomForest
This code reference to:
1. https://www.kaggle.com/syeddanish/titanic/handling-missing-values-in-python
2. https://www.kaggle.com/milkingcows4u/titanic/choooo.
3. http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#examples-using-sklearn-ensemble-randomforestclassifier

Data set is from https://www.kaggle.com/c/titanic/data?train.csv

This project is based on the titanic data set, about feature engineering, dealing with missing data, and predict the survival using Random Forest Algorithm.

1.The visualization of features and their impact on outcomes will be outputted as you run the hw1.py file on terminal.
2.Because there can be only 6 histograms in one picture, you can see the Age_group histogram by making line162 comment line.
3.There are some files: hw1.py is the main part. 
train.csv is the original train data set
test.csv is the original test data set
new_data.csv is the new version of whole data set after feature processing. And new_data.csv is divided into 2 parts:  new_data_train.csv and new_data_test.csv.   prediction.csv is the prediction of survival of new_data_test.csv using features as ’Embarked','Parch','Pclass','Sex','SibSp','Age_group','Fare_group’, and parameters as Alg = RandomForestClassifier(random_state=1, n_estimators=10000,min_samples_split=50)

final_prediction.csv is the prediction of survival of new_data_test using the best parameters.

PS. to calculate the train score, test score and accuracy, it may take some time.
to run the code, you need to install sklearn package.
