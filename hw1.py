#code refer to https://www.kaggle.com/syeddanish/titanic/handling-missing-values-in-python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import csv
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from subprocess import check_output

print(check_output(["ls", "/Users/xuanxuan/Documents/inf552/hw1/titanic"]).decode("utf8")) #you can edit the path
# Any results you write to the current directory are saved as output.

import scipy as sp
import seaborn as sns
import re
# %matplotlib inline
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
# from slearn.ensemble import RandomForestClassifier

train=pd.read_csv('/Users/xuanxuan/Documents/inf552/hw1/titanic/train.csv')#you can edit the path
test=pd.read_csv('/Users/xuanxuan/Documents/inf552/hw1/titanic/test.csv')#you can edit the path

#PART1:feature engineering
test['Survived']=0 #adding survived column to 0 for appending it with train data

#Creating a new variable which denotes the salutation
def parse_des(x):
    x=re.findall(r',\s\w+.',x)
    return (re.findall(r'\w+',str(x)))[0]
train['desig']=train['Name'].apply(parse_des)
test['desig']=test['Name'].apply(parse_des)

data=train.append(test)
# train.to_csv('/Users/xuanxuan/Documents/inf552/hw1/titanic/new_data.csv')

data.set_index(np.arange(0,np.size(data,axis=0)),inplace=True) #no. of elements in the array. 0 col
#checking all the salutation that are extracted from the data
print train.desig.value_counts()

#To create a reference table based on desig, Pclass and average age.
dpa_ref = pd.DataFrame(data.groupby(['desig','Pclass']).Age.mean().dropna())
#print dpa_ref 
#To create a reference table based on desig, Pclass, Embarked and average age.
dpea_ref = pd.DataFrame(data.groupby(['desig','Pclass','Embarked']).Age.mean().dropna())
#Ms is similar to Miss
data[data.desig=="Ms"]="Miss"
#the table to show the Age-missing atrributes
print pd.DataFrame(data[data.Age.isnull()].groupby(['desig','Pclass']).Age.max())
print dpa_ref
for item in data[data.Age.isnull()].index:
	data.Age[item] = np.around(dpa_ref.ix[data.desig[item],data.Pclass[item]].Age)#.ix stands for [] but it can be used to locate multi-dimension index
#until now, we deal with the missing data of Age, continuing deal with Embarked atrribute
print dpea_ref
print data[data.Embarked.isnull()]#+'ttt'
#we can oberserve the dpea_ref table to find out embared atrribute using age,desig and Pclass
for item in data[data.Embarked.isnull()].index:
	data.Embarked[item] = 'S'
	#print data.Embarked[item]+' sss'
# print data,'sssssss'
print data[data.Embarked.isnull()]+'xnxnxnxnxn' #there is no such DataFrame. We fill the attribute successfully 
# with open('new_data.csv', 'w') as csvdata:
# 	for row in data:
# 		a = csv.writer(csvdata)
# 		a.writerows(data)
print type(data)
# print data
# to generate the new data which is already processed!~
data.to_csv('/Users/xuanxuan/Documents/inf552/hw1/titanic/new_data.csv')

# new_train = pd.read_csv('/Users/xuanxuan/Documents/inf552/hw1/titanic/new_train.csv')
# new_test = pd.read_csv('/Users/xuanxuan/Documents/inf552/hw1/titanic/new_data_test.csv')
new_data_train = pd.read_csv('/Users/xuanxuan/Documents/inf552/hw1/titanic/new_data_train.csv')
new_data_test = pd.read_csv('/Users/xuanxuan/Documents/inf552/hw1/titanic/new_data_test.csv')
print ("Shape of new train set:",new_data_train.shape,"\n")
print("Shape of new test set:",new_data_test.shape,"\n")

#reference to: https://www.kaggle.com/milkingcows4u/titanic/choooo
#find all the unique feature values
print("Train set info.")
for col in new_data_train:
	uniq = new_data_train[col].unique()
	print("'{}' has {} unique values".format(col,uniq.size))
	if(uniq.size>20):
		print "Only list 10 unique values:"
	print(uniq[0:9])
	print("\n-----------------------------------------------------")
print("\n-----------------------------------------------------\n")
print("Test set info.")
for col in new_data_test:
	uniq = new_data_test[col].unique()
	print("'{}' has {} unique values".format(col,uniq.size))
	if(uniq.size>20):
		print "Only list 10 unique values:"
	print(uniq[0:9])
	print("\n-----------------------------------------------------")


#convert non-numeric values into numeric ones:Sex, Embarked. And the missing value has been dealt with already
#feature cleaning
new_data_train.loc[new_data_train["Sex"]=="male","Sex"] = 0
new_data_train.loc[new_data_train["Sex"]=="female","Sex"] = 1

new_data_test.loc[new_data_test["Sex"]=="male","Sex"] = 0
new_data_test.loc[new_data_test["Sex"]=="female","Sex"] = 1

new_data_test.loc[new_data_test["Parch"]==9,"Parch"] = 6
#C=0,S=1,Q=2
new_data_train.loc[new_data_train["Embarked"]=="C","Embarked"] = 0
new_data_train.loc[new_data_train["Embarked"]=="S","Embarked"] = 1
new_data_train.loc[new_data_train["Embarked"]=="Q","Embarked"] = 2

new_data_test.loc[new_data_test["Embarked"]=="C","Embarked"] = 0
new_data_test.loc[new_data_test["Embarked"]=="S","Embarked"] = 1
new_data_test.loc[new_data_test["Embarked"]=="Q","Embarked"] = 2
#we can see if the Embarked attribute has been processed successfully
print new_data_train[new_data_train.Embarked.isnull()],'ssssssss'
print new_data_test[new_data_test.Embarked.isnull()],'nnnnnnnn'

print "\nVISUALIZATION OF THE TRAIN DATA BASED ON FEATURES"
print "=================================================\n\n"
sns.set_style("whitegrid")
# ((axis1,axis2),(axis3,axis4)(axis5,axis6)) = plt.subplots(3,2,sharey=True,figsize=(20,30))
f,((axis1,axis2), (axis3,axis4), (axis5,axis6)) = plt.subplots(3,2, sharey=True, figsize=(20,30))
sns.barplot(x='Embarked',y='Survived',data=new_data_train,order=[0,1,2],ax=axis1,palette="Blues_d")
axis1.set_xticklabels(['S','C','Q'],rotation=0)

sns.barplot(x='Pclass',y='Survived',data=new_data_train,order=[1,2,3],ax=axis2,palette="Blues_d")
axis2.set_xticklabels(['First','Second','Third'],rotation=0)

sns.barplot(x='Sex',y='Survived',data=new_data_train,order=[0,1],ax=axis3,palette="Blues_d")
axis3.set_xticklabels(['M','F'],rotation=0)

sns.barplot(x='Parch',y='Survived',data=new_data_train,order=[0,1,2,3,4,5,6],ax=axis4,palette="Blues_d")

sns.barplot(x='SibSp',y='Survived',data=new_data_train,order=[0,1,2,3,4,5,8],ax=axis5,palette="Blues_d")
# plt.show()
 #to get the survival rate for each feature
for feature in ['Embarked','Pclass','Sex','Parch','SibSp']:
 	feature_survived = pd.crosstab(new_data_train[feature],new_data_train['Survived'])
 	feature_survived_frac = feature_survived.apply(lambda r:r/r.sum(),axis=1)
 	print("Tables for {}\n\n{}\n\n{}\n" .format(feature,feature_survived,feature_survived_frac))
 	print("-----------------------------------------\n")

#add the attribute age group and fare group
# survival_age = new_data_train[['Age','Survived']].groupby(['Age'],as_index=False).mean()
new_data_train['Age_group'] = new_data_train.apply(lambda r: int(r.Age/3),axis=1)
# sns.barplot(x='Age',y='Survived',data=survival_age)
# plt.show()
new_data_test['Age_group'] = new_data_test.apply(lambda r: int(r.Age/3), axis=1)
survival_agegroup = new_data_train[['Age_group','Survived']].groupby(['Age_group'],as_index=False).mean()
sns.barplot(x='Age_group',y='Survived',data=survival_agegroup)
#group by fare
new_data_train['Fare_group'] = new_data_train.apply(lambda r:int(r.Fare/6.0),axis=1)
new_data_test['Fare_group'] = new_data_test.apply(lambda r:int(r.Fare/6.0),axis=1)
survival_faregroup = new_data_train[['Fare_group','Survived']].groupby(['Fare_group'],as_index=False).mean()
sns.barplot(x='Fare_group',y='Survived',data=survival_faregroup)
plt.show()

#features to do training
training_features = ['Age','Embarked','Fare','Parch','Pclass','Sex','SibSp','Age_group','Fare_group']
X_train,X_test,y_train,y_test = cross_validation.train_test_split(new_data_train[training_features],new_data_train['Survived'],test_size=0.1,random_state=0)

#standardize the features:
scaler = StandardScaler().fit(X_train)
X_train_trans = scaler.transform(X_train)
X_test_trans = scaler.transform(X_test)
# print type(new_data_test.Age),'ttttttttt'
final_test_trans = scaler.transform(new_data_test[training_features])

X_new = np.delete(X_train_trans,[5,6],axis=1)
#http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#examples-using-sklearn-ensemble-randomforestclassifier
feature_label1 = ['Age','Embarked','Fare','Parch','Pclass','Sex','SibSp']
X_train1 = np.delete(X_train_trans,[7,8],axis=1)
Alg = RandomForestClassifier(random_state=1, n_estimators=10000,min_samples_split=50)
forest = Alg.fit(X_train1,y_train)
importances = forest.feature_importances_
# print feature_importances
indices = np.argsort(importances)[::-1]
for x in range(X_train1.shape[1]):
	print("%2d) %-*s %f" % (x + 1, 30, 
                             feature_label1[indices[x]], 
                             importances[indices[x]]))
feature_label2 = ['Embarked','Parch','Pclass','Sex','SibSp','Age_group','Fare_group']
X_train2 = np.delete(X_train_trans,[0,2],axis=1)
Alg = RandomForestClassifier(random_state=1, n_estimators=10000,min_samples_split=50)
forest = Alg.fit(X_train2,y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for x in range(X_train2.shape[1]):
	print("%2d) %-*s %f" % (x + 1, 30, 
                             feature_label2[indices[x]], 
                             importances[indices[x]]))	
#scores:Returns the mean accuracy on the given test data and labels
Alg = RandomForestClassifier(random_state=1, n_estimators=500,min_samples_split=5,min_samples_leaf=3)
forest = Alg.fit(X_train_trans,y_train)
train_score = forest.score(X_train_trans,y_train)
test_score = forest.score(X_test_trans,y_test)
print ("Train Score: %0.3f\nTest Score: %0.3f" %(train_score, test_score))

#Prediction!
prediction = forest.predict(final_test_trans)
new_frame = pd.DataFrame({"PassengerId":new_data_test["PassengerId"],"Name":new_data_test["Name"],"Survived":prediction})
new_frame.to_csv('/Users/xuanxuan/Documents/inf552/hw1/titanic/prediction.csv')

#in order to improve the performance of prediction, we can use cross-validation to 
 #get the best attributes so as do better prediction
cross_validation_score = cross_validation.cross_val_score(forest,X_train_trans,y_train,cv=3)
print("Accuracy:%4f (+/- %4f)" % (cross_validation_score.mean(),cross_validation_score.std()*2))
param_grid = {"n_estimators":[200,300,500],"max_features":[5],"min_samples_split":[9],"min_samples_leaf":[6],"bootstrap":[True],"max_depth":[None],"criterion":["gini"]}
forest = RandomForestClassifier()
grid_search = GridSearchCV(forest,param_grid=param_grid)
grid_search.fit(X_train_trans,y_train)
print(grid_search.best_estimator_)
train_score = grid_search.score(X_train_trans,y_train)
test_score = grid_search.score(X_test_trans,y_test)
print("Train Score: %0.4f\nTest Score:%0.4f" %(train_score,test_score))
#use the best parameters
final_alg = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features=5, max_leaf_nodes=None,
            min_samples_leaf=6, min_samples_split=9,
            min_weight_fraction_leaf=0.0, n_estimators=300, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
final_forest = final_alg.fit(X_train_trans,y_train)
train_score = final_forest.score(X_train_trans,y_train)
test_score = final_forest.score(X_test_trans,y_test)
print("Train Score: %0.4f\nTest Score:%0.4f" %(train_score,test_score))
scores = cross_validation.cross_val_score(final_forest, X_train_trans, y_train, cv=3)
print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

#make the final predictions
prediction = final_forest.predict(final_test_trans)
new_frame = pd.DataFrame({"PassengerId":new_data_test["PassengerId"],"Name":new_data_test["Name"],"Survived":prediction})
new_frame.to_csv('/Users/xuanxuan/Documents/inf552/hw1/titanic/final_prediction.csv')







