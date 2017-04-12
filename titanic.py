import numpy as np
import pandas as pd
from sklearn import linear_model as lm
from sklearn import preprocessing as pp

label_encoder = pp.LabelEncoder();

titanic_test=pd.read_csv("test.csv")
titanic_train=pd.read_csv("train.csv")


#get title
def get_title(name):
	if '.' in name:
		return name.split(',')[1].split('.')[0].strip()
	else:
		return 'Unknown';

def get_surname(name):
	if "," in name:
		return name.split(",")[0]
	else:
		return 'Unknown';

def replace_title(x):
	title = x.Title;
	if title in ['Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Rev', 'Sir' ]:
		return 'Mr'
	elif title in ['the Countess', 'Mme', 'Lady']:
		return 'Mrs'
	elif title in ['Mlle', 'Ms']:
		return 'Miss'
	elif title =='Dr':
		if x.Sex=='male':
			return 'Mr'
		else:
			return 'Mrs'
	else:
		return title

#to find out how many people with a given surname
def replace_surname(surname):

	#if it's more than 3 then gropu them into one group otherwise into a separate group;
	#this is assuming that most likely 2 people with same surname might help each other to survive but more than that might compete each other to survive
	if titanic_train[titanic_train.Surname.str.contains(surname)]['PassengerId'].count()>=3:
		return 2;
	else:
		return 1;


##grouped=titanic_train.groupby(['Embarked'])
#train data
kids_mean_age_train = titanic_train[pd.notnull(titanic_train["Age"]) & titanic_train["Name"].str.contains("Master")]["Age"].mean();
titanic_train.loc[pd.isnull(titanic_train["Age"]) & titanic_train["Name"].str.contains("Master"), 'Age']=kids_mean_age_train

miss_mean_age_train = titanic_train[ (pd.notnull(titanic_train["Age"])) &  (titanic_train["Sex"]=="female") & (titanic_train["Name"].str.contains("Miss.")) ]["Age"].mean();
titanic_train.loc[ (pd.isnull(titanic_train["Age"])) &  (titanic_train["Sex"]=="female") & (titanic_train["Name"].str.contains("Miss")), 'Age']=miss_mean_age_train

mr_mean_age_train = titanic_train[ (pd.notnull(titanic_train["Age"])) &  (titanic_train["Sex"]=="male") & (titanic_train["Name"].str.contains("mr.")) ]["Age"].mean()
titanic_train.loc[ (pd.isnull(titanic_train["Age"])) &  (titanic_train["Sex"]=="male") & (titanic_train["Name"].str.contains("mr.")), 'Age']=mr_mean_age_train

mean_fare = titanic_train[ (pd.notnull(titanic_train["Fare"])) ]["Fare"].mean()
titanic_train.loc[ pd.isnull(titanic_train["Fare"]), 'Fare'] = mean_fare

titanic_train["Title"]=titanic_train.Name.map(get_title)
titanic_train.Title = titanic_train.apply(replace_title, axis=1)

#assuming people with same surname >=3 might impact their survival rate; 
titanic_train.Surname = titanic_train.Name.map(get_surname)
train_surname= titanic_train.Surname.map(replace_surname)



titanic_train.Fare = titanic_train.Fare.astype(int)
titanic_train.loc[pd.isnull(titanic_train["Embarked"]),"Embarked"]='S'
train_sex = label_encoder.fit_transform(titanic_train["Sex"])
train_embarked = label_encoder.fit_transform(titanic_train["Embarked"])
train_title = label_encoder.fit_transform(titanic_train["Title"])
titanic_train.loc[titanic_train["Age"].isnull(),"Age"]= titanic_train["Age"].mean()



###test data
kids_mean_age_test = titanic_train[pd.notnull(titanic_train["Age"]) & titanic_train["Name"].str.contains("Master")]["Age"].mean();
titanic_test.loc[pd.isnull(titanic_train["Age"]) & titanic_train["Name"].str.contains("Master"), 'Age']=kids_mean_age_train

miss_mean_age_test = titanic_test[ (pd.notnull(titanic_test["Age"])) &  (titanic_test["Sex"]=="female") & (titanic_test["Name"].str.contains("Miss.")) ]["Age"].mean();
titanic_test.loc[ (pd.isnull(titanic_test["Age"])) &  (titanic_test["Sex"]=="female") & (titanic_test["Name"].str.contains("Miss")), 'Age']=miss_mean_age_test

mr_mean_age_test = titanic_test[ (pd.notnull(titanic_test["Age"])) &  (titanic_test["Sex"]=="male") & (titanic_test["Name"].str.contains("mr.")) ]["Age"].mean()
titanic_test.loc[ (pd.isnull(titanic_test["Age"])) &  (titanic_test["Sex"]=="male") & (titanic_test["Name"].str.contains("mr.")), 'Age']=mr_mean_age_test


mean_fare = titanic_test[ (pd.notnull(titanic_test["Fare"])) ]["Fare"].mean()
titanic_test.loc[ pd.isnull(titanic_test["Fare"]), 'Fare'] = mean_fare

titanic_test["Title"]=titanic_test.Name.map(get_title)
titanic_test.Title = titanic_test.apply(replace_title, axis=1)

titanic_test.Surname = titanic_test.Name.map(get_surname)
test_surname= titanic_test.Surname.map(replace_surname)

titanic_test.Fare = titanic_test.Fare.astype(int)
titanic_test.loc[pd.isnull(titanic_test["Embarked"]),"Embarked"]='S'
test_sex= label_encoder.fit_transform(titanic_test["Sex"])
test_embarked = label_encoder.fit_transform(titanic_test["Embarked"])
test_title = label_encoder.fit_transform(titanic_test["Title"])
titanic_test.loc[titanic_test["Age"].isnull(),"Age"]= titanic_test["Age"].mean()



train_features=pd.DataFrame([train_sex
							, train_embarked
							, titanic_train["Pclass"]
							, titanic_train["Age"]
							, titanic_train["SibSp"]
							, titanic_train["Parch"]
							, train_title
							, titanic_train["Fare"]
							,train_surname
							]).T
test_features=pd.DataFrame([test_sex
							, test_embarked
							, titanic_test["Pclass"]
							, titanic_test["Age"]
							, titanic_test["SibSp"]
							, titanic_test["Parch"]
							, test_title
							, titanic_test["Fare"]
							, test_surname
							]).T

log_model = lm.LogisticRegression();

log_model.fit(X=train_features, y=titanic_train["Survived"])

test_preds=log_model.predict(X=test_features)

submission = pd.DataFrame({'PassengerId':titanic_test["PassengerId"],'Survived':test_preds})

submission.to_csv("test_pred10.csv", index=False)
