import numpy as np
import pandas as pd
from sklearn import linear_model as lm
from sklearn import preprocessing as pp

label_encoder = pp.LabelEncoder();

titanic_test=pd.read_csv("test.csv")
titanic_train=pd.read_csv("train.csv")


#get cabin_count
def get_cabin_count(cabin):
	return len(str(cabin).split(" "))

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

#this will require 2 parameters; the titanic data with a new column Title1 and title for each row
def get_mean_age(x):
	titanic_df, title = x

	return titanic_df[ (titanic_df.Title==title) & pd.notnull(titanic_df.Age)]["Age"].mean()

def fill_missing_age(row):
	
	titanic_df, title = x;
	return titanic_df[ (titanic_df.Title==title) & pd.notnull(titanic_df.Age)]["Age"].mean()

def get_fare_value_type(x):
	df, y = x
	if y>df.Fare.mean():
		return 2;
	else:
		return 1;



##grouped=titanic_train.groupby(['Embarked'])
#train data
# kids_mean_age_train = titanic_train[pd.notnull(titanic_train["Age"]) & titanic_train["Name"].str.contains("Master")]["Age"].mean();
# titanic_train.loc[(pd.isnull(titanic_train["Age"]) & titanic_train["Name"].str.contains("Master")), 'Age']=kids_mean_age_train

# miss_mean_age_train = titanic_train[ (pd.notnull(titanic_train["Age"])) &  (titanic_train["Sex"]=="female") & (titanic_train["Name"].str.contains("Miss.")) ]["Age"].mean();
# titanic_train.loc[ (pd.isnull(titanic_train["Age"])) &  (titanic_train["Sex"]=="female") & (titanic_train["Name"].str.contains("Miss")), 'Age']=miss_mean_age_train

# mr_mean_age_train = titanic_train[ (pd.notnull(titanic_train["Age"])) &  (titanic_train["Sex"]=="male") & (titanic_train["Name"].str.contains("mr.")) ]["Age"].mean()
# titanic_train.loc[ (pd.isnull(titanic_train["Age"])) &  (titanic_train["Sex"]=="male") & (titanic_train["Name"].str.contains("mr.")), 'Age']=mr_mean_age_train


mean_fare = titanic_train[ (pd.notnull(titanic_train["Fare"])) ]["Fare"].mean()
titanic_train.loc[ pd.isnull(titanic_train["Fare"]), 'Fare'] = mean_fare
titanic_train.Fare = np.log1p(titanic_train.Fare)

titanic_train["Title"]=titanic_train.Name.map(get_title)
#titanic_train["Title1"]=titanic_train.Name.map(get_title) #will be used for determining missing age

#fill missing ages based on title
titanic_train.loc[pd.isnull(titanic_train.Age), 'Age'] = titanic_train[pd.isnull(titanic_train.Age)].Title.map(lambda x: get_mean_age([titanic_train,x]))
titanic_train.Title = titanic_train.apply(replace_title, axis=1)



#assuming people with same surname >=3 might impact their survival rate; 
titanic_train.Surname = titanic_train.Name.map(get_surname)
train_surname= titanic_train.Surname.map(replace_surname)



titanic_train.Fare = titanic_train.Fare.astype(int)
titanic_train.loc[pd.isnull(titanic_train["Embarked"]),"Embarked"]='S'
train_sex = label_encoder.fit_transform(titanic_train["Sex"])


#if there are any missing age still then this will set to mean of the complete dataset
titanic_train.loc[titanic_train["Age"].isnull(),"Age"]= titanic_train["Age"].mean() 

#a = pd.DataFrame(titanic_train['Age'])
#a['Survived'] = pd.DataFrame(titanic_train['Survived'])
 #ggplot(a, aes(x='Age',fill='factor(Survived)')) + geom_histogram()

train_age_fare = titanic_train.Age * titanic_train.Fare   ##didn't work; reduced the score
train_class_fare = titanic_train.Fare/titanic_train.Pclass
train_class_age = titanic_train.Pclass.apply(np.square) * titanic_train.Age ##didn't work; reduced the score
train_family = (titanic_train.Parch + titanic_train.SibSp + 1)
train_fare_person = titanic_train.Fare/train_family
#train_family=train_family.apply(np.sqrt);

train_family[(train_family<4)]=1
train_family[train_family>=4]=2

train_fare_value_type = titanic_train.Fare.apply(lambda x: get_fare_value_type([titanic_train, x]))

###cabin
titanic_train['Cabincount']=titanic_train.Cabin.apply(get_cabin_count)
titanic_train.loc[ pd.isnull(titanic_train.Cabin),'Cabincount']=0
#titanic_train.loc[ titanic_train.Cabincount>0, 'Cabincount']=1
cabincount_train=titanic_train.Cabincount

##adding new features to consider statistical interaction or synergy affect between predictors
titanic_train['fareXpclass']= titanic_train.Pclass * titanic_train.Fare;




###test data
# kids_mean_age_test = titanic_test[pd.notnull(titanic_test["Age"]) & titanic_test["Name"].str.contains("Master")]["Age"].mean();
# titanic_test.loc[pd.isnull(titanic_test["Age"]) & titanic_test["Name"].str.contains("Master"), 'Age']=kids_mean_age_test

# miss_mean_age_test = titanic_test[ (pd.notnull(titanic_test["Age"])) &  (titanic_test["Sex"]=="female") & (titanic_test["Name"].str.contains("Miss.")) ]["Age"].mean();
# titanic_test.loc[ (pd.isnull(titanic_test["Age"])) &  (titanic_test["Sex"]=="female") & (titanic_test["Name"].str.contains("Miss")), 'Age']=miss_mean_age_test

# mr_mean_age_test = titanic_test[ (pd.notnull(titanic_test["Age"])) &  (titanic_test["Sex"]=="male") & (titanic_test["Name"].str.contains("mr.")) ]["Age"].mean()
# titanic_test.loc[ (pd.isnull(titanic_test["Age"])) &  (titanic_test["Sex"]=="male") & (titanic_test["Name"].str.contains("mr.")), 'Age']=mr_mean_age_test


mean_fare = titanic_test[ (pd.notnull(titanic_test["Fare"])) ]["Fare"].mean()
titanic_test.loc[ pd.isnull(titanic_test["Fare"]), 'Fare'] = mean_fare
titanic_test.Fare = np.log1p(titanic_test.Fare)

titanic_test["Title"]=titanic_test.Name.map(get_title)
#titanic_test["Title1"]=titanic_test.Name.map(get_title)


#fill missing ages based on title; this works
titanic_test.loc[pd.isnull(titanic_test.Age), 'Age'] = titanic_test[pd.isnull(titanic_test.Age)].Title.map(lambda x: get_mean_age([titanic_test,x]))
titanic_test.Title = titanic_test.apply(replace_title, axis=1)


titanic_test.Surname = titanic_test.Name.map(get_surname)
test_surname= titanic_test.Surname.map(replace_surname)

titanic_test.Fare = titanic_test.Fare.astype(int)
titanic_test.loc[pd.isnull(titanic_test["Embarked"]),"Embarked"]='S'
test_sex= label_encoder.fit_transform(titanic_test["Sex"])
titanic_test.loc[titanic_test["Age"].isnull(),"Age"]= titanic_test["Age"].mean()

test_age_fare = titanic_test.Age * titanic_test.Fare
test_class_fare = titanic_test.Fare/titanic_test.Pclass
test_class_age = titanic_test.Pclass.apply(np.square) * titanic_test.Age
test_family = (titanic_test.Parch + titanic_test.SibSp + 1)
test_fare_person = titanic_test.Fare/test_family

#test_family = test_family.apply(np.sqrt);
test_family[(test_family<4)]=1
test_family[test_family>=4]=2

test_fare_value_type = titanic_test.Fare.apply(lambda x: get_fare_value_type([titanic_test, x])) ##din't improve score

###cabin
titanic_test['Cabincount']=titanic_test.Cabin.apply(get_cabin_count)
titanic_test.loc[ pd.isnull(titanic_test.Cabin),'Cabincount']=0
#titanic_test.loc[ titanic_test.Cabincount>0, 'Cabincount']=1
cabincount_test=titanic_test.Cabincount

titanic_test['fareXpclass']= titanic_test.Pclass * titanic_test.Fare;


titanic_train = pd.get_dummies(titanic_train, prefix="col");
titanic_test = pd.get_dummies(titanic_test, prefix="col");

train_features=pd.DataFrame([train_sex
							, titanic_train["Pclass"]
							, titanic_train["Age"]
							, titanic_train["SibSp"]
							, titanic_train["Parch"]
							, titanic_train["Fare"]
							,train_surname
							#,train_fare_value_type
							#,train_class_fare  
							 ,train_family
							 #,train_fare_person
							,cabincount_train
							,titanic_train.col_S  		#embarked
							,titanic_train.col_Q		#embarked
							,titanic_train.col_C		#embarked
							,titanic_train.col_Master	#title
							,titanic_train.col_Miss		#title
							,titanic_train.fareXpclass
							]).T
test_features=pd.DataFrame([test_sex
							, titanic_test["Pclass"]
							, titanic_test["Age"]
							, titanic_test["SibSp"]
							, titanic_test["Parch"]
							, titanic_test["Fare"]
							, test_surname
							#,test_fare_value_type
							#,test_class_fare
							 ,test_family
							 #,test_fare_person
							,cabincount_test
							,titanic_test.col_S
							,titanic_test.col_Q
							,titanic_test.col_C
							,titanic_test.col_Master
							,titanic_test.col_Miss
							,titanic_test.fareXpclass

							]).T


#data transfromation to higher degree polynomial
poly = pp.PolynomialFeatures(2)
train_features = poly.fit_transform(train_features)
test_features= poly.fit_transform(test_features)

log_model = lm.LogisticRegression();

log_model.fit(X=train_features, y=titanic_train["Survived"])
log_model.score(X = train_features ,
                y = titanic_train["Survived"])

test_preds=log_model.predict(X=test_features)

submission = pd.DataFrame({'PassengerId':titanic_test["PassengerId"],'Survived':test_preds})

submission.to_csv("test_pred10.csv", index=False)

# from sklearn import datasets
# from sklearn.naive_bayes import GaussianNB
# titanic_dataset_tr = sklearn.datasets.base.Bunch(data=train_features, target=titanic_train.Survived)
# gnb = GaussianNB()
# y_pred = gnb.fit(titanic_dataset_tr.data, titanic_dataset_tr.target).predict(test_features)

#import statsmodels.regression.linear_model.OLS as ols
#x=np.array(data_X_train.PoolArea)
#ols = sm.OLS(y, x)
#ols_result=ols.fit()
#ols_result.summary()