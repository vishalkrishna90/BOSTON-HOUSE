
# Boston House Price Preciction 

This is a Boston House Price Prediction project in which I tried to train ML Model Very well by that anybody can predict house prices in Boston, I also created a web app for predict house price in boston. You can check my web app, Thank You :)

![Boston House](https://github.com/vishalkrishna90/BOSTON-HOUSE-PRICE-PREDICTION/blob/main/Images/BOSTON_1.jpg)


**Author = Vishal Kumar Mridha**

**Domain = Real Estate**

**Level = Beginner**

**Accuracy Score = 88%>**

**Project Type = End To End Project**

![User Libraries](https://github.com/vishalkrishna90/BOSTON-HOUSE-PRICE-PREDICTION/blob/main/Images/li_im.jpg)

## Process Followed To Complite This Project
- Problem Statement
- Data Collection 
- Data Description
- Data Preprocessing 
- Exploratory Data Analysis (EDA)
- Feature Selection
- Data Scaling
- Model Building
- Model Performances & Feature Importance
- Rebuild Model With Imp Features
- Final Score By The Best Model
- Make Pickle File
- Create New Enviornment
- Create Web App With Streamlit
- Upload All Files In Github repository
- Deploy Model On Heroku

**Web App Overview**

![Web App](https://github.com/vishalkrishna90/BOSTON-HOUSE-PRICE-PREDICTION/blob/main/Images/Web_App_2.png)
## Problem Statement

The problem that we are going to solve here is that given a set of features that describe a house in Boston, our machine learning model must predict the house price. To train our machine learning model with boston housing data, we will be using scikit-learn’s boston dataset.
## Data Collection
I got this dataset from sklearn datasets, Boston House data comes with 
sklearn library, so we don't need to go anywhere else to get Boston House data

```
from sklearn.datasets import load_boston
dfs = load_boston()
df = pd.DataFrame(dfs.data, columns=dfs.feature_names)
df['TARGET'] = dfs.target
```
**Data Overview**

![DataFrame Overview](https://github.com/vishalkrishna90/BOSTON-HOUSE/blob/1981dc1fde4e7961339856d1c31f7f732985d54f/Images/Data_Overview.png)
## Data Description

The Boston data frame has 506 rows and 14 columns. This data frame contains the following columns:

CRIM -
per capita crime rate by town.

ZN -
proportion of residential land zoned for lots over 25,000 sq.ft.

INDUS -
proportion of non-retail business acres per town.

CHAS - 
Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).

NOX - 
nitrogen oxides concentration (parts per 10 million).

RM - 
average number of rooms per dwelling.

AGE - 
proportion of owner-occupied units built prior to 1940.

DIS - 
weighted mean of distances to five Boston employment centres.

RAD - 
index of accessibility to radial highways.

TAX - 
full-value property-tax rate per \$10,000.

PTRATIO - 
pupil-teacher ratio by town.

B -
1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town.

LSTAT - 
lower status of the population (percent).

TARGET - 
median value of owner-occupied homes in \$1000s.
## Data Preprocessing
In this step we first check there are any null and duplicate values are present or not, If present we have to handel them, after that we check there are any incorrect data present or not, If present then we have to handel them
after that we check outliers and If present handel them

![Data Preprocessing](https://github.com/vishalkrishna90/BOSTON-HOUSE/blob/623b5600cd3e2ac471ce64114e9b6b9d51907dd8/Images/Data_preprocessing.png)

## Exploratory Data Analysis (EDA)
In this step we try to Analyze our data very efficiently and deeply, first we check correlation between features
, then check feature distribution and then relation between features and target by graph

![Feature Correlation](https://github.com/vishalkrishna90/BOSTON-HOUSE/blob/babf095a4d682498aeac34bbbbb5467e298c91b5/Images/EDA.png)
![Relation Between Features And Target](https://github.com/vishalkrishna90/BOSTON-HOUSE/blob/babf095a4d682498aeac34bbbbb5467e298c91b5/Images/EDA_2.png)

## Feature Selection

In this step first we split features and target, then split data into train
and test data

```
# split data into features and target
X  = df.drop('TARGET',axis = 1)
y = df['TARGET']
```

```
# import train test split to split train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3, random_state=42)
```

## Data Scaling

In this step we scale or features train and test data 

```
# import standard scaler to scale data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

## Model Building
In this step we build different different model and check there performance
and then choose best model for the dataset

![Feature Correlation](https://github.com/vishalkrishna90/BOSTON-HOUSE/blob/babf095a4d682498aeac34bbbbb5467e298c91b5/Images/Model_Building.png)
![Relation Between Features And Target](https://github.com/vishalkrishna90/BOSTON-HOUSE/blob/babf095a4d682498aeac34bbbbb5467e298c91b5/Images/Model_Buinding_2.png)

## Model Performances & Feature Importance
After model Building we check there performance by r2 score and the model
whose r2 score is higher we consider that model to be our final model
and then we check important features based on that model 

![Model Performances](https://github.com/vishalkrishna90/BOSTON-HOUSE/blob/babf095a4d682498aeac34bbbbb5467e298c91b5/Images/Models_Score.png)
![Important Features](https://github.com/vishalkrishna90/BOSTON-HOUSE/blob/babf095a4d682498aeac34bbbbb5467e298c91b5/Images/Feature_Importance_Xgb.png)


## Rebuild Model With Important Features

After getting feature Importance we create new dataframe by Important features
then do data spliting and data scaling and then rebuild our model by 
important features

```
new_df = df[['LSTAT','RM','DIS','NOX','PTRATIO','CRIM','TARGET']]
```

```
X  = new_df.drop('TARGET',axis = 1)
y = new_df['TARGET']
```

```
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3, random_state=42)
```

```
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

## Final Score By The Best Model

![Final Score](https://github.com/vishalkrishna90/BOSTON-HOUSE/blob/17f37f16d97dcca25eddf96758e9b34702bff1eb/Images/Final_Score.png)

## Make Pickle File
After getting final score from the best model we make a Pickle file for web app

```
import pickle as pkl
pkl.dump(grid_xgb, (open('xgb_model.pkl','wb')))
```

## Create New Enviornment
After making pickle file we create new virtual environment for the 
project and install required libraries and create web app in Coding IDE

```
conda create -p bostonhouse python==3.9 -y
```

```
pip install streamlit numpy pandas sklearn xgboost
``` 

## Create Web App With Streamlit
After installing all required libraries and dependencies we create web app 

![Web App](https://github.com/vishalkrishna90/BOSTON-HOUSE-PRICE-PREDICTION/blob/main/Images/Web_App_1.png)

## Upload All Files In Github repository

After creating web app we upload all files in github repository by git cli 
```
git config --global user.name "FIRST_NAME LAST_NAME"
```

```
git config --global user.email "myemail@gmail.com"
```

```
git add files_name
```

```
git commit -m  "about the commit"
```

```
git push origin main
```

## Deploy Model On Heroku

At the end we deploy our model on heroku, so that anybody can use the web app

[Boston House Price Prediction Web App](https://chennaihousepricepredict.herokuapp.com/)

![Boston House Price Precictin](https://github.com/vishalkrishna90/BOSTON-HOUSE-PRICE-PREDICTION/blob/main/Images/Web_App_2.png)
## Deployment Requirement Tools 

![Deploy](https://github.com/vishalkrishna90/BOSTON-HOUSE-PRICE-PREDICTION/blob/main/Images/st_im.png)

 - [Streamlit](https://streamlit.io/)
 - [Github Account](https://github.com/)
 - [Heroku Account](https://dashboard.heroku.com/apps)
 - [Visual Studio Code](https://code.visualstudio.com/)
 - [Git CLI](https://git-scm.com/book/en/v2/Getting-Started-The-Command-Line)


