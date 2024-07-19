import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv(r"C:\Users\HP\Downloads\archive\Titanic-Dataset.csv")
df.head()
df.info()
df.isnull().sum()

# Dropping unnecessary columns

df.drop(columns="Cabin", axis=1, inplace=True)
# Filling the missing values of the age column by Mean value of the column

df.fillna({'Age': df['Age'].mean()}, inplace=True)

# Filling the missing values of the Embarked column by Mode value of the column

df.fillna({'Embarked': df['Embarked'].mode()[0]}, inplace=True)
print(df.isnull().sum())

# Checking for duplicate values in the dataset

print('DUPLICATE VALUES: ', df.duplicated().sum())

# Checking the survival of people

print("PEOPLE WHO SURVIVED: ", df['Survived'].value_counts())

# Visualization of survival of people

# Count-plot

sns.countplot(x='Survived', hue='Survived', data=df)
plt.xlabel("Survival status")
plt.ylabel("Number of people")
plt.xticks(ticks=[0, 1], labels=['Not survived', 'survived'])
plt.show()

# Pie chart

plt.pie(df['Survived'].value_counts(), explode=[0, 0.04], autopct="%1.2f%%", labels=['Not survived', 'Survived'])
plt.title("Survival of people")
plt.show()

# Visualization of people survived from different passenger class

sns.countplot(x='Pclass', hue='Pclass', data=df)
plt.xlabel("Pclass")
plt.ylabel("Number of people")
plt.show()

sns.countplot(x='Survived', hue='Pclass', data=df)
plt.xlabel("Survival status")
plt.ylabel("Number of people")
plt.xticks(ticks=[0, 1], labels=['Not survived', 'survived'])
plt.show()

sns.catplot(x='Pclass', hue='Survived', col='Sex', kind='count', data=df)
plt.tight_layout()
plt.show()

# Visualization of people survived from different gender

print(df['Sex'].unique())

# Visualizing the population of male and female passenger

sns.countplot(x='Sex', hue='Sex', data=df)
plt.xlabel("Gender")
plt.ylabel("Number of people")
plt.show()

sns.countplot(x='Survived', hue='Sex', data=df)
plt.xlabel("Survival status")
plt.ylabel("Number of people")
plt.xticks(ticks=[0, 1], labels=['Not survived', 'survived'])
plt.show()

df[df['Sex'] == 'male'].Survived.groupby(df.Survived).count().plot(kind='pie',
                                                                   figsize=(3, 6), explode=[0, 0.05], autopct='%1.1f%%',
                                                                   labels=["Not survived", "Survived"])
plt.ylabel("")
plt.title("Male survival rate")
plt.show()

df[df['Sex'] == 'female'].Survived.groupby(df.Survived).count().plot(kind='pie',
                                                                     figsize=(3, 6), explode=[0, 0.05],
                                                                     autopct='%1.1f%%',
                                                                     labels=["Not survived", "Survived"])
plt.ylabel("")
plt.title("Female survival rate")
plt.show()

# visualizing the population of different passenger class

sns.countplot(x='Pclass', hue='Pclass', data=df)
plt.xlabel("Pclass")
plt.ylabel("Number of people")
plt.show()

# Visualization of people survived from different passenger class

sns.countplot(x='Survived', hue='Pclass', data=df)
plt.xlabel("Survival status")
plt.ylabel("Number of people")
plt.xticks(ticks=[0, 1], labels=['Not survived', 'survived'])
plt.show()

sns.catplot(x='Pclass', hue='Survived', col='Sex', kind='count', data=df)
plt.tight_layout()
plt.show()

# Visualization of people survived from different Embarkment

sns.countplot(x='Embarked', hue='Embarked', data=df)
plt.xlabel("Embarked")
plt.ylabel("Number of people")
plt.show()

sns.countplot(x='Survived', hue='Embarked', data=df)
plt.xlabel("Survival status")
plt.ylabel("Number of people")
plt.xticks(ticks=[0, 1], labels=['Not survived', 'survived'])
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
sns.countplot(x='SibSp', hue='SibSp', data=df, ax=axes[0])
sns.countplot(x='Parch', hue='Parch', data=df, ax=axes[1])
plt.show()

sns.countplot(x='Survived', hue='SibSp', data=df)
plt.xticks(ticks=[0, 1], labels=['Not survived', 'survived'])
plt.xlabel("Survival status")
plt.ylabel("Number of people")
plt.title("Survival population of Sibsp")
plt.show()

sns.countplot(x='Survived', hue='Parch', data=df)
plt.xticks(ticks=[0, 1], labels=['Not survived', 'survived'])
plt.title("Survival population of Parch")
plt.xlabel("Survival status")
plt.ylabel("Number of people")
plt.show()

# Distribution of Fare and age

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
sns.histplot(df['Fare'], kde=True, ax=axes[0])
sns.histplot(df['Age'].dropna(), kde=True, ax=axes[1])
plt.show()

sns.histplot(x='Fare', hue='Survived', data=df, kde=True)
plt.legend(labels=['survived', 'not survived'])
plt.show()

sns.histplot(x='Age', hue='Survived', data=df, kde=True)
plt.legend(labels=['survived', 'not survived'])
plt.show()

print(df)

# Changing the Sex column and Embarked column from categorical to numerical for model training

l_encoder = LabelEncoder()
df['Sex'] = l_encoder.fit_transform(df['Sex'])
df.replace({'Embarked': {'S': 0, 'C': 1, 'Q': 2}}, inplace=True)
print(df.head())
print(df.info())

df_num = df[['Fare', 'Parch', 'SibSp', 'Age', 'Sex', 'Pclass', 'Embarked', 'Survived']]
print(df_num)

sns.heatmap(df_num.corr(), annot=True)
plt.show()

# Model Prediction
# Features

x = df_num.drop(columns=(['Parch', 'SibSp', 'Survived']))
print(x)

# Target

y = df['Survived']
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
model = LogisticRegression()
model.fit(x_train, y_train)
x_train_prediction = model.predict(x_train)
print(x_train_prediction)

train_data_accuracy = accuracy_score(y_train, x_train_prediction)
print("Accuracy Score of training data: ", train_data_accuracy)

x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(y_test, x_test_prediction)
print("Accuracy score of testing data:", test_data_accuracy)
