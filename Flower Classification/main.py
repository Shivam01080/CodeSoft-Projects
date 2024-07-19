import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, classification_report
from sklearn.tree import DecisionTreeClassifier

# Reading the dataset

df = pd.read_csv(r"C:\Users\HP\Downloads\archive\IRIS.csv")
print(df.head())

# Information about the dataframe
print(df.shape)
print(df.info())
print(df.describe())

# Checking for duplicate values

print("\nDuplicated Sum(Before): ", df.duplicated().sum())

# Dropping the duplicate rows

df.drop_duplicates(inplace=True)
print("\nDuplicated Sum(After): ", df.duplicated().sum())

# EDA

df['species'].value_counts().plot(kind='bar', color='g')
plt.show()

plt.pie(df['species'].value_counts(), labels=df['species'].value_counts().index, autopct="%1.2f%%")
plt.title("Different Species")
plt.show()

plt.figure(figsize=(10, 6))
sns.swarmplot(x="species", y="sepal_length", data=df)
plt.title("Swarm Plot of Sepal Length by Iris Species")
plt.xlabel("Species")
plt.ylabel("Sepal Length (cm)")
plt.show()

sns.scatterplot(x='sepal_length', y='sepal_width', hue='species', data=df)
plt.title("Scatter plot of sepal length and sepal width of different species")
plt.show()

sns.scatterplot(x='petal_length', y='petal_width', hue='species', data=df)
plt.title("Scatter plot of Petal length and Petal width of different species")
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='sepal_length', bins=25, kde=True, hue='species')
plt.title("Distribution of Sepal Length")
plt.show()

plt.figure(figsize=(8, 4))
sns.histplot(data=df, x='sepal_width', bins=25, kde=True, hue='species')
plt.title("Distribution of Sepal Width")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
sns.histplot(data=df, x='petal_length', bins=25, kde=True, hue='species')
plt.title("Distribution of Petal Length")
plt.show()

plt.figure(figsize=(8, 4))
sns.histplot(data=df, x='petal_width', bins=25, kde=True, hue='species')
plt.title("Distribution of Petal Width")
plt.tight_layout()
plt.show()

sns.pairplot(df, hue='species')
plt.show()

corr = df.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(corr, annot=True, ax=ax, cmap='coolwarm')
plt.show()

# Machine learning
# Checking for outlier

numerical_data = [col for col in df.columns if df.dtypes[col] != 'object']
plt.figure(figsize=(12, 10))
for i in range(len(numerical_data)):
    plt.subplot(len(numerical_data), 1, i + 1)
    sns.boxplot(x=numerical_data[i], y='species', data=df)
    plt.title(f'Boxplot for {numerical_data[i]}')
plt.tight_layout()
plt.show()

# Label encoding
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])
df.head()

x = df.drop('species', axis=1)
y = df['species']
print(x, y, sep='\n')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=1)

# Model building
# Logistic regression

lr = LogisticRegression()
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)
accuracy = accuracy_score(y_pred, y_test)
print("Accuracy score(Logistic Regression):", accuracy * 100)

cls_report = classification_report(y_pred, y_test)
print('classification report for our model(Logistic Regression) is:\n', cls_report)

# Decision tree classifier

dc = DecisionTreeClassifier()
dc.fit(x_train,y_train)
y_pred_dc = dc.predict(x_test)
accuracy_dc = accuracy_score(y_pred_dc, y_test)
print("Accuracy score(Decision Tree classifier):",accuracy_dc*100)
cls_report_dc = classification_report(y_pred_dc, y_test)
print('classification report for our model(Decision tree classifier) is:\n', cls_report_dc)

models = ['Logistic Regression', 'Decision Tree Classifier']

# Accuracy values

acc = [accuracy*100,accuracy_dc*100]

# Create the bar plot

plt.figure(figsize=(6, 5))
plt.bar(models, acc, color=['blue', 'green'])
plt.ylim(90, 100)
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Comparison of Machine Learning Models')
plt.show()

