import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

df = pd.read_csv(r"C:\Users\HP\Downloads\archive\advertising.csv")

# Dataset Details

print(df.head(), '\n')
print(df.info(), '\n')
print(df.describe(), '\n')

# Checking for null values

print(df.isnull().sum(), '\n')

# Checking for Duplicate Values

print("Duplicate elements: ", df.duplicated().sum(), '\n')

# Variables distribution

for i in df.columns:
    plt.title("Distribution of Features")
    sns.histplot(df[i], kde=True, bins=20, color='b')
    plt.xlabel(i)
    plt.ylabel('Frequency')
    plt.show()

# Correlation between data

print(df.corr())
sns.heatmap(df.corr(), annot=True)
plt.title('Correlation')
plt.show()

# Money spent on each

tot = [df['TV'].sum(), df['Newspaper'].sum(), df['Radio'].sum()]
x_axis = ['TV', 'Newspaper', 'Radio']

sns.barplot(x=x_axis, y=tot)
plt.xlabel('Advertisement Type')
plt.ylabel('Total Money Spent')
plt.title('Total Money Spent on Each Type of Advertisement')
plt.show()

df['Total'] = df['TV'] + df['Newspaper'] + df['Radio']
sns.scatterplot(x=df['Total'], y=df['Sales'])
plt.title("Scatter PLot")
plt.show()

# Outliers

sns.boxplot(df)
plt.show()

# Train test split

y = df['Sales']
x = df[['TV', 'Newspaper', 'Radio']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=11)

# Model building

lr = LinearRegression()
lr.fit(x_train, y_train)

y_train_pre = lr.predict(x_train)
y_test_pre = lr.predict(x_test)

# Model Evaluation

print("\nMean absolute error of train data:", mean_absolute_error(y_train, y_train_pre))
print("\nMean absolute error of test data:", mean_absolute_error(y_test, y_test_pre))

print("\nMean squared error of train data:", mean_squared_error(y_train, y_train_pre))
print("\nMean squared error of test data:", mean_squared_error(y_test, y_test_pre))

print("\nr2 score(Train data):", r2_score(y_train, y_train_pre))
print("\nr2 score(Test data):", r2_score(y_test, y_test_pre))
