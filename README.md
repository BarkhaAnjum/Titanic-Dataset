from google.colab import files
uploaded = files.upload() 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Titanic-Dataset.csv/Titanic-Dataset.csv")
print(df.shape)
df.head()

print(df.info())
print("\nMissing values:\n", df.isnull().sum())

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
if 'Cabin' in df.columns:
    df = df.drop(columns=['Cabin'])

    df = pd.get_dummies(df, columns=['Sex','Embarked'], drop_first=True)

    scaler = StandardScaler()
num_cols = ['Age','Fare']
df[num_cols] = scaler.fit_transform(df[num_cols])


print("Cleaned dataset shape:", df.shape)
df.head(10)
print("\nColumn info after preprocessing:")
print(df.info())

