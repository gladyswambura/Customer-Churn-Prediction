# data cleaning tasks
import pandas as pd

def handle_missing_values(df):
    mean_age = df['Age'].mean()
    df['Age'].fillna(mean_age, inplace=True)
    return df

def encode_categorical_variables(df):
    df = pd.get_dummies(df, columns=['Gender', 'Married', 'Internet Service'], drop_first=True)
    return df

# Load the dataset
df = pd.read_csv('data/dataset.csv')

# Perform data cleaning
df = handle_missing_values(df)
df = encode_categorical_variables(df)

# Save the cleaned data
df.to_csv('data/cleaned_dataset.csv', index=False)
