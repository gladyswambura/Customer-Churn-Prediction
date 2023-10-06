# I use this script for defining functions and automating data preprocessing. it allows me to execute the data preprocessing pipeline
import pandas as pd
from sklearn.preprocessing import StandardScaler

def handle_missing_values(df):
    mean_age = df['Age'].mean()
    df['Age'].fillna(mean_age, inplace=True)
    return df

def encode_categorical_variables(df):
    df = pd.get_dummies(df, columns=['Gender', 'Married', 'Internet Service'], drop_first=True)
    return df

def scale_features(df):
    scaler = StandardScaler()
    df[['Age', 'Avg Monthly Long Distance Charges']] = scaler.fit_transform(df[['Age', 'Avg Monthly Long Distance Charges']])
    return df

def main():
    # Load the dataset
    df = pd.read_csv('data/dataset.csv')

    # Perform data preprocessing
    df = handle_missing_values(df)
    df = encode_categorical_variables(df)
    df = scale_features(df)

    # Save the preprocessed data
    df.to_csv('data/preprocessed_dataset.csv', index=False)

if __name__ == "__main__":
    main()
