# defining functions for building and training my ml model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Loading the preprocessed data here because the training data is already  loaded 
df = pd.read_csv('data/preprocessed_dataset.csv')

# Spliting the data into features (X) and target (y)
X = df.drop(columns=['Churn Category'])
y = df['Churn Category']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train a machine learning model (ihave used Logistic Regression here for simplicity)
model = LogisticRegression()
model.fit(X_train, y_train)

# my goal was to Make predictions so i use this function
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(report)
