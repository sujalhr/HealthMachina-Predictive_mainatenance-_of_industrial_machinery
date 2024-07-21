import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

data = pd.read_csv("hackproj\predictive_maintenance.csv")

# Drop unnecessary columns
df = data.drop(["UDI", "Product ID", "Target"], axis=1)

# Initialize encoders
lE = LabelEncoder()
oh = OneHotEncoder()

# Apply OneHotEncoder and LabelEncoder
l1 = oh.fit_transform(df["Type"].values.reshape(-1, 1)).toarray()
l2 = lE.fit_transform(df["Failure Type"]).reshape(-1, 1)

# Create DataFrames for the encoded features
ohdf = pd.DataFrame(l1, columns=oh.get_feature_names_out(["Type"]))
ledf = pd.DataFrame(l2, columns=["Failure_Type"])

# Concatenate the new DataFrames with the original DataFrame
df = pd.concat([df, ohdf, ledf], axis=1)

# Drop original columns that were encoded
df = df.drop(["Type", "Failure Type"], axis=1)

# Define target and features
y = df['Failure_Type']
X = df.drop(['Failure_Type'], axis=1)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Initialize the model
fmodel = RandomForestClassifier()

# Train the model
fmodel.fit(X_train, y_train)

# Evaluate the model on the validation set
val_accuracy = fmodel.score(X_val, y_val)
print(f'Validation Accuracy: {val_accuracy:.2f}')

# Evaluate the model on the test set
test_accuracy = fmodel.score(X_test, y_test)
print(f'Test Accuracy: {test_accuracy:.2f}')

joblib.dump(fmodel, "wmodel.pkl")