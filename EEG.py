import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
import pickle

warnings.filterwarnings("ignore")

# Load your EEG dataset
data = pd.read_csv("acquiredDataset.csv")

# Extract features and target variable
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Create a Logistic Regression model
log_reg = LogisticRegression()

# Train the model
log_reg.fit(X_train, y_train)

# Save the model using pickle
pickle.dump(log_reg, open('model.pkl', 'wb'))

# Load the model
model = pickle.load(open('model.pkl', 'rb'))
