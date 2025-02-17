import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("IMDb Movies India.csv")
print(df.head())

# Drop irrelevant columns (adjust based on dataset structure)
df = df[['Title', 'Genre', 'Director', 'Actors', 'Rating']]

# Handle missing values
df.dropna(inplace=True)

# Feature Engineering - Convert categorical data into numerical
encoder = LabelEncoder()
df['Director'] = encoder.fit_transform(df['Director'])

# Process Genre (Multi-Label Encoding using CountVectorizer)
vectorizer = CountVectorizer(tokenizer=lambda x: x.split(','))
genre_encoded = vectorizer.fit_transform(df['Genre']).toarray()
genre_df = pd.DataFrame(genre_encoded, columns=vectorizer.get_feature_names_out())

df = pd.concat([df, genre_df], axis=1)
df.drop(columns=['Genre'], inplace=True)

# Process Actors (Taking only the first actor for simplicity)
df['First_Actor'] = df['Actors'].apply(lambda x: x.split(',')[0])
df['First_Actor'] = encoder.fit_transform(df['First_Actor'])
df.drop(columns=['Actors'], inplace=True)

# Define features and target variable
X = df.drop(columns=['Title', 'Rating'])
y = df['Rating']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")
