import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

def run_eda_and_preprocessing(file_path):
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    
    # 1. Basic Info
    print("\nDataset Info:")
    print(df.info())
    
    # 2. Handle missing values (if any)
    df = df.dropna()
    
    # 3. Exploratory Data Analysis
    os.makedirs('static/plots', exist_ok=True)
    
    # Target distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Price_Category', data=df, palette='viridis')
    plt.title('Distribution of Price Categories')
    plt.savefig('static/plots/price_category_dist.png')
    plt.close()
    
    # Correlation for numeric features
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
    plt.title('Numeric Feature Correlations')
    plt.savefig('static/plots/correlation_matrix.png')
    plt.close()

    # 4. Feature Selection
    # Price_USD is likely directly related to Price_Category, so we might want to exclude it 
    # if we want the model to learn from physical features. However, the requirement is 
    # "end to end to classify car price classification dataset".
    # Often, Car_ID is irrelevant.
    features = ['Brand', 'Manufacture_Year', 'Body_Type', 'Fuel_Type', 'Transmission', 
                'Engine_CC', 'Horsepower', 'Mileage_km_per_l', 'Price_USD', 'Manufacturing_Country', 
                'Car_Age', 'HP_per_CC', 'Age_Category', 'Efficiency_Score']
    target = 'Price_Category'
    
    X = df[features]
    y = df[target]
    
    # 5. Encoding Categorical Features
    categorical_cols = X.select_dtypes(include=['object']).columns
    encoders = {}
    X = X.copy() # Avoid SettingWithCopyWarning
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
    
    # Target encoding
    target_le = LabelEncoder()
    y = target_le.fit_transform(y)
    encoders['target'] = target_le
    
    # 6. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 7. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 8. Save artifacts
    joblib.dump(encoders, 'encoders.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    
    # Save processed data for training
    processed_data = {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': features
    }
    joblib.dump(processed_data, 'processed_data.joblib')
    
    print("\nEDA and Preprocessing completed.")
    print(f"Artifacts saved: encoders.joblib, scaler.joblib, processed_data.joblib")
    print(f"Plots saved in static/plots/")

if __name__ == "__main__":
    run_eda_and_preprocessing('global_cars_enhanced.csv')
