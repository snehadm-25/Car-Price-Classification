import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def train_and_evaluate():
    print("Loading processed data...")
    data = joblib.load('processed_data.joblib')
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    feature_names = data['feature_names']
    
    encoders = joblib.load('encoders.joblib')
    target_le = encoders['target']
    class_names = target_le.classes_

    # Using Decision Tree for perfect separation of threshold-based categories
    print("\nTraining Decision Tree...")
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    print("Decision Tree Results:")
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=class_names))

    print(f"\nSaving model with {acc*100:.2f}% accuracy")
    joblib.dump(model, 'car_price_model.joblib')
    
    # Feature Importance
    plt.figure(figsize=(10, 8))
    importances = model.feature_importances_
    feat_importances = pd.Series(importances, index=feature_names)
    feat_importances.sort_values().plot(kind='barh', color='teal')
    plt.title('Feature Importance (Decision Tree)')
    plt.tight_layout()
    plt.savefig('static/plots/feature_importance.png')
    plt.close()

    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, model.predict(X_test))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Decision Tree)')
    plt.savefig('static/plots/confusion_matrix.png')
    plt.close()

if __name__ == "__main__":
    train_and_evaluate()
