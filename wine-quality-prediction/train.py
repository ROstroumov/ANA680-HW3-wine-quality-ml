
import pandas as pd
import numpy as np
import argparse
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    args = parser.parse_args()
    
    print("Loading training data from:", args.train)
    
    # Load data from SageMaker input channel
    train_data_path = os.path.join(args.train, 'train.csv')
    train_data = pd.read_csv(train_data_path)
    
    print(f"Data shape: {train_data.shape}")
    print("Columns:", train_data.columns.tolist())
    
    # Simple wine quality prediction
    if 'quality' in train_data.columns:
        X = train_data.drop('quality', axis=1)
        y = train_data['quality']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("Training Random Forest model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Model accuracy: {accuracy:.4f}")
        
        # Save model
        model_path = os.path.join(args.model_dir, 'model.joblib')
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
    else:
        print("Error: 'quality' column not found in data")
        print("Available columns:", train_data.columns.tolist())

if __name__ == "__main__":
    main()
