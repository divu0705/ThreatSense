"""
Data Preprocessing Module for DDoS Detection System
Handles data loading, cleaning, and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        
    def load_data(self, filepath):
        """Load the CICIDS2017 dataset"""
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    
    def clean_data(self, df):
        """Clean the dataset by handling missing values and infinities"""
        print("Cleaning data...")
        
        # Remove leading/trailing whitespace from column names
        df.columns = df.columns.str.strip()
        
        # Replace infinity values with NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Fill NaN values with median for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isna().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        # Remove duplicate rows
        initial_shape = df.shape[0]
        df.drop_duplicates(inplace=True)
        print(f"Removed {initial_shape - df.shape[0]} duplicate rows")
        
        return df
    
    def encode_labels(self, df, label_column='Label'):
        """Encode target labels"""
        print("Encoding labels...")
        
        # Handle various label formats (BENIGN vs DDoS)
        if label_column in df.columns:
            # Convert to binary: BENIGN = 0, DDoS/Attack = 1
            df['Label_Encoded'] = df[label_column].apply(
                lambda x: 0 if 'BENIGN' in str(x).upper() else 1
            )
            print(f"Label distribution:\n{df['Label_Encoded'].value_counts()}")
            return df
        else:
            raise ValueError(f"Label column '{label_column}' not found in dataset")
    
    def feature_engineering(self, df):
        """Select and engineer features"""
        print("Engineering features...")
        
        # Remove non-numeric and target columns
        exclude_cols = ['Label', 'Label_Encoded', 'Timestamp', 'Flow ID', 
                       'Source IP', 'Destination IP', 'Source Port', 'Destination Port']
        
        # Get numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        self.feature_columns = feature_cols
        print(f"Selected {len(feature_cols)} features")
        
        return df[feature_cols], df['Label_Encoded']
    
    def scale_features(self, X_train, X_test):
        """Scale features using StandardScaler"""
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def preprocess(self, filepath, test_size=0.2, random_state=42):
        """Complete preprocessing pipeline"""
        # Load data
        df = self.load_data(filepath)
        
        # Clean data
        df = self.clean_data(df)
        
        # Encode labels
        df = self.encode_labels(df)
        
        # Feature engineering
        X, y = self.feature_engineering(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        print("\nPreprocessing complete!")
        print(f"Training set size: {X_train_scaled.shape}")
        print(f"Test set size: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def save_preprocessor(self, filepath='models/preprocessor.pkl'):
        """Save the preprocessor for later use"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns
        }, filepath)
        print(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath='models/preprocessor.pkl'):
        """Load a saved preprocessor"""
        data = joblib.load(filepath)
        self.scaler = data['scaler']
        self.label_encoder = data['label_encoder']
        self.feature_columns = data['feature_columns']
        print(f"Preprocessor loaded from {filepath}")


if __name__ == "__main__":
    print("Preprocessing module ready to use!")
