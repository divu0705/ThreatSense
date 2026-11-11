"""
Model Comparison Module - Evaluates multiple machine learning models for DDoS detection
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import time
import joblib

class ModelComparator:
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def initialize_models(self):
        """Initialize different ML models for comparison"""
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1),
            'XGBoost': XGBClassifier(n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42, n_jobs=-1, eval_metric='logloss'),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=100, random_state=42, early_stopping=True)
        }
        print("Initialized models:", list(self.models.keys()))
    
    def train_and_evaluate(self, model_name, model, X_train, X_test, y_train, y_test):
        """Train and evaluate a single model"""
        print(f"\n{'='*60}\nTraining {model_name}...\n{'='*60}")
        
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        self.results[model_name] = {
            'model': model,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.0,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'training_time': training_time
        }
        
        r = self.results[model_name]
        print(f"Accuracy: {r['accuracy']:.4f}, F1: {r['f1_score']:.4f}, Time: {r['training_time']:.2f}s")
        return r
    
    def compare_all_models(self, X_train, X_test, y_train, y_test):
        """Train and compare all models"""
        self.initialize_models()
        for model_name, model in self.models.items():
            self.train_and_evaluate(model_name, model, X_train, X_test, y_train, y_test)
        return self.results
    
    def plot_comparison(self, save_path='models/model_comparison.png'):
        """Plot comparison of all models"""
        metrics_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Accuracy': [r['accuracy'] for r in self.results.values()],
            'F1-Score': [r['f1_score'] for r in self.results.values()]
        })
        
        fig, ax = plt.subplots(figsize=(12, 6))
        metrics_df.set_index('Model')[['Accuracy', 'F1-Score']].plot(kind='bar', ax=ax, rot=45)
        ax.set_title('Model Performance Comparison')
        ax.set_ylabel('Score')
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"Comparison plot saved to {save_path}")
        plt.close()
    
    def save_comparison_report(self, save_path='models/comparison_report.txt'):
        """Save detailed comparison report"""
        with open(save_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("DDOS DETECTION - MODEL COMPARISON REPORT\n")
            f.write("="*80 + "\n\n")
            for model_name, r in self.results.items():
                f.write(f"{model_name}\n" + "-"*80 + "\n")
                f.write(f"Accuracy: {r['accuracy']:.4f}\nF1-Score: {r['f1_score']:.4f}\n\n")
        print(f"Report saved to {save_path}")

if __name__ == "__main__":
    print("Model Comparison done!")
