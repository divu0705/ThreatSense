"""
Hybrid Model Training - Random Forest + XGBoost (REQUIRED)
This implements the hybrid ensemble model combining RF and XGBoost
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
from preprocessing import DataPreprocessor

class HybridRFXGBoost:
    """Hybrid ensemble model combining Random Forest and XGBoost with weighted voting"""
    
    def __init__(self, rf_weight=0.5, xgb_weight=0.5):
        self.rf_weight = rf_weight
        self.xgb_weight = xgb_weight
        
        self.rf_model = RandomForestClassifier(
            n_estimators=150, max_depth=25, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1, verbose=0
        )
        
        self.xgb_model = XGBClassifier(
            n_estimators=150, max_depth=15, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            n_jobs=-1, eval_metric='logloss', verbosity=0
        )
        
        print(f"Hybrid RF+XGBoost Model Initialized (RF: {rf_weight}, XGB: {xgb_weight})")
    
    def fit(self, X_train, y_train):
        """Train both Random Forest and XGBoost models"""
        print("\n" + "="*60)
        print("TRAINING HYBRID MODEL (RF + XGBOOST)")
        print("="*60)
        
        print("\n[1/2] Training Random Forest...")
        start = time.time()
        self.rf_model.fit(X_train, y_train)
        print(f"Random Forest trained in {time.time()-start:.2f}s")
        
        print("\n[2/2] Training XGBoost...")
        start = time.time()
        self.xgb_model.fit(X_train, y_train)
        print(f"XGBoost trained in {time.time()-start:.2f}s")
        
        return self
    
    def predict_proba(self, X):
        """Get weighted probability predictions"""
        rf_proba = self.rf_model.predict_proba(X)
        xgb_proba = self.xgb_model.predict_proba(X)
        return self.rf_weight * rf_proba + self.xgb_weight * xgb_proba
    
    def predict(self, X):
        """Get class predictions"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def evaluate(self, X_test, y_test):
        """Evaluate the hybrid model"""
        print("\n" + "="*60)
        print("EVALUATING HYBRID MODEL")
        print("="*60)
        
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'y_test': y_test
        }
        
        print(f"\nAccuracy:  {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall:    {results['recall']:.4f}")
        print(f"F1-Score:  {results['f1_score']:.4f}")
        print(f"ROC-AUC:   {results['roc_auc']:.4f}")
        print(f"\nConfusion Matrix:\n{results['confusion_matrix']}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Benign', 'DDoS']))
        
        return results
    
    def save_model(self, filepath='models/hybrid_model.pkl'):
        """Save the trained hybrid model"""
        joblib.dump({
            'rf_model': self.rf_model,
            'xgb_model': self.xgb_model,
            'rf_weight': self.rf_weight,
            'xgb_weight': self.xgb_weight
        }, filepath)
        print(f"\nHybrid model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath='models/hybrid_model.pkl'):
        """Load a trained hybrid model"""
        data = joblib.load(filepath)
        instance = cls(rf_weight=data['rf_weight'], xgb_weight=data['xgb_weight'])
        instance.rf_model = data['rf_model']
        instance.xgb_model = data['xgb_model']
        print(f"Hybrid model loaded from {filepath}")
        return instance

def plot_results(results, save_path='models/hybrid_model_results.png'):
    """Plot evaluation results"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Confusion Matrix
    sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=axes[0],
               xticklabels=['Benign', 'DDoS'], yticklabels=['Benign', 'DDoS'])
    axes[0].set_title('Confusion Matrix')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    # Metrics Bar Chart
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    values = [results['accuracy'], results['precision'], results['recall'], 
              results['f1_score'], results['roc_auc']]
    axes[1].bar(metrics, values, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12'])
    axes[1].set_ylim([0, 1.1])
    axes[1].set_ylabel('Score')
    axes[1].set_title('Performance Metrics')
    axes[1].grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(values):
        axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Results plot saved to {save_path}")
    plt.close()

def main():
    """Main training pipeline"""
    print("\n" + "="*60)
    print("HYBRID RF+XGBOOST DDOS DETECTION SYSTEM")
    print("="*60)
    
    preprocessor = DataPreprocessor()
    dataset_path = 'data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
    
    try:
        X_train, X_test, y_train, y_test = preprocessor.preprocess(dataset_path)
        preprocessor.save_preprocessor()
        
        hybrid_model = HybridRFXGBoost(rf_weight=0.5, xgb_weight=0.5)
        hybrid_model.fit(X_train, y_train)
        results = hybrid_model.evaluate(X_test, y_test)
        plot_results(results)
        hybrid_model.save_model()
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        
    except FileNotFoundError:
        print(f"\n❌ ERROR: Dataset not found at {dataset_path}")
        print("Please download CICIDS2017 dataset and place in data/ folder")

if __name__ == "__main__":
    main()
