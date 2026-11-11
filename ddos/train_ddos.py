"""Train the DDoS hybrid model for ThreatSense.
Usage: python ddos/train_ddos.py path/to/CICIDS2017.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import DataPreprocessor
from model_train import HybridRFXGBoost


def main(dataset_path):
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.preprocess(dataset_path)
    out_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(out_dir, exist_ok=True)
    preprocessor.save_preprocessor(os.path.join(out_dir, 'preprocessor.pkl'))
    model = HybridRFXGBoost(0.5, 0.5)
    model.fit(X_train, y_train)
    model.evaluate(X_test, y_test)
    model.save_model(os.path.join(out_dir, 'hybrid_model.pkl'))
    print('ThreatSense DDoS model training complete.')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python ddos/train_ddos.py <path_to_CICIDS2017_csv>')
        raise SystemExit(1)
    main(sys.argv[1])
