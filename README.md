# ThreatSense

Unified Streamlit dashboard combining:
- **NIDS:** NSL-KDD + Random Forest intrusion-category detection.
- **DDoS:** CICIDS2017 + Hybrid Random Forest/XGBoost detection.

## Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The NIDS module is ready immediately because its trained artifacts are included.

## Activate DDoS module

The original DDoS repository did not include the CICIDS2017 CSV or trained model artifacts. Download the CSV used by the original project and run:

```bash
python ddos/train_ddos.py "path/to/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
```

This creates:
- `ddos/models/hybrid_model.pkl`
- `ddos/models/preprocessor.pkl`

Restart Streamlit, then use the DDoS tab to upload CICIDS2017-compatible CSV data for prediction.
