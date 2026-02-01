This repository provides a minimal, runnable implementation of Adaptive Bandit-Based Anomaly Ranking for selective audit scheduling in Edge Data Integrity Verification. 

# Files
- `abar.py`    : ABAR core algorithm (anomaly score, reward learning, UCB index, global budget allocation)
- `simulate.py`: fully runnable synthetic experiment (no dataset required)
- `run_csv.py` : template runner for CSV datasets (e.g., QWS-mapped inputs)

# Install
```bash
pip install -r requirements.txt
