cat > src/detect.py <<'EOF'
import numpy as np
import pandas as pd

def fit_zscore_baseline(train: pd.DataFrame):
    mu = train.mean(axis=0)
    sig = train.std(axis=0).replace(0.0, 1e-9)
    return mu, sig

def detect_event(df: pd.DataFrame, mu, sig, threshold: float, min_nodes_flagged: int):
    z = (df - mu) / sig
    flagged = (z.abs() >= threshold)

    # primer tiempo donde hay al menos min_nodes_flagged nodos disparados
    counts = flagged.sum(axis=1)
    idx = np.where(counts.values >= min_nodes_flagged)[0]
    if len(idx) == 0:
        return None, [], z

    t_event = int(idx[0])
    suspicious_nodes = list(flagged.columns[flagged.iloc[t_event].values])
    return t_event, suspicious_nodes, z
EOF
