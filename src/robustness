cat > src/robustness.py <<'EOF'
import numpy as np
import pandas as pd

def add_noise(df: pd.DataFrame, sigma: float, rng: np.random.Generator):
    if sigma <= 0:
        return df.copy()
    return df + rng.normal(0.0, sigma, size=df.shape)

def apply_missing(df: pd.DataFrame, rate: float, rng: np.random.Generator):
    if rate <= 0:
        return df.copy()
    out = df.copy()
    mask = rng.random(size=df.shape) < rate
    out.values[mask] = np.nan
    # imputación simple para baseline: forward fill + mean
    out = out.ffill().bfill()
    out = out.fillna(out.mean())
    return out

def ema_smooth_scores(score_series: pd.DataFrame, alpha: float):
    # score_series: (t x nodes) de scores tipo z/residual
    return score_series.ewm(alpha=alpha, adjust=False).mean()

def jaccard(a: set, b: set):
    if len(a) == 0 and len(b) == 0:
        return 1.0
    return len(a & b) / len(a | b)
EOF
