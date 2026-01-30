import numpy as np
import pandas as pd

def add_noise(df: pd.DataFrame, sigma: float, rng: np.random.Generator) -> pd.DataFrame:
    if sigma <= 0:
        return df.copy()
    arr = df.to_numpy(copy=True)
    arr = arr + rng.normal(0.0, sigma, size=arr.shape)
    return pd.DataFrame(arr, columns=df.columns, index=df.index)

def apply_missing(df: pd.DataFrame, rate: float, rng: np.random.Generator) -> pd.DataFrame:
    if rate <= 0:
        return df.copy()
<<<<<<< HEAD

    # numpy copy writable
=======
>>>>>>> 72a1283 (Make repo runnable (fix src + config))
    arr = df.to_numpy(copy=True)
    mask = rng.random(size=arr.shape) < rate
    arr[mask] = np.nan

    out = pd.DataFrame(arr, columns=df.columns, index=df.index)
<<<<<<< HEAD

=======
    # imputación simple para baseline: forward fill + mean
>>>>>>> 72a1283 (Make repo runnable (fix src + config))
    out = out.ffill().bfill()
    out = out.fillna(out.mean())
    return out

def ema_smooth_scores(score_series: pd.DataFrame, alpha: float) -> pd.DataFrame:
    return score_series.ewm(alpha=alpha, adjust=False).mean()

def jaccard(a: set, b: set) -> float:
    if len(a) == 0 and len(b) == 0:
        return 1.0
    return len(a & b) / len(a | b)
