import numpy as np
import pandas as pd
import networkx as nx
from .dag import topo_order, parents

def simulate_linear_gaussian(
    g: nx.DiGraph,
    n_steps: int,
    noise_std: float,
    rng: np.random.Generator
) -> pd.DataFrame:
    order = topo_order(g)

    # pesos por nodo (padre->peso) y bias
    W, b = {}, {}
    for node in order:
        ps = parents(g, node)
        W[node] = {p: rng.normal(0.0, 0.9) for p in ps}
        b[node] = rng.normal(0.0, 0.2)

    X = {node: np.zeros(n_steps, dtype=float) for node in order}

    for t in range(n_steps):
        for node in order:
            val = b[node]
            for p, w in W[node].items():
                val += w * X[p][t]
            val += rng.normal(0.0, noise_std)
            X[node][t] = val

    return pd.DataFrame({node: X[node] for node in order})

def inject_anomaly(df: pd.DataFrame, t0: int, kind: str, strength: float, target: str, rng: np.random.Generator):
    nodes = list(df.columns)
    if target == "auto":
        target = rng.choice(nodes)

    df2 = df.copy()
    if kind == "spike":
        df2.loc[t0, target] += strength
    elif kind == "drop":
        df2.loc[t0, target] -= strength
    elif kind == "drift":
        drift = np.linspace(0.0, strength, df2.shape[0] - t0)
        df2.loc[t0:, target] += drift
    else:
        raise ValueError(f"Unknown anomaly kind: {kind}")

    return df2, target
