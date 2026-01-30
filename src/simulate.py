cat > src/simulate.py <<'EOF'
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
    nodes = order[:]  # ya strings
    W = {}  # pesos por nodo: dict parent->weight
    b = {}  # bias por nodo

    for node in nodes:
        ps = parents(g, node)
        W[node] = {p: rng.normal(0.0, 0.9) for p in ps}
        b[node] = rng.normal(0.0, 0.2)

    X = {node: np.zeros(n_steps, dtype=float) for node in nodes}

    for t in range(n_steps):
        for node in nodes:
            val = b[node]
            for p, w in W[node].items():
                val += w * X[p][t]
            val += rng.normal(0.0, noise_std)
            X[node][t] = val

    df = pd.DataFrame({node: X[node] for node in nodes})
    return df

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
        # drift desde t0 hasta el final
        drift = np.linspace(0.0, strength, df2.shape[0] - t0)
        df2.loc[t0:, target] += drift
    else:
        raise ValueError(f"Unknown anomaly kind: {kind}")

    return df2, target
EOF
