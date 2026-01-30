import numpy as np
import pandas as pd
import networkx as nx
from sklearn.linear_model import LinearRegression
from .dag import topo_order, parents

def fit_local_causal_models(g: nx.DiGraph, train: pd.DataFrame):
    models = {}
    for node in topo_order(g):
        ps = parents(g, node)
        if len(ps) == 0:
            models[node] = None  # raíz
        else:
            X = train[ps].values
            y = train[node].values
            lr = LinearRegression().fit(X, y)
            models[node] = (ps, lr)
    return models

def residual_scores(models, df: pd.DataFrame, t: int):
    scores = {}
    for node in df.columns:
        m = models.get(node, None)
        if m is None:
            scores[node] = abs(df.loc[t, node])
        else:
            ps, lr = m
            x = df.loc[t, ps].values.reshape(1, -1)
            pred = lr.predict(x)[0]
            scores[node] = abs(df.loc[t, node] - pred)
    return scores

def candidate_explanation_sets(g: nx.DiGraph, top_nodes):
    roots = [n for n in g.nodes if g.in_degree(n) == 0]
    cand_sets = []

    for v in top_nodes:
        best_path = None
        for r in roots:
            if nx.has_path(g, r, v):
                p = nx.shortest_path(g, r, v)
                if best_path is None or len(p) < len(best_path):
                    best_path = p

        if best_path is None:
            best_path = [v]

        cand_sets.append((v, set(best_path)))

    return cand_sets

def greedy_min_expl(cand_sets, target_nodes, max_nodes: int):
    covered = set()
    chosen = []
    all_targets = set(target_nodes)

    while covered != all_targets and len(chosen) < max_nodes:
        best = None
        best_gain = -1

        for name, s in cand_sets:
            gain = len((s & all_targets) - covered)
            if gain > best_gain:
                best_gain = gain
                best = (name, s)

        if best is None or best_gain <= 0:
            break

        chosen.append(best)
        covered |= (best[1] & all_targets)

    expl_nodes = set()
    for _, s in chosen:
        expl_nodes |= s

    # si excede max_nodes, recorta por frecuencia
    if len(expl_nodes) > max_nodes:
        freq = {}
        for _, s in chosen:
            for n in s:
                freq[n] = freq.get(n, 0) + 1
        expl_nodes = set(sorted(expl_nodes, key=lambda n: -freq[n])[:max_nodes])

    return expl_nodes

def fidelity_score(expl_nodes, target_nodes, alpha: float):
    target_nodes = set(target_nodes)
    if len(target_nodes) == 0:
        return 0.0

    coverage = len(expl_nodes & target_nodes) / len(target_nodes)
    size_penalty = 1.0 / (1.0 + len(expl_nodes))
    return alpha * coverage + (1 - alpha) * size_penalty

def metaheuristic_search(cand_sets, target_nodes, max_nodes: int, alpha: float, rng: np.random.Generator):
    m = len(cand_sets)
    if m == 0:
        return set(), 0.0

    def decode(sel):
        expl = set()
        for i in sel:
            expl |= cand_sets[i][1]

        if len(expl) > max_nodes:
            freq = {}
            for i in sel:
                for n in cand_sets[i][1]:
                    freq[n] = freq.get(n, 0) + 1
            expl = set(sorted(expl, key=lambda n: -freq[n])[:max_nodes])
        return expl

    sel = set(rng.choice(np.arange(m), size=min(2, m), replace=False))
    best_expl = decode(sel)
    best = fidelity_score(best_expl, target_nodes, alpha)

    for _ in range(250):
        cand = set(sel)
        i = int(rng.integers(0, m))
        if i in cand:
            cand.remove(i)
        else:
            cand.add(i)

        expl = decode(cand)
        score = fidelity_score(expl, target_nodes, alpha)

        if score > best or rng.random() < 0.05:
            sel = cand
            if score > best:
                best = score
                best_expl = expl

    return best_expl, best

def explain_event(
    g: nx.DiGraph,
    train: pd.DataFrame,
    df: pd.DataFrame,
    t_event: int,
    suspicious_nodes: list,
    topk: int,
    max_expl_nodes: int,
    alpha: float,
    use_metaheuristic: bool,
    rng: np.random.Generator
):
    models = fit_local_causal_models(g, train)
    scores = residual_scores(models, df, t_event)

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_nodes = [n for n, _ in ranked[:topk]]

    target_nodes = list(dict.fromkeys(suspicious_nodes + top_nodes))
    cand_sets = candidate_explanation_sets(g, top_nodes)

    greedy_nodes = greedy_min_expl(cand_sets, target_nodes, max_expl_nodes)
    greedy_fid = fidelity_score(greedy_nodes, target_nodes, alpha)

    method = "greedy"
    expl_nodes = greedy_nodes
    fid = greedy_fid

    if use_metaheuristic:
        mh_nodes, mh_fid = metaheuristic_search(cand_sets, target_nodes, max_expl_nodes, alpha, rng)
        if mh_fid >= greedy_fid:
            method, expl_nodes, fid = "metaheuristic", mh_nodes, mh_fid

    subg = g.subgraph(expl_nodes).copy()
    return {
        "t_event": t_event,
        "suspicious_nodes": suspicious_nodes,
        "top_nodes": top_nodes,
        "node_scores": ranked,
        "explanation_nodes": sorted(list(expl_nodes)),
        "explanation_edges": list(subg.edges()),
        "fidelity": float(fid),
        "method": method
    }
