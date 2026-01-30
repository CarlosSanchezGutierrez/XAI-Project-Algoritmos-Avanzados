cat > src/explain.py <<'EOF'
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.linear_model import LinearRegression
from .dag import topo_order, parents

def fit_local_causal_models(g: nx.DiGraph, train: pd.DataFrame):
    models = {}
    order = topo_order(g)
    for node in order:
        ps = parents(g, node)
        if len(ps) == 0:
            models[node] = None  # raíz: no modelo condicional
        else:
            X = train[ps].values
            y = train[node].values
            lr = LinearRegression().fit(X, y)
            models[node] = (ps, lr)
    return models

def residual_scores(g: nx.DiGraph, models, df: pd.DataFrame, t: int):
    # score por nodo: |residual| normalizado (aprox)
    scores = {}
    for node in df.columns:
        m = models.get(node, None)
        if m is None:
            # raíz: score = |x - median(train)| se maneja fuera si hace falta
            scores[node] = abs(df.loc[t, node])
        else:
            ps, lr = m
            x = df.loc[t, ps].values.reshape(1, -1)
            pred = lr.predict(x)[0]
            res = df.loc[t, node] - pred
            scores[node] = abs(res)
    return scores

def candidate_explanation_sets(g: nx.DiGraph, top_nodes):
    # Para cada candidato, explicamos como "rutas" desde raíces a ese nodo
    roots = [n for n in g.nodes if g.in_degree(n) == 0]
    cand_sets = []

    for v in top_nodes:
        # tomar una ruta más corta desde cualquier raíz (si existe)
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
    # target_nodes: nodos que queremos "cubrir" (p.ej top suspicious)
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
    return expl_nodes, covered

def fidelity_score(expl_nodes, target_nodes, alpha: float):
    # fidelidad simple: cobertura de target_nodes penalizada por tamaño
    target_nodes = set(target_nodes)
    if len(target_nodes) == 0:
        return 0.0
    coverage = len(expl_nodes & target_nodes) / len(target_nodes)
    size_penalty = 1.0 / (1.0 + len(expl_nodes))
    return alpha * coverage + (1 - alpha) * size_penalty

def metaheuristic_search(cand_sets, target_nodes, max_nodes: int, alpha: float, rng: np.random.Generator):
    # “Evolutivo” simple: hill-climbing con mutación sobre selección de candidatos
    # Representación: conjunto de índices de cand_sets activados
    m = len(cand_sets)
    if m == 0:
        return set(), 0.0

    def decode(sel):
        expl = set()
        for i in sel:
            expl |= cand_sets[i][1]
        # limitar por max_nodes: si se pasa, recorta por heurística
        if len(expl) > max_nodes:
            # recorte: quedarse con nodos más frecuentes en sets seleccionados
            freq = {}
            for i in sel:
                for n in cand_sets[i][1]:
                    freq[n] = freq.get(n, 0) + 1
            expl = set(sorted(expl, key=lambda n: -freq[n])[:max_nodes])
        return expl

    # init: greedy-like start (selecciona algunos al azar sesgado a los primeros)
    sel = set(rng.choice(np.arange(m), size=min(2, m), replace=False))
    best_expl = decode(sel)
    best = fidelity_score(best_expl, target_nodes, alpha)

    for _ in range(250):  # iteraciones pequeñas
        cand = set(sel)
        # mutación: toggle 1 índice
        i = int(rng.integers(0, m))
        if i in cand:
            cand.remove(i)
        else:
            cand.add(i)

        expl = decode(cand)
        score = fidelity_score(expl, target_nodes, alpha)

        # aceptar si mejora o con pequeña prob (simula exploración)
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
    scores = residual_scores(g, models, df, t_event)

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_nodes = [n for n, _ in ranked[:topk]]

    cand_sets = candidate_explanation_sets(g, top_nodes)
    target_nodes = list(dict.fromkeys(suspicious_nodes + top_nodes))  # único, preserva orden

    # baseline greedy
    greedy_expl, covered = greedy_min_expl(cand_sets, target_nodes, max_expl_nodes)
    greedy_fid = fidelity_score(greedy_expl, target_nodes, alpha)

    if use_metaheuristic:
        mh_expl, mh_fid = metaheuristic_search(cand_sets, target_nodes, max_expl_nodes, alpha, rng)
        if mh_fid >= greedy_fid:
            expl_nodes = mh_expl
            fid = mh_fid
            method = "metaheuristic"
        else:
            expl_nodes = greedy_expl
            fid = greedy_fid
            method = "greedy"
    else:
        expl_nodes = greedy_expl
        fid = greedy_fid
        method = "greedy"

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
EOF
