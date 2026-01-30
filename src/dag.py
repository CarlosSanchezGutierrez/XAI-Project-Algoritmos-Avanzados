cat > src/dag.py <<'EOF'
import numpy as np
import networkx as nx

def make_random_dag(n_nodes: int, edge_prob: float, rng: np.random.Generator) -> nx.DiGraph:
    # Para garantizar aciclicidad: solo permitimos edges i -> j con i < j (según un orden aleatorio)
    perm = rng.permutation(n_nodes)
    inv = np.empty(n_nodes, dtype=int)
    inv[perm] = np.arange(n_nodes)

    g = nx.DiGraph()
    g.add_nodes_from(range(n_nodes))

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < edge_prob:
                u = perm[i]
                v = perm[j]
                g.add_edge(u, v)

    # Renombrar nodos a strings X0..Xn-1
    mapping = {i: f"X{i}" for i in range(n_nodes)}
    g = nx.relabel_nodes(g, mapping)
    return g

def topo_order(g: nx.DiGraph):
    return list(nx.topological_sort(g))

def parents(g: nx.DiGraph, node: str):
    return list(g.predecessors(node))
EOF
